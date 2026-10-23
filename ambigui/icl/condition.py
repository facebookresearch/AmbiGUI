# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The frozen ICL4 condition: the one request change the ICL arm makes, and its guards.

The ICL arm runs the same online harness, served model, settings, simulated user and judges as
the zero-shot arm. The only difference is that every policy request gets the four fixed
demonstration messages (eight messages: user, assistant, x4) inserted right after the unchanged
system message. Nothing else in the request changes: not the system prompt, the tool schema, the
history, the screenshots, the token budget or the decoding settings. Requests to other models
(simulated user, judges) are never touched.

``condition.json`` (written by ``ambigui.icl.prepare_condition``) pins the demonstration
messages, system prompt, tool schema and processor files by sha256, and carries its own
fingerprint. ``FrozenCondition`` refuses a condition whose files or fingerprint changed, or whose
system prompt and tool schema are not the harness's own. ``verify_for_run`` re-checks the
demonstrations against the task file a run uses, before the ICL arm starts.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
from pathlib import Path

CONDITION = 'ICL4'


class ContractError(ValueError):
    pass


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def digest(x):
    return sha(canonical(x))


def image_count(x):
    if isinstance(x, dict):
        if x.get('type') in {'image_url', 'image', 'image_file'}:
            return 1
        return sum(image_count(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(image_count(v) for v in x)
    return 0


class FrozenCondition:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        self.root = self.path.parent
        m = json.loads(self.path.read_bytes())
        payload = {k: v for k, v in m.items() if k != 'condition_fingerprint'}
        if digest(payload) != m.get('condition_fingerprint'):
            raise ContractError('condition fingerprint mismatch')
        if m.get('prompt_condition') != CONDITION or not m.get('served_model'):
            raise ContractError('condition tag/model differs')
        for name, expected in m['assets_sha256'].items():
            p = self.root / name
            if not p.resolve().is_relative_to(self.root) or sha(p.read_bytes()) != expected:
                raise ContractError('frozen preparation asset differs: ' + name)
        self.manifest = m
        self.served_model = m['served_model']
        self.messages = json.loads((self.root / 'assets/demo_messages.json').read_bytes())
        self.system = (self.root / 'assets/system_prompt.txt').read_bytes().decode()
        self.tools = json.loads((self.root / 'assets/tools.json').read_bytes())
        if (sha(self.system.encode()) != m['system_sha256'] or digest(self.messages) != m['canonical_message_digest']
                or digest(self.tools) != m['canonical_tools_sha256']):
            raise ContractError('frozen system/messages/tools differ')
        if len(self.messages) != 8 or [x['role'] for x in self.messages] != ['user', 'assistant'] * 4:
            raise ContractError('fixed4 message shape/order differs')
        if any(set(x) != {'role', 'content'} or not isinstance(x['content'], str) for x in self.messages):
            raise ContractError('demos must remain original role/content TEXT, not native tool calls')
        if image_count(self.messages):
            raise ContractError('demo images are not admitted')
        self.fingerprint = m['condition_fingerprint']

    def verify_runtime_sources(self):
        """The condition's system prompt and tool schema must be the harness's own, byte for byte."""
        from training.cascp_common import CASCP_SYSTEM
        from evaluation.tool_contract import tool_json_schema
        if self.system != CASCP_SYSTEM or self.tools != [tool_json_schema()]:
            raise ContractError('condition system prompt/tool schema differ from the harness in ambigui/icl/src')

    def expanded_request(self, request):
        # Non-policy roles are never given demonstrations or altered transport.
        if request.get('model') != self.served_model:
            return request
        messages = request.get('messages')
        if not isinstance(messages, list) or len(messages) < 2:
            raise ContractError('policy request requires system and live query')
        if messages[0] != {'role': 'system', 'content': self.system}:
            raise ContractError('p0 system must be byte-identical; no ASK paragraph variant')
        if any(x.get('role') == 'system' for x in messages[1:]):
            raise ContractError('unexpected later system message')
        if not any(x.get('role') == 'user' for x in messages[1:]):
            raise ContractError('live user query missing')
        if messages[1:9] == self.messages:
            raise ContractError('request is already injected; refuse duplicate demonstrations')
        if request.get('tools') != self.tools or request.get('tool_choice') != 'none':
            raise ContractError('complete tool schema/protocol differs')
        if request.get('max_tokens') != 8192 or request.get('temperature') != 0 or request.get('seed') != 0:
            raise ContractError('matched 8192/temperature0/request-seed0 budget differs')
        if request.get('stream') is True:
            raise ContractError('streaming not admitted by lossless recorder')
        if image_count(messages) > 4:
            raise ContractError('existing image4 budget exceeded; never silently prune here')
        expanded = dict(request)
        expanded['messages'] = [copy.deepcopy(messages[0]), *copy.deepcopy(self.messages), *copy.deepcopy(messages[1:])]
        if image_count(expanded['messages']) != image_count(messages):
            raise ContractError('query image count changed')
        return expanded


def install_outer_injection(condition, completions_type=None):
    """Install AFTER the existing recorder, so it journals the EXPANDED request.

    The harness's run_eval can call recorder.apply again: wraps propagates its
    marker, so the inner recorder recognises itself and remains inside this hook.
    This process-local patch touches neither shared source nor the engine.
    """
    if completions_type is None:
        from openai.resources.chat.completions import Completions
        completions_type = Completions
    original = completions_type.create
    existing = getattr(original, '_ambigui_icl4_fingerprint', None)
    if existing is not None:
        if existing != condition.fingerprint:
            raise ContractError('another ICL4 condition is already installed')
        return original
    if not getattr(original, '_ambigui_recorded', False):
        raise ContractError('must install lossless recorder BEFORE outer demo injection')

    @functools.wraps(original)
    def create(self, *args, **kwargs):
        if kwargs.get('model') == condition.served_model and args:
            raise ContractError('matched policy request must use named API arguments')
        expanded = condition.expanded_request(kwargs)
        return original(self, *args, **expanded)

    create._ambigui_icl4_fingerprint = condition.fingerprint
    completions_type.create = create
    return create


def verify_for_run(condition, tasks_path):
    """Run ``demos.verify_frozen`` on the condition's demonstrations against the task file of this run.

    The task file must be the one the demonstrations were checked against when they were prepared
    (same sha256), and no demonstration may share an ID, task family or normalised instruction with
    any of its tasks. The demonstrations must come from approved training sources.
    """
    from .demos import verify_frozen
    m = condition.manifest
    if 'assets/demo_source_records.json' not in m.get('assets_sha256', {}):
        raise ContractError('condition has no demonstration source records; prepare it again with this code')
    if 'demo_qualification' not in m:
        raise ContractError('condition does not record the demonstration qualification gates; prepare it again with this code')
    manifest = json.loads((condition.root / 'assets/demonstrations.json').read_bytes())
    records = json.loads((condition.root / 'assets/demo_source_records.json').read_bytes())
    if manifest.get('fingerprint') != m.get('demo_fingerprint'):
        raise ContractError('demonstrations.json is not the one the condition was frozen with')
    try:
        verify_frozen(manifest, {'train': None, 'eval': Path(tasks_path)}, m['demo_qualification'], records)
    except ValueError as exc:
        raise ContractError('demonstration check against the run task file failed: ' + str(exc)) from exc
    return manifest


def validate_activation_environment(env):
    if env.get('ASK_PROMPT_VARIANT'):
        raise ContractError('do not use ASK_PROMPT_VARIANT; p0 is unchanged')
    if env.get('RECORD_POLICY_CALLS') != '1':
        raise ContractError('lossless policy recording is mandatory')
    if int(env.get('POLICY_MAX_CONTEXT', '65536')) != 65536:
        raise ContractError('matched context must remain65536')


def flag_value(argv, name, default=None):
    """Value of evaluator flag ``name`` in ``argv`` (``--flag value`` or ``--flag=value``)."""
    hits = [argv[i+1] for i, x in enumerate(argv[:-1]) if x == name]
    hits += [x.split('=', 1)[1] for x in argv if x.startswith(name + '=')]
    if len(hits) > 1:
        raise ContractError('duplicate evaluator flag: ' + name)
    return hits[0] if hits else default


def validate_eval_arguments(argv, condition, owned_output_root):
    # Keep the existing evaluation arguments, rejecting only unsupported changes.
    # Return them unchanged.
    def value(name, default=None):
        return flag_value(argv, name, default)
    roles = condition.manifest['roles']
    if value('--served-model') != condition.served_model:
        raise ContractError('serve the SAME frozen BASE model name, never an ICL alias')
    if value('--max-tokens') != '8192':
        raise ContractError('matched response budget differs')
    if value('--answerer-model') != roles['simulator']:
        raise ContractError('simulated user differs from the condition')
    if value('--judge-model') != roles['judge'] or value('--judge-kind') != roles['judge_kind']:
        raise ContractError('matched lookup task judge required')
    if value('--describer-model') != roles['describer']:
        raise ContractError('matched describer required')
    out = value('--out')
    if not out:
        raise ContractError('separate ICL4 score output is required')
    p = Path(out).resolve()
    boundary = Path(owned_output_root).resolve()
    if not p.is_relative_to(boundary) or len(p.relative_to(boundary).parts) < 2:
        raise ContractError('score/cache output must be <ICL4 output>/<shard>/scores.json')
    if p.name != 'scores.json':
        raise ContractError('preserve the existing shard score filename')
    return list(argv), p.parent


def identity(condition, tasks_sha256, n_shards, shard=None):
    """Condition identity stamped on every ICL4 shard, preflight record and merged output, so ICL4
    outputs can never be mixed with zero-shot outputs or with another condition's (cache key is
    the condition fingerprint and the shard index, never the served model alone)."""
    x = {'prompt_condition': CONDITION, 'condition_fingerprint': condition.fingerprint,
         'served_model': condition.served_model, 'system_sha256': condition.manifest['system_sha256'],
         'demo_fingerprint': condition.manifest['demo_fingerprint'],
         'canonical_message_digest': condition.manifest['canonical_message_digest'],
         'prompt_prefix_sha256': condition.manifest['prompt_prefix_sha256'],
         'benchmark_sha256': tasks_sha256, 'global_shards': int(n_shards), 'model_only_cache_key_forbidden': True}
    if shard is not None:
        x['global_shard'] = int(shard)
    return x


def main(argv=None):
    """``python -m ambigui.icl.condition <condition dir> <task file>``: load the condition and run
    ``verify_for_run`` (run_online.sh calls this before the ICL arm starts). Exit 2 on refusal."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print('usage: python -m ambigui.icl.condition <condition dir> <task file>', file=sys.stderr)
        return 2
    try:
        condition = FrozenCondition(Path(argv[0]) / 'condition.json')
        manifest = verify_for_run(condition, argv[1])
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print('ICL4 condition check: ' + f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({'status': 'ok', 'condition_fingerprint': condition.fingerprint,
                      'demonstration_sources': [d['source'] for d in manifest['demonstrations']],
                      'eval_tasks_sha256': manifest['eval_tasks_sha256']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
