# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Freeze the ICL4 condition from prepared demonstrations; CPU only, no model calls.

Copies into ``--out/assets/``: the demonstration files written by ``ambigui.icl.prepare``
(``demo_messages.json``, ``demonstrations.json``, ``demo_source_records.json``,
``system_prompt.txt``, ``tools.json``) and the policy checkpoint's nine processor/tokenizer/template
files (read by the headroom guard). Each of the nine files must have the sha256 the config pins for
the policy's Hugging Face revision (``processor_files_sha256``); a checkpoint folder with other
files is refused. Writes ``--out/condition.json``: every asset's sha256, the served model name and
roles from the run config, the demonstration selection rule (approved training sources, the
qualification gates, the sha256 of the task file the demonstrations were checked against) and a
fingerprint over all of it. Pass ``--out`` to ``run_online.sh`` as ``ICL_CONDITION_DIR``.

usage: python -m ambigui.icl.prepare_condition --config configs/icl/q38_27b_matched84.json \
         --demos-dir runs/icl_demos --checkpoint weights/qwen3.8-27b-base-online --out runs/icl_condition
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .condition import CONDITION, canonical, digest, sha
from .demos import SOURCE_RULE

QUALIFICATION_KEYS = ('require_completed_clear_demos', 'exclude_secret_request_demos', 'preserve_original_named_argument_demos')

PROCESSOR_NAMES = ['chat_template.jinja', 'config.json', 'generation_config.json', 'preprocessor_config.json',
                   'tokenizer_config.json', 'video_preprocessor_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--demos-dir', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True, help='local policy checkpoint directory (processor files)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    root = args.out
    assets = {}

    def copy_source(p, name):
        p = Path(p)
        b = p.read_bytes()
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)  # local byte copy ONLY
        if sha(dest.read_bytes()) != sha(b):
            raise ValueError('copy differs from its source: ' + name)
        assets[name] = sha(b)
        return b
    expected_processor = cfg.get('processor_files_sha256') or {}
    if sorted(expected_processor) != sorted(PROCESSOR_NAMES):
        raise ValueError('config must pin processor_files_sha256 for exactly these files: ' + ', '.join(PROCESSOR_NAMES))
    for name in PROCESSOR_NAMES:
        if sha((args.checkpoint / name).read_bytes()) != expected_processor[name]:
            raise ValueError(f'{args.checkpoint / name} is not the file of {cfg["model"]}@{cfg["model_revision"]} (sha256 differs)')
    for name in ['demo_messages.json', 'system_prompt.txt', 'tools.json', 'demonstrations.json', 'demo_source_records.json']:
        copy_source(args.demos_dir / name, 'assets/' + name)
    original_demos = json.loads((root / 'assets/demonstrations.json').read_bytes())
    if digest({k: v for k, v in original_demos.items() if k != 'fingerprint'}) != original_demos['fingerprint']:
        raise ValueError('demonstrations.json fingerprint differs from its content')
    if original_demos.get('source_rule') != SOURCE_RULE:
        raise ValueError('demonstrations were not selected with the approved-source rule; run ambigui.icl.prepare again')
    qualification = {k: cfg.get(k) for k in QUALIFICATION_KEYS}
    if original_demos.get('qualification_predicates') != qualification:
        raise ValueError('demonstration qualification gates differ from the config')
    processor_sha = {}
    for name in PROCESSOR_NAMES:
        processor_sha[name] = sha(copy_source(args.checkpoint / name, 'assets/processor/' + name))
    messages = json.loads((root / 'assets/demo_messages.json').read_bytes())
    system = (root / 'assets/system_prompt.txt').read_bytes().decode()
    tools = json.loads((root / 'assets/tools.json').read_bytes())
    if digest(messages) != original_demos['demo_messages_sha256']:
        raise ValueError('demo_messages.json differs from demonstrations.json')
    prefix = [{'role': 'system', 'content': system}, *messages]
    prefix_path = root / 'assets/prompt_prefix.json'
    prefix_path.write_bytes(canonical(prefix))
    assets['assets/prompt_prefix.json'] = sha(prefix_path.read_bytes())
    processor_descriptor = {'version': 'exact_processor_file_map_v1', 'files_sha256': processor_sha}
    descriptor = {
        'version': 'frozen_base_fixed4_icl_runtime_v1', 'prompt_condition': CONDITION, 'served_model': cfg['served_model'],
        'policy': {'hf_repo': cfg['model'], 'hf_revision': cfg['model_revision']},
        'demo_fingerprint': original_demos['fingerprint'], 'canonical_message_digest': digest(messages),
        'demo_source_rule': original_demos['source_rule'], 'demo_qualification': qualification,
        'demo_eval_tasks_sha256': original_demos['eval_tasks_sha256'],
        'demonstration_sources': [d['source'] for d in original_demos['demonstrations']],
        'system_sha256': sha(system.encode()), 'canonical_tools_sha256': digest(tools),
        'prompt_prefix_sha256': assets['assets/prompt_prefix.json'],
        'prompt_prefix_definition': 'canonical JSON of the unchanged p0 system message plus EXACT eight frozen text role/content messages; NOT a query-specific rendered prompt',
        'processor_descriptor': processor_descriptor, 'processor_fingerprint': digest(processor_descriptor),
        'assets_sha256': assets,
        'budgets': {'context': 65536, 'max_tokens': 8192, 'image_budget': 4, 'temperature': 0.0, 'request_seed': 0,
                    'per_row_caps': "unchanged: each row's max_steps; no global override"},
        'roles': {'simulator': cfg['answerer_model'], 'judge': cfg['judge_model'], 'judge_kind': 'lookup',
                  'describer': cfg['judge_model'], 'ask_judge_kind': 'set'},
        'recording': {'required': True, 'wrapper_order': 'OUTER exact demo injection -> INNER lossless recorder -> strict headroom guard -> original create',
                      'expanded_request_recorded': True, 'added_native_tool_calls': False, 'browse_code_rewrites': False},
    }
    descriptor['condition_fingerprint'] = digest(descriptor)
    (root / 'condition.json').write_text(json.dumps(descriptor, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'prepared', 'condition_fingerprint': descriptor['condition_fingerprint'],
                      'demo_fingerprint': descriptor['demo_fingerprint'],
                      'canonical_message_digest': descriptor['canonical_message_digest'],
                      'prompt_prefix_sha256': descriptor['prompt_prefix_sha256'],
                      'processor_fingerprint': descriptor['processor_fingerprint'], 'inference_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
