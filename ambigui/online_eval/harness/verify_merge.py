#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Qualify the serving merge mechanically, not by its evaluation score.

The source release is already training-qualified. Here require identical visual
weights and at least one actually changed language weight, finite model tensors,
and exact serving/template metadata. Record byte-level output identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def index(path):
    return json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']


def tensor_digest(tensor):
    # Byte view supports BF16 without lossy dtype conversion.
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(dtype=__import__('torch').uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--merged', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    base, merged, adapter = map(Path, (args.base, args.merged, args.adapter))
    bi, mi = index(base), index(merged)
    if set(bi) != set(mi):
        raise ValueError('merged checkpoint parameter set changed')
    handles = {}
    def tensor(root, mapping, key):
        name = str(root / mapping[key])
        if name not in handles:
            handles[name] = safe_open(name, framework='pt', device='cpu')
        return handles[name].get_tensor(key)
    visual = [key for key in bi if any(part in key.lower() for part in ('visual.', 'vision.', 'vision_tower.'))]
    if not visual:
        raise ValueError('could not identify vision tensors')
    for key in visual:
        if tensor_digest(tensor(base, bi, key)) != tensor_digest(tensor(merged, mi, key)):
            raise ValueError('frozen vision weight was modified')
    changed = []
    with safe_open(str(adapter / 'adapter_model.safetensors'), framework='pt', device='cpu') as source:
        keys = list(source.keys())
    targets = sorted({key.removeprefix('base_model.model.').split('.lora_')[0] + '.weight'
                      for key in keys if '.lora_' in key})
    missing = [key for key in targets if key not in bi]
    if missing:
        raise ValueError('adapter target is not a base parameter: ' + str(missing[:3]))
    for key in targets:
        if tensor_digest(tensor(base, bi, key)) != tensor_digest(tensor(merged, mi, key)):
            changed.append(key)
    if not changed:
        raise ValueError('adapter merge had no effect on any target tensor')
    for key in mi:
        value = tensor(merged, mi, key).reshape(-1)
        for chunk in value.split(1024 * 1024):
            if not torch.isfinite(chunk).all().item():
                raise ValueError('merged checkpoint contains nonfinite values')
    for name in ('chat_template.jinja', 'tokenizer_config.json', 'preprocessor_config.json'):
        if (base / name).exists() and sha(base / name) != sha(merged / name):
            # Processor save_pretrained may rewrite formatting. Compare parsed JSON;
            # the literal chat template is never permitted to change.
            if name.endswith('.json') and json.loads((base / name).read_text()) == json.loads((merged / name).read_text()):
                continue
            raise ValueError('serving/template metadata changed: ' + name)
    report = {'status': 'passed', 'vision_tensors_unchanged': len(visual),
              'adapter_target_tensors': len(targets), 'changed_target_tensors': len(changed),
              'changed_targets': changed, 'all_merged_tensors_finite': True,
              'merged_files_sha256': {p.name: sha(p) for p in sorted(merged.iterdir()) if p.is_file()},
              'base_config_sha256': sha(base / 'config.json'), 'adapter_sha256': sha(adapter / 'adapter_model.safetensors')}
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print('[adapter] merge verified: actual language-weight changes, frozen vision, finite weights and unchanged template', flush=True)


if __name__ == '__main__':
    main()
