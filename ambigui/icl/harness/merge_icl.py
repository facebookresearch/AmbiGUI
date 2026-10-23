#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Merge the ICL arm's shards with merge_shards.py, then stamp the condition identity on the outputs.

Refuses to merge if any shard was written under a different condition (mixed caches).
usage: merge_icl.py <arm folder> <tasks file> <condition dir>
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent


def main():
    pdir, tasks, condition_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
    sys.path.insert(0, str(HERE.parent / 'src'))
    from ambigui.icl.condition import FrozenCondition, identity
    c = FrozenCondition(condition_dir / 'condition.json')
    tasks_sha = hashlib.sha256(tasks.read_bytes()).hexdigest()
    n = sum(1 for line in tasks.read_text().splitlines() if line.strip())
    for sd in pdir.glob('shard[0-9]*'):
        index = int(sd.name.removeprefix('shard'))
        if (sd / 'scores.json').exists() and json.loads((sd / 'prompt_condition_identity.json').read_bytes()) != identity(c, tasks_sha, n, index):
            raise ValueError('mixed condition cache refused: ' + sd.name)
    sys.argv = [str(HERE / 'merge_shards.py'), str(pdir), str(tasks)]
    runpy.run_path(str(HERE / 'merge_shards.py'), run_name='__main__')
    meta = identity(c, tasks_sha, n)
    merged = pdir / 'merged'
    (merged / 'integration_identity.json').write_text(json.dumps(meta, indent=2) + '\n')
    summary = json.loads((merged / 'summary.json').read_bytes())
    summary.update(meta)
    (merged / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    results = [json.loads(x) for x in (merged / 'results.jsonl').read_text().splitlines() if x]
    with (merged / 'results.jsonl').open('w') as f:
        for r in results:
            f.write(json.dumps(dict(r, prompt_condition='ICL4', condition_fingerprint=meta['condition_fingerprint'])) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
