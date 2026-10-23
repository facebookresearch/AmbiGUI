# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Phase 1 of the ICL arm for one shard: the unchanged harness runner with the ICL4 condition.

Called by ``harness/run_online.sh`` in place of ``harness/run_eval.py`` when the arm is
``icl4``, with the same evaluator arguments plus ``--condition-dir`` and ``--icl-output-root``.
It checks the frozen condition and the arguments, re-checks the demonstrations against the task
file of this run (``condition.verify_for_run``), writes the shard's condition identity, then
installs, in this order, the strict headroom guard (innermost), the lossless recorder and the
demonstration injection (outermost), and runs ``harness/run_eval.py`` in this process. The
recorder therefore journals the EXPANDED request the server receives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / 'harness'
SRC = HERE / 'src'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--condition-dir', type=Path, required=True, help='folder written by ambigui.icl.prepare_condition')
    ap.add_argument('--icl-output-root', type=Path, required=True, help='the arm folder that holds shard<i>/')
    flags, args = ap.parse_known_args(argv)
    for p in reversed([HARNESS, SRC]):
        sys.path.insert(0, str(p))
    from ambigui.icl.condition import (ContractError, FrozenCondition, flag_value, identity, install_outer_injection,
                                       validate_activation_environment, validate_eval_arguments, verify_for_run)
    try:
        condition = FrozenCondition(flags.condition_dir / 'condition.json')
        validate_activation_environment(os.environ)
        condition.verify_runtime_sources()
        args, shard_dir = validate_eval_arguments(args, condition, flags.icl_output_root)
        tasks = Path(flag_value(args, '--eval-rollouts'))
        verify_for_run(condition, tasks)
        shard, total = map(int, flag_value(args, '--shard').split('/'))
    except ContractError as exc:
        print('ICL4 condition guard: ' + str(exc), file=sys.stderr)
        return 2
    shard_dir.mkdir(parents=True, exist_ok=True)
    meta = identity(condition, hashlib.sha256(tasks.read_bytes()).hexdigest(), total, shard)
    p = shard_dir / 'prompt_condition_identity.json'
    if p.exists() and json.loads(p.read_bytes()) != meta:
        print('ICL4 condition guard: existing shard output belongs to a different condition', file=sys.stderr)
        return 2
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    # An empty journal means a task that made zero policy calls (for example a blocked start page).
    (shard_dir / 'policy_calls.jsonl').touch(exist_ok=True)
    os.environ['CASCP_SHOT_DIR'] = str(shard_dir / 'frames')
    from ambigui.icl.headroom.strict_headroom import install_for_runtime
    import policy_call_recording
    install_for_runtime(condition)       # strict guard, INSIDE the recorder
    policy_call_recording.apply()        # lossless recorder
    install_outer_injection(condition)   # demonstrations, OUTSIDE: the recorder sees the expanded request
    os.chdir(SRC)
    sys.argv = [str(HARNESS / 'run_eval.py'), *args]
    runpy.run_path(str(HARNESS / 'run_eval.py'), run_name='__main__')
    return 0


if __name__ == '__main__':
    sys.exit(main())
