#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The dependency preflight of the ICL arm: preflight_roles.py with the ICL4 condition installed.

The policy probe then sends exactly what the arm will send (demonstrations inserted, headroom
guard active) and is journaled losslessly under ``$CASCP_SHOT_DIR/..``. Its result is the same
JSON as preflight_roles.py; the condition identity is written beside it.
usage: preflight_icl.py <condition dir> <tasks file>
"""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))


def main():
    condition_dir, tasks = Path(sys.argv[1]), Path(sys.argv[2])
    from ambigui.icl.condition import (FrozenCondition, identity, install_outer_injection,
                                       validate_activation_environment, verify_for_run)
    from ambigui.icl.headroom.strict_headroom import install_for_runtime
    c = FrozenCondition(condition_dir / 'condition.json')
    validate_activation_environment(os.environ)
    c.verify_runtime_sources()
    verify_for_run(c, tasks)
    install_for_runtime(c)  # strict guard INSIDE the lossless recorder
    import policy_call_recording
    policy_call_recording.apply()
    install_outer_injection(c)
    import preflight_roles
    out = Path(os.environ['CASCP_SHOT_DIR']).parent
    out.mkdir(parents=True, exist_ok=True)
    n = sum(1 for line in tasks.read_text().splitlines() if line.strip())
    (out / 'integration_identity.json').write_text(
        json.dumps(identity(c, hashlib.sha256(tasks.read_bytes()).hexdigest(), n), indent=2) + '\n')
    return preflight_roles.main()


if __name__ == '__main__':
    sys.exit(main())
