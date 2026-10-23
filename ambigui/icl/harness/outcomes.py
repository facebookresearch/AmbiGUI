#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Which recorded rollout endings are valid outcomes, and which shards a rerun must redo.

A rollout that ended because the policy server could not be reached is not a policy failure.
The harness records it as ``policy_unreachable``; records written before that ending existed
carry ``generation_error`` with the transport error in ``end_detail`` (connection error, timeout,
gateway error, or a proxy reporting a refused connection). Both are read as ``policy_unreachable``
here. A request the strict headroom guard refused (``ICL4_STRICT_HEADROOM_REJECTED``) stays a
``generation_error``, as documented.

usage: outcomes.py rerun <shard dir>   exit 0 if the shard must be run again, 1 if it is kept
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

POLICY_UNREACHABLE = "policy_unreachable"
#: Endings that are not evidence about the policy: set aside from every rate.
SET_ASIDE = ("env_blocked", POLICY_UNREACHABLE)
_OUTAGE = re.compile(r"^(APIConnectionError|APITimeoutError)\b|Connection refused|\[Errno 111\]"
                     r"|Error code: 50[234]\b")


def ending(row: dict) -> str | None:
    """The row's end reason, with recorded policy-server outages read as ``policy_unreachable``."""
    reason = row.get("end_reason")
    detail = str(row.get("end_detail") or "")
    if (reason == "generation_error" and "ICL4_STRICT_HEADROOM_REJECTED" not in detail
            and _OUTAGE.search(detail)):
        return POLICY_UNREACHABLE
    return reason


def shard_needs_rerun(shard: Path) -> bool:
    """Missing or failed shard, a shard that ended env_unreachable, or a policy-server outage."""
    try:
        status = json.loads((shard / "status.json").read_text())
        rows = json.loads((shard / "scores.json").read_text()).get("results") or []
    except (OSError, ValueError):
        return True
    if status.get("phase1_rc") != 0 or not rows:
        return True
    return any(ending(r) in ("env_unreachable", POLICY_UNREACHABLE) for r in rows)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "rerun":
        print(__doc__, file=sys.stderr)
        return 2
    return 0 if shard_needs_rerun(Path(argv[1])) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
