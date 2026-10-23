# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Score an online evaluation's clarifying questions, after the rollouts are done.

The rollouts and the judging are separate jobs on purpose. A rollout holds a GPU for
the policy, a browser, and a served user simulator; it must not also depend on the ask
judge being up, because :func:`evaluation.simulated_user.ask_targets` raises when the
judge's identity is unset and the evaluator files the resulting rollout as a harness
error — losing the one behaviour the evaluation exists to measure.

So ``scripts/eval_cascp_online.py`` records every question beside its scores file and
exits. This pass reads that file, puts the questions to the judge with several in
flight at once, and writes the ask-quality block. It needs no GPU of its own, it can
run while the next evaluation occupies the cluster, and it can be re-run under a
different judge without paying for the rollouts again.

Usage::

    python scripts/score_online_ask_quality.py \\
        --tasks   <scores-dir>/ask_tasks.json \\
        --out     <scores-dir>/ask_quality.json \\
        --workers 8

The judge's identity comes from the environment: ``ASK_JUDGE_URL`` and
``ASK_JUDGE_MODEL``, which ``harness/run_online.sh`` sets. The judge kind is ``set``,
the only one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from evaluation import simulated_user
from evaluation.online_ask_quality import judge_unavailable, read_tasks, score
from evaluation.serve_preflight import assert_endpoint_live


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="ask_tasks.json, written beside the run's scores file",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="questions in flight at once; each is one judge round trip",
    )
    args = ap.parse_args(argv)

    # Refused up front rather than reported as an unscored block. The rollouts are
    # right to carry on without a judge; this job is the judging, so a missing one
    # is a failed job. A zero-filled block here would be read as a measurement.
    unavailable = judge_unavailable()
    if unavailable is not None:
        print(f"ERROR: {unavailable}", file=sys.stderr)
        return 1
    if not args.tasks.exists():
        print(
            f"ERROR: {args.tasks} does not exist. The run writes it beside its scores "
            f"file; a run that predates this pass has none.",
            file=sys.stderr,
        )
        return 1

    assert_endpoint_live(
        endpoint=os.environ["ASK_JUDGE_URL"], model=os.environ["ASK_JUDGE_MODEL"]
    )
    tasks = read_tasks(args.tasks)
    block = score(tasks, workers=args.workers)
    # The identity travels with the number. A rate is not comparable across runs
    # without the judge that produced it, and this pass runs long after the job
    # whose environment held it.
    block["judge"] = {
        "model": os.environ["ASK_JUDGE_MODEL"],
        "kind": simulated_user.ASK_JUDGE_KIND,
    }
    args.out.write_text(json.dumps(block, indent=2))
    report = block.get("report", {})
    # Every figure the ratios were computed over, so the line can be reconciled
    # without opening the file. `n_asks` counts what the policy sent; the ratios
    # are over what the judge read as a question. `ask_recall` was printed here
    # and is not a field of the report -- the name is `blocker_recall`, so the
    # line read "recall None" on every run.
    print(
        f"{block['n_tasks']} tasks | {block['n_asks']} messages from "
        f"{block['n_tasks_that_asked']} tasks "
        f"({block.get('n_statements', 0)} were reports, not questions; "
        f"{block.get('n_asks_scored', 0)} scored) | "
        f"precision {report.get('ask_precision')} "
        f"blocker recall {report.get('blocker_recall')} | "
        f"{block.get('n_judge_failures', 0)} judge failures -> {args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
