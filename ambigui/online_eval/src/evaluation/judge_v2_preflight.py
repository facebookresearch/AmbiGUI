# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Matched-version judge-only preflight from explicitly supplied source evidence."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from evaluation.judge_v2_adapter import activation_from_args, exact_task_id, score_future_task
from evaluation.score_rollout_v2 import VERSION


def preflight_203(args: Any, *, transport=None) -> tuple[int, dict]:
    """Never contacts answerer/policies; this proves ONLY the chosen task judge route."""
    config, rows = activation_from_args(args)
    if args.judge_version != VERSION or not args.judge_trajectory:
        raise ValueError("203 preflight requires exact VERSION and --judge-trajectory")
    path = Path(args.judge_trajectory)
    trajectory = json.loads(path.read_bytes())
    task_id = exact_task_id(trajectory)
    # All locators come from the supplied source, never directory discovery.
    recorded = trajectory.get("shots")
    if not isinstance(recorded, list) or any(not isinstance(s, str) for s in recorded):
        raise ValueError("preflight_missing_original_shot_sequence")
    if "original_final_url" not in trajectory or not isinstance(trajectory["original_final_url"], str):
        raise ValueError("preflight_requires_explicit_original_final_url: empty if original caller omitted it")
    result = score_future_task(row=trajectory,
        shots=[str(path.parent / p) for p in recorded], actions=trajectory.get("actions"),
        config=config, curated_row=rows.get(task_id), journal_root=args.judge_journal_root,
        kind=args.judge_kind, judge_url=args.judge_url, judge_model=args.judge_model,
        describer_url=args.describer_url, describer_model=args.describer_model,
        final_url=trajectory["original_final_url"],
        original_task_time_evidence=trajectory.get("original_task_time_evidence"),
        population_metadata=trajectory.get("population_metadata"), transport=transport)
    out = {"scope": "task_judge_only_preflight", "version": VERSION, "task_id": task_id,
            "judge": {"verdict": result.success, "why": result.why, "version": result.version,
                     "journal_path": result.journal_path,
                     "source_error": result.source_error, "checks": result.checks},
           "other_dependency_roles_proved": False}
    # A genuine failure verdict proves the protocol; unresolved is not admitted as green.
    return (0 if type(result.success) is bool and result.source_error is None else 3), out
