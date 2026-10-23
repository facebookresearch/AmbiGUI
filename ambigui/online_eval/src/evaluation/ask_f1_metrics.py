# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Ask-F1 metrics for the RLVR eval, including the blocker-restricted precision.

The published ``ask_precision`` is ``relevant asks / ALL asks`` over the whole eval. That
denominator is mostly unjudgeable by construction: only a minority of eval tasks carry a
gold clarifying question, and on the rest the reward's relevance judge short-circuits to
``False`` without being called, because there is no reference to grade against. An ask
there can never be counted relevant.

Measured on the toy-200 eval: 36 of 192 tasks carry a gold blocker, so ~81% of asks are
unjudgeable and ``ask_precision`` is **capped at ≈0.19**. A checkpoint that asks less
raises it without asking any better. The metric tracks over-asking, not ask quality.

So this module reports both:

* ``ask_precision`` — unchanged, so every number already published stays comparable.
* ``ask_precision_on_blocker`` — the same ratio over asks on gold-blocker tasks only.
  This is the one to rank checkpoints and hyperparameters on.
* ``judgeable_ask_share`` — the fraction of asks that could be judged at all, i.e. the
  ceiling on ``ask_precision``. Reporting it stops the legacy figure being misread.

``ask_f1`` keeps its original definition and ``ask_f1_on_blocker`` mirrors it with the
restricted precision. Recall is unchanged: it is already macro-averaged over gold-blocker
tasks, so it never had this problem.

Pure: no file or network I/O. The caller parses the rollout dump and judges the asks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Rollout:
    """One rollout's ask counts: how many questions it asked, how many were relevant."""

    n_ask: int
    n_rel: int

    @property
    def blocker_found(self) -> bool:
        """The rollout hit the task's blocker, i.e. at least one ask was judged relevant."""
        return self.n_rel >= 1


@dataclass(frozen=True)
class AskF1Report:
    """Every figure the eval reports, legacy and restricted side by side."""

    n_rollouts: int
    n_tasks: int
    n_gold_blocker_tasks: int
    k: int
    total_asks: int
    relevant_asks: int
    asks_on_blocker_tasks: int
    ask_precision: float
    ask_precision_on_blocker: float
    judgeable_ask_share: float
    blocker_recall: float
    blocker_recall_passk: float
    ask_f1: float
    ask_f1_on_blocker: float
    avg_asks_per_rollout: float
    overask_rate_no_blocker: float

    def as_dict(self) -> dict[str, Any]:
        """Plain dict, for writing to a summary JSON file."""
        return asdict(self)


def _ratio(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def _harmonic(a: float, b: float) -> float:
    return _ratio(2 * a * b, a + b)


def _validate(per_task: Mapping[str, Sequence[Rollout]]) -> None:
    if not per_task:
        raise ValueError("summarise needs at least one task")
    for task_id, rollouts in per_task.items():
        if not rollouts:
            raise ValueError(f"task {task_id!r} has no rollouts")
        for r in rollouts:
            if r.n_ask < 0 or r.n_rel < 0:
                raise ValueError(f"task {task_id!r} has negative ask counts")
            if r.n_rel > r.n_ask:
                raise ValueError(
                    f"task {task_id!r}: n_rel ({r.n_rel}) exceeds n_ask ({r.n_ask})"
                )


def summarise(
    per_task: Mapping[str, Sequence[Rollout]], gold_ids: set[str]
) -> AskF1Report:
    """Aggregate per-task rollouts into the report.

    ``gold_ids`` are the task ids that carry a gold clarifying question. Ids in
    ``gold_ids`` with no rollouts in ``per_task`` are ignored — they were never run.
    """
    _validate(per_task)
    # A relevant ask on a task with no gold question is incoherent: the judge returns
    # False without a reference. Left unchecked it inflated the restricted precision.
    for tid, rs in per_task.items():
        if tid not in gold_ids and any(r.n_rel for r in rs):
            raise ValueError(f"task {tid!r} has a relevant ask but no gold question")
    gold_tasks = [tid for tid in per_task if tid in gold_ids]
    no_blocker = [tid for tid in per_task if tid not in gold_ids]

    total_asks = sum(r.n_ask for rs in per_task.values() for r in rs)
    relevant = sum(r.n_rel for rs in per_task.values() for r in rs)
    asks_on_blocker = sum(r.n_ask for tid in gold_tasks for r in per_task[tid])

    # Macro over gold-blocker tasks: per-task blocker-found rate, then averaged.
    recalls = [
        _ratio(sum(r.blocker_found for r in per_task[tid]), len(per_task[tid]))
        for tid in gold_tasks
    ]
    blocker_recall = _ratio(sum(recalls), len(recalls))
    passk = [any(r.blocker_found for r in per_task[tid]) for tid in gold_tasks]
    passk_recall = _ratio(sum(passk), len(passk))

    ask_precision = _ratio(relevant, total_asks)
    # The numerator must be restricted exactly as the denominator is. Summing relevant
    # asks over ALL tasks while dividing by asks on gold-blocker tasks lets this exceed
    # 1.0, and breaks the invariant that judgeable_ask_share caps ask_precision.
    relevant_on_blocker = sum(r.n_rel for tid in gold_tasks for r in per_task[tid])
    ask_precision_on_blocker = _ratio(relevant_on_blocker, asks_on_blocker)

    no_blocker_rollouts = [r for tid in no_blocker for r in per_task[tid]]
    n_rollouts = sum(len(rs) for rs in per_task.values())

    return AskF1Report(
        n_rollouts=n_rollouts,
        n_tasks=len(per_task),
        n_gold_blocker_tasks=len(gold_tasks),
        k=max(len(rs) for rs in per_task.values()),
        total_asks=total_asks,
        relevant_asks=relevant,
        asks_on_blocker_tasks=asks_on_blocker,
        ask_precision=ask_precision,
        ask_precision_on_blocker=ask_precision_on_blocker,
        judgeable_ask_share=_ratio(asks_on_blocker, total_asks),
        blocker_recall=blocker_recall,
        blocker_recall_passk=passk_recall,
        ask_f1=_harmonic(ask_precision, blocker_recall),
        ask_f1_on_blocker=_harmonic(ask_precision_on_blocker, blocker_recall),
        avg_asks_per_rollout=_ratio(total_asks, n_rollouts),
        overask_rate_no_blocker=_ratio(
            sum(1 for r in no_blocker_rollouts if r.n_ask >= 1),
            len(no_blocker_rollouts),
        ),
    )
