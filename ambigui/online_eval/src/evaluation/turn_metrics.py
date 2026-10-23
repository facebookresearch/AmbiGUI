# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""How many turns a task took, reported so the mean cannot be read as unbounded.

A turn is one policy generation the harness consumed. That is deliberately not the
number of calls that reached the page: a generation spent on an unparseable reply, or
on a tool the harness cannot place, still cost the user a turn and dispatched nothing.
``dispatch_efficiency`` is the share of turns that did reach the page.

Read it against one, not as pure waste. The terminal ``stop`` is counted as a turn and
is never dispatched -- the loop breaks on it first -- so even a flawless rollout, every
call placed and a clean finish, scores below 1.

**The headline is the completed mean.** A rollout the policy ended itself -- it emitted
``stop`` -- ran to its natural length, so the mean over that subset is a mean. Every
other rollout ended for a reason that truncates it: the step budget running out, the
loop-breaker firing on a repeated call, an unreadable reply, a refused completion. Those
keep their own ``mean_turns_incomplete`` and their own count, because they are the
*slowest* tasks and dropping them quietly would flatter the headline, while folding them
in quietly would turn the headline into a floor without saying so.

``mean_turns`` over everything measured is still reported, and is a **lower bound**
whenever ``n_incomplete`` is above zero. ``censored_rate`` names the share cut off by a
harness cap specifically, which is the subset whose true length could have been much
larger.

A task that never returned a result carries no turn count at all and lands in
``n_missing`` -- counting it as zero would drag the mean down by exactly the tasks that
ran longest, since they are unfinished because they were slow.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from evaluation.online_records import (
    END_MAX_STEPS,
    END_REPEAT_BREAK,
    END_UNFINISHED,
    SELF_ENDED,
)

#: The reasons the rollout was cut short while the policy was still going. A rollout
#: that ended because the policy said ``stop``, emitted nothing readable, or hit a
#: generation error ended on the policy's or the serve's account, not on a cap the
#: harness imposed -- only a cap makes the recorded length a floor on the true one.
CENSORING_REASONS: frozenset[str] = frozenset({END_MAX_STEPS, END_REPEAT_BREAK})

_NOT_MEASURED = (
    "no task recorded a turn count: the cut is empty, or every task in it was "
    "unfinished, or the run was written before turn counting landed. Not measured, as "
    "distinct from measured zero."
)


def _mean(xs: Sequence[float]) -> float | None:
    """``None`` when there is nothing to average -- undefined, not zero."""
    return round(sum(xs) / len(xs), 4) if xs else None


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


@dataclass(frozen=True)
class TurnStats:
    """Turn counts for one set of tasks, every mean carrying its denominator.

    Every statistic below the counts is taken over the **measured** tasks only -- the
    ones that recorded a turn count. The names say so, because ``n_asked`` here and
    ``ask_rate.k`` in the same scores file would otherwise disagree on any run with an
    unfinished task, with nothing in the file explaining why.
    """

    n_tasks: int
    n_measured: int
    n_missing: int
    applicable: bool
    reason: str | None
    mean_turns: float | None
    median_turns: float | None
    total_turns: int
    n_dispatch_measured: int
    n_completed: int
    mean_turns_completed: float | None
    median_turns_completed: float | None
    n_incomplete: int
    mean_turns_incomplete: float | None
    mean_dispatched: float | None
    dispatch_efficiency: float | None
    n_asked_measured: int
    mean_turns_if_asked: float | None
    n_acted_measured: int
    mean_turns_if_acted: float | None
    max_steps: int | None
    n_censored_measured: int
    censored_rate: float | None
    end_reasons: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_tasks": self.n_tasks,
            "n_measured": self.n_measured,
            "n_missing": self.n_missing,
            "applicable": self.applicable,
            "reason": self.reason,
            "mean_turns": self.mean_turns,
            "median_turns": self.median_turns,
            "total_turns": self.total_turns,
            "n_dispatch_measured": self.n_dispatch_measured,
            "n_completed": self.n_completed,
            "mean_turns_completed": self.mean_turns_completed,
            "median_turns_completed": self.median_turns_completed,
            "n_incomplete": self.n_incomplete,
            "mean_turns_incomplete": self.mean_turns_incomplete,
            "mean_dispatched": self.mean_dispatched,
            "dispatch_efficiency": self.dispatch_efficiency,
            "n_asked_measured": self.n_asked_measured,
            "mean_turns_if_asked": self.mean_turns_if_asked,
            "n_acted_measured": self.n_acted_measured,
            "mean_turns_if_acted": self.mean_turns_if_acted,
            "max_steps": self.max_steps,
            "n_censored_measured": self.n_censored_measured,
            "censored_rate": self.censored_rate,
            "end_reasons": dict(self.end_reasons),
        }


def summarise_turns(
    rows: Sequence[Mapping[str, Any]], *, max_steps: int | None = None
) -> TurnStats:
    """Turn statistics over one cut of a run's per-task records.

    ``rows`` are the records :func:`evaluation.online_records.task_record` wrote. A row
    whose ``n_turns`` is ``None`` stays in ``n_tasks`` and in the ``end_reasons``
    histogram but enters no mean, and is not counted in the censoring rate either: that
    rate exists to qualify ``mean_turns`` as a lower bound, so it has to be taken over
    the same set the mean was.
    """
    end_reasons: dict[str, int] = {}
    for r in rows:
        key = str(r.get("end_reason") or END_UNFINISHED)
        end_reasons[key] = end_reasons.get(key, 0) + 1

    measured = [r for r in rows if r.get("n_turns") is not None]
    turns = [int(r["n_turns"]) for r in measured]
    # A rollout the policy ended itself was not cut off, so its length is its true
    # length and the mean over that subset is a mean rather than a floor. Either
    # deliberate ending counts -- `report_task_complete` and `report_infeasible` are
    # both the policy deciding the episode is over; the `end_reasons` histogram keeps
    # them apart, because *why* it ended is a different question from *how long*.
    # Everything else -- a harness cap, an unreadable reply, a dead serve -- ended without the
    # policy deciding it was finished, and keeps its own column so it is neither
    # dropped (which would flatter the headline, since these are the slowest tasks)
    # nor silently averaged in with the clean ones.
    #
    # A row carrying no end reason at all is in NEITHER group. Those come from runs
    # written before end reasons were recorded, and filing them as cut short would
    # report a whole historical run as though nothing in it ever finished.
    with_end = [r for r in measured if r.get("end_reason") is not None]
    completed = [int(r["n_turns"]) for r in with_end if r["end_reason"] in SELF_ENDED]
    incomplete = [
        int(r["n_turns"]) for r in with_end if r["end_reason"] not in SELF_ENDED
    ]
    # Likewise for the other two counters: an absent value is an unknown, not a zero.
    # Defaulting a missing ask count to zero turns an unknown into positive evidence
    # that the policy committed without asking.
    with_ask = [r for r in measured if r.get("n_ask") is not None]
    asked = [int(r["n_turns"]) for r in with_ask if int(r["n_ask"]) > 0]
    acted = [int(r["n_turns"]) for r in with_ask if int(r["n_ask"]) == 0]
    with_dispatch = [r for r in measured if r.get("n_dispatched") is not None]
    dispatched = [int(r["n_dispatched"]) for r in with_dispatch]
    n_censored = sum(1 for r in with_end if r["end_reason"] in CENSORING_REASONS)
    # Over the rows that recorded both, so the ratio is not deflated by rows that
    # recorded a turn count and no dispatch count.
    turns_with_dispatch = sum(int(r["n_turns"]) for r in with_dispatch)
    total_turns = sum(turns)

    return TurnStats(
        n_tasks=len(rows),
        n_measured=len(measured),
        n_missing=len(rows) - len(measured),
        applicable=bool(measured),
        reason=None if measured else _NOT_MEASURED,
        mean_turns=_mean(turns),
        median_turns=round(statistics.median(turns), 4) if turns else None,
        total_turns=total_turns,
        n_completed=len(completed),
        mean_turns_completed=_mean(completed),
        median_turns_completed=(
            round(statistics.median(completed), 4) if completed else None
        ),
        n_incomplete=len(incomplete),
        mean_turns_incomplete=_mean(incomplete),
        n_dispatch_measured=len(with_dispatch),
        mean_dispatched=_mean(dispatched),
        # Over turns, not over tasks: this is the share of generations that reached the
        # page, so a task that took no turn contributes to neither side. `total_turns`
        # is published beside it so the denominator is on the page.
        dispatch_efficiency=_rate(sum(dispatched), turns_with_dispatch),
        n_asked_measured=len(asked),
        mean_turns_if_asked=_mean(asked),
        n_acted_measured=len(acted),
        mean_turns_if_acted=_mean(acted),
        max_steps=max_steps,
        n_censored_measured=n_censored,
        censored_rate=_rate(n_censored, len(with_end)),
        end_reasons=end_reasons,
    )


def with_n(value: float | None, n: int) -> str:
    """A mean is never printed without the count it was taken over."""
    return f"n/a (n={n})" if value is None else f"{value:.2f} (n={n})"


def format_turns(tag: str, stats: TurnStats) -> str:
    """One line per cut, naming the censoring so the mean is read against its ceiling."""
    if not stats.applicable:
        return f"{tag:<10} NOT MEASURED: {stats.reason}"
    ceiling = "" if stats.max_steps is None else f" max_steps={stats.max_steps}"
    median = "n/a" if stats.median_turns is None else f"{stats.median_turns:.2f}"
    return (
        f"{tag:<10} "
        f"completed {with_n(stats.mean_turns_completed, stats.n_completed)} "
        f"cut short {with_n(stats.mean_turns_incomplete, stats.n_incomplete)} | "
        f"all {with_n(stats.mean_turns, stats.n_measured)} median {median} | "
        f"asked {with_n(stats.mean_turns_if_asked, stats.n_asked_measured)} "
        f"acted {with_n(stats.mean_turns_if_acted, stats.n_acted_measured)} | "
        f"dispatched {with_n(stats.mean_dispatched, stats.n_dispatch_measured)} "
        f"eff {stats.dispatch_efficiency} of {stats.total_turns} turns | "
        f"censored {stats.n_censored_measured}/{stats.n_completed + stats.n_incomplete}{ceiling}"
    )


def turns_block(
    cuts: Mapping[str, Sequence[Mapping[str, Any]]], *, max_steps: int | None
) -> dict[str, Any]:
    """The ``turns`` block of a scores file: one :class:`TurnStats` per named cut.

    Assembling it here rather than in the evaluator keeps the only logic that decides
    what the published block contains inside the gated package.
    """
    return {
        cut: summarise_turns(rows, max_steps=max_steps).as_dict()
        for cut, rows in cuts.items()
    }
