# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Whether a question arrived while it could still change anything.

The protocol scorer counts a question as credited whenever the judge matches it to an
open item, with no regard for when it was asked. An agent that asks about the hotel city
at step 0 and one that asks after it has already booked a hotel score the same. The
released gold evaluation file carries the information needed to tell them apart, so this
module adds the distinction.

**The window.** Every open item has a ``window`` with a ``latest_step``: the last step at
which asking still helps, which is one step before the action that fixed the value. That
is the convention the annotators used themselves.

**Never-committed items are not a measurement.** Only ``latest_step`` is read. 149 of the
313 items
have ``never_committed`` true -- the run never fixed the value, so the window runs to the
last step and "asked in time" is trivially true. They are counted and reported, never
scored.

Every count in this module describes the released gold evaluation set as published. They
are here to explain the design, and no code reads them. A later drop will differ, and
the numbers a run reports come from the run itself, never from these lines.

**The two window sources are never averaged.** 12 windows were recorded by hand; 301 were
proposed by a model reading the recording. A number mixing them would inherit the weaker
provenance without saying so, so each source gets its own report and the recorded one
carries *n = 12*. How well the proposed windows agree with the recorded ones is a
measurement, and it belongs to whichever run measured it, not to this docstring.

Pure: no model, no network, no randomness.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from offline_evaluation.gold_benchmark import WINDOW_PROPOSED
from offline_evaluation.metrics import ratio
from offline_evaluation.substrate import StepItem


@dataclass(frozen=True)
class TimingReport:
    """Asking-in-time over the open items of one window source."""

    #: ``recorded`` (hand-written) or ``proposed`` (model-read). Never mixed.
    source: str
    #: Open items with a real bound: the run fixed the value, so asking had a deadline.
    n_bounded: int
    #: Of those, the ones the run asked about at all, at any step.
    n_asked: int
    #: Of those, the ones first asked about at or before ``latest_step``.
    n_in_time: int
    #: ``n_in_time / n_bounded``. An item never asked about counts against this, because
    #: never asking is the failure the metric exists to catch.
    asked_in_time: float


@dataclass(frozen=True)
class TimingSummary:
    """Every timing number from one run, with what is excluded stated beside it."""

    by_source: tuple[TimingReport, ...]
    #: Items whose value the run never fixed. "In time" is trivially true for them, so
    #: they are outside every rate above. 149 of the 313 released items.
    n_never_committed: int
    #: Distinct named open items the run's substrate carries, never-committed ones
    #: included. Reported so the denominators above can be checked against the
    #: benchmark's own item count.
    n_items: int

    def as_dict(self) -> dict[str, Any]:
        """The report as JSON, one block per window source."""
        return {
            "n_items": self.n_items,
            "n_never_committed": self.n_never_committed,
            "by_source": {
                report.source: {
                    "n_bounded": report.n_bounded,
                    "n_asked": report.n_asked,
                    "n_in_time": report.n_in_time,
                    "asked_in_time": report.asked_in_time,
                }
                for report in self.by_source
            },
        }


def first_ask_step(asks: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], int]:
    """The earliest step at which each open item was credited, keyed by task and name.

    The earliest, not the latest: a policy that asks about the same thing twice is
    judged on when it first got there.
    """
    out: dict[tuple[str, str], int] = {}
    for row in asks:
        task = str(row["task_key"])
        step = int(row["step_index"])
        for name in row.get("resolved") or ():
            key = (task, str(name))
            out[key] = min(step, out.get(key, step))
    return out


def timing(
    items: Sequence[StepItem], asks: Sequence[Mapping[str, Any]]
) -> TimingSummary:
    """Asking-in-time over every open item the run's substrate carries.

    The open items come off the substrate rather than off the benchmark, so the report
    describes the tasks that were actually posed. A task the run never produced steps for
    contributes nothing, rather than counting as an item nobody asked about.
    """
    asked_at = first_ask_step(asks)
    seen: set[tuple[str, str]] = set()
    never_committed = 0
    buckets: dict[str, list[tuple[int, int | None]]] = {}
    for item in items:
        for open_item in item.open_items:
            name = str(open_item.get("name") or "")
            if not name or (item.task_key, name) in seen:
                continue
            seen.add((item.task_key, name))
            window = open_item.get("window") or {}
            latest = window.get("latest_step")
            if window.get("never_committed") or not isinstance(latest, int):
                never_committed += 1
                continue
            source = str(window.get("source") or WINDOW_PROPOSED)
            buckets.setdefault(source, []).append(
                (latest, asked_at.get((item.task_key, name)))
            )
    reports = []
    for source in sorted(buckets):
        bounded = buckets[source]
        asked = [step for _, step in bounded if step is not None]
        in_time = [
            step for latest, step in bounded if step is not None and step <= latest
        ]
        reports.append(
            TimingReport(
                source=source,
                n_bounded=len(bounded),
                n_asked=len(asked),
                n_in_time=len(in_time),
                asked_in_time=ratio(len(in_time), len(bounded)),
            )
        )
    return TimingSummary(
        by_source=tuple(reports),
        n_never_committed=never_committed,
        n_items=len(seen),
    )


@dataclass(frozen=True)
class AnswerUseReport:
    """Whether the policy did anything with the answers it was given.

    A request is not a commitment. The scorer credits a question the moment the judge
    matches it to an open item, which measures that the policy asked -- not that it then
    used the reply. This is reported beside the ask score and never folded into it: the
    two measure different things, and an agent can legitimately ask about something the
    recording settles several steps later.
    """

    #: Credited open items, whether or not the recording holds an answer for them. An
    #: item with no recorded answer can never be counted as used, so it lowers the rate.
    n_answered: int
    #: Of those, the ones whose answer text appears in an action the policy emitted at
    #: or after the step it asked at.
    n_used: int
    #: ``n_used / n_answered``.
    used: float


def answer_use(
    items: Sequence[StepItem],
    asks: Sequence[Mapping[str, Any]],
    turns: Sequence[Mapping[str, Any]],
) -> AnswerUseReport:
    """How often a credited answer turns up in an action the policy emitted.

    **At or after the asking step, not strictly after it.** One recorded step holds the
    whole exchange: the policy asks, the reply joins the conversation, the step is put to
    it again, and the action it then takes is filed under that same step. Looking only at
    later steps would skip the action that uses the answer in the ordinary case, and
    would score 0 on every single-step task by construction.

    Matching is case-insensitive substring containment against the action's payload. It
    is deliberately generous: a stricter test would mostly measure formatting, and the
    number is reported as evidence of use rather than as a score.

    **Only actions the run reached count.** A step the run never posed carries an empty
    payload and cannot hold the answer, so including it changes nothing; a step after the
    cut-off and a question do change the number, and both are excluded. A question is not
    use of the answer: a policy that asks "did you mean Lisbon?" has repeated the reply
    back, not acted on it, and counting that would score the loop this protocol caps.

    **Every alternative the policy offered counts, not only the first.** A step is
    credited from any of its candidates, so the answer may sit in the second one.
    """
    answers = {
        (item.task_key, str(open_item.get("name") or "")): str(
            open_item.get("answer") or ""
        )
        for item in items
        for open_item in item.open_items
        if open_item.get("name") and open_item.get("answer")
    }
    actions: dict[str, list[tuple[int, str]]] = {}
    for turn in turns:
        if not _is_action(turn):
            continue
        payload = " ".join(
            [
                str(turn.get(field) or "")
                for field in ("predicted_target", "predicted_text")
            ]
            # Every candidate, not only the first. The step is credited from any of them,
            # so reading the first alone can mark one step both correct and as having
            # ignored the reply. A row written before alternatives existed has none.
            + [
                str(candidate.get(field) or "")
                for candidate in turn.get("candidates") or ()
                for field in ("target", "text")
            ]
        ).lower()
        actions.setdefault(str(turn["task_key"]), []).append(
            (int(turn["step_index"]), payload)
        )
    used = 0
    credited = first_ask_step(asks)
    for (task, name), step in credited.items():
        answer = answers.get((task, name), "").strip().lower()
        if answer and any(
            answer in payload for at, payload in actions.get(task, ()) if at >= step
        ):
            used += 1
    return AnswerUseReport(
        n_answered=len(credited),
        n_used=used,
        used=ratio(used, len(credited)),
    )


def reached_turn(turn: Mapping[str, Any] | None) -> bool:
    """Whether the run got as far as this turn row.

    The default on a missing key is True, and it is load-bearing: ``reached`` was added
    after the first runs, and every row written before it exists because the run reached
    that step. Reading an absent key as False marks every one of those rows unreached.

    Shared so that the reader of a turn row cannot disagree with the scorer of one about
    the same row.
    """
    return bool((turn or {}).get("reached", True))


def _is_action(turn: Mapping[str, Any]) -> bool:
    """Whether this turn row holds an action the run reached and finished.

    A row written before these three keys existed is read as a reached action, which is
    what every such row was.
    """
    return (
        reached_turn(turn)
        and not turn.get("cut_off", False)
        and str(turn.get("action") or "") != "ask"
    )
