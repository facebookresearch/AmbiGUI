# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Mark a teacher-forced pass: tool calls and per-step actions.

Nothing here talks to a model. It reads the items and what the policy said about them, and
produces numbers, so a grading rule can change and a pass is re-marked in seconds without
the model being run again. The score stage (``offline_evaluation.run.score_ask_protocol``)
uses these pieces; the asking and timing measures are computed there and in
``offline_evaluation.ask_timing``.

*Tool calls* (:func:`call_report`). Whether the emission was a call at all, whether the
action it named exists, and whether the argument that action needs was supplied. A policy
that is right about what to do and cannot express it fails for a different reason from one
that is wrong about what to do, and one number would hide that.

*Actions* (:func:`act_report`). At each step, did it perform the recorded operation on the
recorded element. Operation and target are scored separately, and the target only where the
recording names one -- on the phone recordings in this pool most clicks were captured as a
bare coordinate, so counting them as target misses would deflate the column by construction
rather than measure anything.

*Checks* (:func:`assert_step_ids`, :func:`assert_run_coverage`, :func:`ask_step_items`).
A substrate with broken step identity, or a run whose saved evidence does not cover it, is
refused rather than scored.

**Nothing unreadable is silently dropped.** An emission that could not be parsed is a failed
step, not a skipped one; skipping would shrink the denominator and flatter a policy that
returned nothing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from offline_evaluation.calls import (
    ParsedCall,
    parse_candidates,
    target_match,
)

from offline_evaluation.substrate import StepItem

#: The argument names the ``browse`` tool advertises. Anything else the policy writes was
#: invented, which is a tool-call failure even when the action itself is right.
ADVERTISED_ARGUMENTS = frozenset({"target", "text", "question"})

#: What each action must supply to be a well-formed call. ``stop`` takes nothing.
#:
#: ``select`` requires the option as well as the control. The recording names the option --
#: ``select "Large" from "Size"`` is marked on both halves -- so a call that names only the
#: control cannot be scored right whatever it chose, and accepting it as well-formed would
#: report a policy as compliant on a call the scorer must fail.
REQUIRED_ARGUMENT: Mapping[str, tuple[str, ...]] = {
    "click": ("target",),
    "select": ("target", "text"),
    "scroll": ("target",),
    "goto": ("target",),
    "type": ("target", "text"),
    "ask": ("question",),
    "stop": (),
}


@dataclass(frozen=True)
class CallReport:
    """Whether the policy could express itself at all."""

    n: int
    n_unparseable: int
    parse_rate: float
    #: Of the calls that parsed, how many supplied every argument their action requires.
    schema_valid_rate: float
    #: Of the calls that parsed with named arguments, how many invented one.
    unknown_argument_rate: float
    #: Emissions that offered more than one alternative action for their step.
    n_multi: int = 0
    #: Alternatives per emission that parsed. 1.0 means every emission named one action.
    mean_alternatives: float = 0.0
    #: Of ``n_unparseable``, how many ran out of token budget before writing an action.
    #: A reasoning model cut off mid-thought has not failed the output format; it was
    #: never given room to reach it, and the fix is the budget rather than the prompt.
    #: 0 when the run recorded no finish reasons.
    n_truncated: int = 0


@dataclass(frozen=True)
class ActReport:
    """How well it acted, on the steps where acting was what was due."""

    n_scored: int
    #: Steps left out because the recorded tool has no ``browse`` equivalent -- a ``read``
    #: observation or a key press. The policy was never offered a way to emit them.
    n_inexpressible: int
    #: Steps left out because the policy asked a question that was due at that step.
    n_exempt_ask: int
    operation_match: float
    n_target_scorable: int
    target_match_strict: float
    target_match_tolerant: float
    #: Steps whose recording names what was typed or selected -- a ``type`` into the right box
    #: with the wrong text is a different failure from a ``type`` into the wrong box.
    n_text_scorable: int
    #: Compared strictly. A typed value has no reworded form.
    text_match: float
    #: Steps the deterministic check rejected and the judge overturned. 0 when no judge ran.
    n_judge_overturned: int
    #: The right operation on the right element with the right content. Where the recording
    #: names no element, the operation alone.
    step_success: float
    #: Scored steps credited against a recorded action FURTHER ALONG the trajectory rather
    #: than their own. The tolerant rule allows this on purpose, because a policy offering
    #: several actions may name one the run has not reached yet. Published because a reader
    #: otherwise cannot tell it from a policy that got each step right where it stood.
    n_credited_ahead: int = 0
    #: ``step_success`` counting only the steps credited against their OWN recorded action.
    #: The strict reading, printed beside the tolerant one so neither can stand alone.
    step_success_own_step: float = 0.0


@dataclass(frozen=True)
class StepOutcome:
    """One scored step, kept so a number can be traced back to the emissions behind it."""

    item_id: str
    task_key: str
    step_index: int
    arm: str
    predicted_action: str
    predicted_payload: str
    #: The element the policy named. Read from the ``target`` argument rather than from the
    #: payload, because for a ``type`` the payload is the text typed, not the box typed into.
    predicted_target: str
    #: The text the policy said it would type or select.
    predicted_text: str
    gold_operation: str
    gold_target: str
    gold_text: str
    #: True when this step was an ask that was due. Such a step is not an action failure.
    asked_in_time: bool
    #: ``None`` when the step is not scored for acting at all.
    act_correct: bool | None
    #: True when the deterministic check rejected this step and the judge overturned it.
    judge_overturned: bool = False
    #: The question judge's verdict on this step's clarifying question. ``None`` when no
    #: question was graded here -- either none was asked in time, or the task records no
    #: reference question to compare against, or the judge reached no verdict.
    question_relevant: bool | None = None
    #: The id of the gold action this step was credited with, or ``""``. Usually this
    #: step's own id; it is a later step's id when the policy named an action the run had
    #: not reached yet. Recorded so that no gold action can be counted twice.
    credited_item_id: str = ""


def ratio(num: float, den: float) -> float:
    """``num / den`` to four places, or 0.0 when there is nothing to divide."""
    return round(num / den, 4) if den else 0.0


def call_report(
    outputs: Sequence[str], finish_reasons: Sequence[str | None] = ()
) -> CallReport:
    """Parsing and tool-call validity over every raw emission in the pass.

    One emission is one row of this report however many alternatives it holds, because the
    question the parse rate answers is "could the policy express itself at all". Counting a
    four-alternative emission four times would make a verbose policy look like a policy
    that answered four steps.

    The schema check runs over every alternative, not just the first. An alternative that
    omits the argument its action requires is a malformed call whether or not the policy
    put it first, and the policy may be credited for it later.

    ``finish_reasons`` is the serve's reason for stopping each emission, in the same order.
    Supply it to separate an emission that ran out of token budget from one that reached
    the end and still broke the format: the first is fixed by raising the budget and the
    second by the prompt, and one number cannot say which to do. Leave it empty when the
    run did not record them, and every unreadable emission counts as malformed.
    """
    if not outputs:
        raise ValueError("call_report needs at least one emission; got none")
    if finish_reasons and len(finish_reasons) != len(outputs):
        raise ValueError(
            f"call_report needs one finish reason per emission or none; got "
            f"{len(finish_reasons)} for {len(outputs)} emissions"
        )
    reads = [parse_candidates(text) for text in outputs]
    good = [read for read in reads if read.calls]
    truncated = sum(
        not read.calls and reason == "length"
        for read, reason in zip(reads, finish_reasons)
    )
    every = [call for read in good for call in read.calls]
    named = [call for call in every if not call.bare]
    return CallReport(
        n=len(reads),
        n_unparseable=len(reads) - len(good),
        parse_rate=ratio(len(good), len(reads)),
        schema_valid_rate=ratio(sum(_schema_ok(c) for c in every), len(every)),
        unknown_argument_rate=ratio(
            sum(bool(set(c.args) - ADVERTISED_ARGUMENTS) for c in named), len(named)
        ),
        n_multi=sum(len(read.calls) > 1 for read in good),
        mean_alternatives=ratio(len(every), len(good)),
        n_truncated=truncated,
    )


def _schema_ok(call: ParsedCall) -> bool:
    """Whether the call supplied every argument its action requires.

    The bare ``click OK`` shape names no arguments by construction, so it is taken as valid:
    charging it for a missing argument would be a complaint about a syntax the policy was
    never told not to use.
    """
    if call.bare:
        return True
    return all(call.args.get(name) for name in REQUIRED_ARGUMENT[call.action])


def predicted_target(call: ParsedCall) -> str:
    """The element the call names.

    The bare ``click OK`` shape has no named arguments, so its payload is the element. In every
    other shape the element is the ``target`` argument: reading the payload instead would give
    the typed text on a ``type`` and score it against an element name.
    """
    return call.payload if call.bare else str(call.args.get("target") or "")


def predicted_text(call: ParsedCall) -> str:
    """The text the call would type or select, or ``""``."""
    return "" if call.bare else str(call.args.get("text") or "")


def act_correct(item: StepItem, call: ParsedCall) -> bool:
    """Whether the step's recorded action was performed.

    The element and the content are each only required where the recording names one. Where it
    names neither -- most phone clicks were captured as a bare coordinate -- the operation is
    the whole of what can be checked.
    """
    if call.action != item.gold_operation:
        return False
    if item.gold_target and not target_match(
        predicted_target(call), item.gold_target, tolerant=True
    ):
        return False
    # Strictly for the typed value: the loose match passes on half the content words, which
    # would accept "San Diego" for "San Francisco".
    if item.gold_text and not target_match(
        predicted_text(call), item.gold_text, tolerant=False
    ):
        return False
    return True


def credit_from(
    ahead: Sequence[StepItem],
    calls: Sequence[ParsedCall],
    credited: Sequence[str] | set[str],
) -> str:
    """The id of the gold action this step's candidates perform, or ``""`` for none.

    The policy offers up to four alternatives for one step, because on many steps more than
    one action is defensible. The step is credited if **any** of them performs a gold action
    that the run has not reached yet, which is what ``ahead`` holds: this step's own recorded
    action first, then the ones after it in trajectory order. Own-step first, so a step that
    did exactly what was recorded is never charged against a later action instead.

    A gold action pays for one step only. ``credited`` holds the ids already spent, and a
    candidate matching one of them earns nothing -- otherwise a policy repeating a single
    correct call would collect the whole trajectory. Nothing before this step is reachable
    at all: that action has already happened, and performing it again is not progress.

    Only the same staged instruction is eligible. This prevents credit for a follow-up
    not yet visible; it does NOT prove DOM availability or action prerequisites.
    """
    spent = set(credited)
    for gold in ahead:
        if ahead and (
            gold.task_key != ahead[0].task_key
            or gold.instruction != ahead[0].instruction
        ):
            continue
        if not gold.gold_operation or gold.item_id in spent:
            continue
        if any(act_correct(gold, call) for call in calls):
            return gold.item_id
    return ""


def assert_step_ids(items: Sequence[StepItem]) -> None:
    """Refuse a substrate whose step identity is broken, naming what is wrong.

    Two defects are silent, and each corrupts every per-task number. A repeated ``item_id``
    makes one emission stand for two steps, because emissions are keyed by it. A task whose
    step indices are not a complete ``0..n-1`` run has lost or gained a step, so "the first
    ask by the deadline" is compared against step numbers that no longer mean what the
    deadline was written against.

    :raises ValueError: an id repeats, or a task's steps are not a complete run.
    """
    repeated = sorted(
        item_id for item_id, n in Counter(i.item_id for i in items).items() if n > 1
    )
    if repeated:
        raise ValueError(f"step ids must be unique; repeated: {', '.join(repeated)}")
    steps: dict[str, list[int]] = {}
    for item in items:
        steps.setdefault(item.task_key, []).append(item.step_index)
    broken = {
        task: sorted(found)
        for task, found in steps.items()
        if sorted(found) != list(range(len(found)))
    }
    if broken:
        shown = "; ".join(
            f"{task} has {found}" for task, found in sorted(broken.items())
        )
        raise ValueError(f"every task needs a complete 0..n-1 step run; {shown}")


def ask_step_items(
    items: Sequence[StepItem], asks: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, int], StepItem]:
    """Resolve every ask to its exact staged step; unknown/duplicate joins are errors."""
    steps = {(item.task_key, item.step_index): item for item in items}
    if len(steps) != len(items):
        raise ValueError("substrate task/step joins must be unique")
    seen: set[tuple[str, int, int]] = set()
    for row in asks:
        key = (str(row["task_key"]), int(row["step_index"]))
        if key not in steps:
            raise ValueError(f"unknown ask task/step: {key}")
        identity = (*key, int(row["ask_index"]))
        if identity[2] < 0 or identity in seen:
            raise ValueError(f"invalid or duplicate ask identity: {identity}")
        seen.add(identity)
    return steps


def assert_run_coverage(
    items: Sequence[StepItem],
    turns: Sequence[Mapping[str, Any]],
    asks: Sequence[Mapping[str, Any]],
    *,
    require_credit_ids: bool = False,
) -> None:
    """Validate saved evidence, without changing its action-marking semantics."""
    expected = {item.item_id: item for item in items}
    counts = Counter(str(row["item_id"]) for row in turns)
    if any(n != 1 for n in counts.values()) or set(counts) != set(expected):
        raise ValueError("turns must uniquely cover every substrate item exactly once")
    ask_step_items(items, asks)
    asks_by_step: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for ask in asks:
        key = (str(ask["task_key"]), int(ask["step_index"]))
        asks_by_step.setdefault(key, []).append(ask)
    spent: set[str] = set()
    for turn in turns:
        item = expected[str(turn["item_id"])]
        if require_credit_ids:
            # Status-bearing runs use the current format. Legacy files may lack these
            # counters, but new runs must retain every recorded exchange exactly once.
            step_asks = asks_by_step.get((item.task_key, item.step_index), [])
            n_asks = int(turn.get("n_asks", -1))
            if n_asks != len(step_asks) or sorted(
                int(a["ask_index"]) for a in step_asks
            ) != list(range(n_asks)):
                raise ValueError(
                    f"ask records do not cover turn {item.item_id}'s n_asks"
                )
            if step_asks and not turn.get("reached"):
                raise ValueError(f"unreached turn {item.item_id} has ask records")
            if any(
                bool(a.get("cut_off")) != bool(turn.get("cut_off")) for a in step_asks
            ):
                raise ValueError(f"ask cutoff flags disagree with turn {item.item_id}")
        for field in (
            "task_key",
            "step_index",
            "arm",
            "gold_operation",
            "gold_target",
            "gold_text",
        ):
            if turn.get(field) != getattr(item, field):
                raise ValueError(
                    f"turn {item.item_id} disagrees with substrate {field}"
                )
        credited = str(turn.get("credited_item_id") or "")
        if not credited:
            # Older runs did not record gold-credit IDs. New complete runs must.
            if turn.get("act_correct") and (
                require_credit_ids or "credited_item_id" in turn
            ):
                raise ValueError(f"correct turn {item.item_id} has no gold-credit ID")
            continue
        gold = expected.get(credited)
        if (
            gold is None
            or credited in spent
            or gold.task_key != item.task_key
            or gold.step_index < item.step_index
            or gold.instruction != item.instruction
            or not gold.gold_operation
            or not turn.get("act_correct")
            or not turn.get("reached")
            or turn.get("cut_off")
        ):
            raise ValueError(f"invalid or duplicate gold-credit ID: {credited}")
        spent.add(credited)


def act_report(outcomes: Sequence[StepOutcome]) -> ActReport:
    """The action columns, over the steps where acting was due."""
    scored = [o for o in outcomes if o.act_correct is not None]
    with_target = [o for o in scored if o.gold_target]
    with_text = [o for o in scored if o.gold_text]
    # Naming the element is scored whatever operation was chosen. Requiring the operation to
    # match first would fold two failures into one column, and the point of reporting them
    # apart is to tell "clicked the wrong thing" from "typed into the right thing".
    strict = sum(
        target_match(o.predicted_target, o.gold_target, tolerant=False)
        for o in with_target
    )
    tolerant = sum(
        target_match(o.predicted_target, o.gold_target, tolerant=True)
        for o in with_target
    )
    # Typed text is compared STRICTLY. The loose match asks whether two strings name the same
    # element, and passes on half the content words -- which makes "San Diego" match
    # "San Francisco" and "3 adults" match "2 adults". A typed value has no reworded form:
    # entering a different value is entering a different value.
    typed = sum(
        target_match(o.predicted_text, o.gold_text, tolerant=False) for o in with_text
    )
    return ActReport(
        n_scored=len(scored),
        # Disjoint from n_exempt_ask, so n_scored + n_inexpressible + n_exempt_ask is the
        # item count and the denominator can always be reconstructed.
        n_inexpressible=sum(
            1 for o in outcomes if not o.gold_operation and not o.asked_in_time
        ),
        n_exempt_ask=sum(1 for o in outcomes if o.asked_in_time),
        operation_match=ratio(
            sum(o.predicted_action == o.gold_operation for o in scored), len(scored)
        ),
        n_target_scorable=len(with_target),
        target_match_strict=ratio(strict, len(with_target)),
        target_match_tolerant=ratio(tolerant, len(with_target)),
        n_text_scorable=len(with_text),
        text_match=ratio(typed, len(with_text)),
        n_judge_overturned=sum(1 for o in scored if o.judge_overturned),
        step_success=ratio(sum(bool(o.act_correct) for o in scored), len(scored)),
        # Compared with the step's own id, never tested as a suffix: `task__s1` is a tail
        # of `task__s11`, and credit from ten steps along would read as credit from here.
        # An empty id is a run recorded before alternatives existed, where every credit
        # was necessarily against the step's own recorded action.
        n_credited_ahead=sum(1 for o in scored if _credited_ahead(o)),
        step_success_own_step=ratio(
            sum(1 for o in scored if o.act_correct and not _credited_ahead(o)),
            len(scored),
        ),
    )


def _credited_ahead(outcome: StepOutcome) -> bool:
    """Whether this step's credit came from a recorded action further along."""
    return bool(
        outcome.act_correct
        and outcome.credited_item_id
        and outcome.credited_item_id != outcome.item_id
    )
