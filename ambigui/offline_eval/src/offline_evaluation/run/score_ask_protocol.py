# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Mark a run of the ask-and-answer protocol.

``offline_evaluation.run.run_ask_protocol`` writes ``turns.jsonl`` and ``asks.jsonl``,
and :mod:`offline_evaluation.run.judge_asks` adds the judge's verdicts to ``asks.jsonl``.
This program reads that finished, judged run directory and writes ``outcomes.jsonl``,
``emissions.jsonl`` and ``report.json``.

**No judge and no network.** Every number comes from the run's own files. ``asks.jsonl``
carries the judge's verdicts, and ``substrate.jsonl`` carries each task's open items, so
the score is reproducible from the files alone and re-running it cannot produce a
different answer.

**Nothing is exempt for having asked.** Every step with an expressible recorded action
is marked, whether or not the policy asked at it, so an agent that asked and one that
acted are marked on the same steps. A step the run never reached is wrong, not absent. A
step cut off at the question limit is wrong, not excused.

**Call errors.** For a run whose ``run.json`` has ``error_accounting`` (every run made by
this release's runner), ``report.json`` also holds ``calls.n_errored`` -- reached steps
whose policy call failed (HTTP error, timeout, unreadable body); each is scored wrong --
and ``statement_gate``: the simulated user's statement-gate calls read from
``inference.jsonl``, with how many failed and how many gave no readable verdict. A run
recorded without that field keeps the recorded report layout, so re-scoring it is
byte-identical.

**Asking is marked on when as well as on what.** Each open item carries the last step at
which asking it still helps. :mod:`offline_evaluation.ask_timing` reports the fraction of
questions that arrived by then, split by how the window was settled and never averaged
across the two. It also reports, separately, how often a credited answer then appears in
an action -- asking is not the same as acting on the reply.

Usage::

    python -m offline_evaluation.run.score_ask_protocol <run-dir>

Reads ``substrate.jsonl``, ``turns.jsonl``, ``asks.jsonl`` and ``run.json`` from the
directory, and writes ``outcomes.jsonl``, ``emissions.jsonl`` and ``report.json`` beside
them. Returns 1, and prints no action or question table, when the run was not staged.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from offline_evaluation.ask_timing import (
    AnswerUseReport,
    TimingSummary,
    answer_use,
    timing,
)
from offline_evaluation.ask_variants import (
    VARIANTS,
    AskedQuestion,
    TaskAsks,
    score_variants,
)
from offline_evaluation.inference_log import STATEMENT_GATE, require_complete_run
from offline_evaluation.jsonl import read_jsonl
from offline_evaluation.metrics import (
    StepOutcome,
    act_report,
    assert_run_coverage,
    assert_step_ids,
    call_report,
    ratio,
)
from offline_evaluation.gold_benchmark import LEADS, WINDOW_RECORDED
from offline_evaluation.substrate import StepItem, read_substrate


@dataclass(frozen=True)
class StagingReport:
    """Whether each step was posed only the conversation said at or before it.

    A run made before that fix posed the whole conversation at every step, so the policy
    could act on a request the person had not made yet, and a step was marked against a
    request it does not serve. Its action and question numbers are not comparable with a
    staged run's, and reporting them side by side would invite exactly that comparison.
    """

    n_items: int
    n_tasks: int
    n_staged_items: int
    n_staged_tasks: int
    #: Tasks whose steps were not all posed the same text. Staging that changes nothing
    #: across a task is staging that did no work, even where the lead line is present.
    n_tasks_varying: int

    @property
    def staged(self) -> bool:
        """Whether every step of this run was posed a staged conversation.

        Every step, not one. A run that mixes the two is the harder case, not the easier
        one: its staged tasks and its unstaged tasks are marked under different
        conditions, and one table over both says nothing about either.

        A step is read as staged from its lead line -- either
        :data:`offline_evaluation.substrate.DIALOGUE_LEAD` or, on a row that records no
        dialogue, :data:`offline_evaluation.gold_benchmark.SINGLE_TURN_LEAD`. Only a
        staged read writes either, so a run posed the raw instruction carries neither
        and reads as not staged, which is what this check is for. 183 of the 298 gold
        rows have no dialogue to stage; they are correctly staged trivially, and
        refusing them would refuse every run of the benchmark.
        """
        return self.n_items > 0 and self.n_staged_items == self.n_items


@dataclass(frozen=True)
class AskBehaviourReport:
    """What the policy asked, and what asking cost it."""

    n_tasks: int
    n_ask_tasks: int
    n_act_tasks: int
    n_tasks_that_asked: int
    n_questions: int
    #: Questions the run's judge matched to an open item of that task. Over
    #: :attr:`n_request_judged`, like the two below it.
    n_questions_credited: int
    #: Judged questions that earned nothing, which still cost a call. Unjudged rows are
    #: **not** in here: a question the judge never read is not a question it refused, and
    #: counting the two together put two denominators under three names.
    n_questions_irrelevant: int
    #: Asks the request judge ran on, and the denominator of the three counts around it.
    #: Over :attr:`n_questions`. A run scored before that judge existed reports 0, so the
    #: counts read as 0 of 0 rather than as a measurement.
    n_request_judged: int
    #: Judged asks the judge said were not a request at all: a report, a refusal, or an
    #: offer of further help. These earn nothing however much they are about an open item.
    n_not_a_request: int
    n_tasks_cut_off: int
    n_steps_cut_off: int
    #: Act tasks the policy put a question on. Those tasks leave nothing open.
    n_overask_tasks: int
    #: Steps the run never posed, which happens only after a task was cut off.
    n_unreached_steps: int
    #: Questions per task, over every task. The direct cost of the protocol.
    mean_asks_per_task: float
    #: Policy calls per reached step. 1.0 is a policy that never asked.
    mean_calls_per_step: float


def questions_by_step(asks: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int], str]:
    """The last question put at each step.

    A cut-off step's recorded action is ``ask`` and the question itself lives in the ask
    file, so without this join such a step records as an action with no content at all.
    """
    out: dict[tuple[str, int], str] = {}
    for row in asks:
        out[(str(row["task_key"]), int(row["step_index"]))] = str(
            row.get("question") or ""
        )
    return out


def outcome(
    turn: Mapping[str, Any], questions: Mapping[tuple[str, int], str]
) -> StepOutcome:
    """One scored step, from the turn the runner recorded.

    The runner's own verdict is carried rather than recomputed. It marked the step
    against the item it actually posed, and re-deriving the mark here would let a later
    change to the marking rules silently re-score a finished run.

    ``asked_in_time`` is always false. It is kept for a stable output layout; this
    protocol exempts no step from the action denominator.
    Whether a question arrived in time is measured, but as a property of the open item
    rather than of the step, by :mod:`offline_evaluation.ask_timing`.
    """
    action = str(turn.get("action") or "")
    target = str(turn.get("predicted_target") or "")
    text = str(turn.get("predicted_text") or "")
    if action == "type":
        payload = text
    elif action == "ask":
        payload = questions.get((str(turn["task_key"]), int(turn["step_index"])), "")
    else:
        payload = target
    return StepOutcome(
        item_id=str(turn["item_id"]),
        task_key=str(turn["task_key"]),
        step_index=int(turn["step_index"]),
        arm=str(turn.get("arm") or ""),
        predicted_action=action,
        predicted_payload=payload,
        predicted_target=target,
        predicted_text=text,
        gold_operation=str(turn.get("gold_operation") or ""),
        gold_target=str(turn.get("gold_target") or ""),
        gold_text=str(turn.get("gold_text") or ""),
        asked_in_time=False,
        act_correct=turn.get("act_correct"),
        credited_item_id=str(turn.get("credited_item_id") or ""),
    )


def staging(items: Sequence[StepItem]) -> StagingReport:
    """Whether this run's steps were posed a staged conversation."""
    staged = [i for i in items if i.instruction.startswith(LEADS)]
    posed: dict[str, set[str]] = {}
    for i in items:
        posed.setdefault(i.task_key, set()).add(i.instruction)
    return StagingReport(
        n_items=len(items),
        n_tasks=len(posed),
        n_staged_items=len(staged),
        n_staged_tasks=len({i.task_key for i in staged}),
        n_tasks_varying=sum(1 for texts in posed.values() if len(texts) > 1),
    )


def ask_behaviour(
    turns: Sequence[Mapping[str, Any]], asks: Sequence[Mapping[str, Any]]
) -> AskBehaviourReport:
    """The question columns, counted off the turn and ask files."""
    arms = {str(t["task_key"]): str(t.get("arm") or "") for t in turns}
    asked = {str(t["task_key"]) for t in turns if int(t.get("n_asks") or 0)}
    cut_off = {str(t["task_key"]) for t in turns if t.get("cut_off")}
    reached = [t for t in turns if t.get("reached")]
    judged = [a for a in asks if a.get("is_request") is not None]
    return AskBehaviourReport(
        n_tasks=len(arms),
        n_ask_tasks=sum(1 for a in arms.values() if a == "ask"),
        n_act_tasks=sum(1 for a in arms.values() if a != "ask"),
        n_tasks_that_asked=len(asked),
        n_questions=len(asks),
        n_questions_credited=sum(1 for a in asks if a.get("resolved")),
        # Judged rows only. A row the judge never read -- every open item of its task was
        # already spent -- carries no verdict, and reading that as "not relevant" counted
        # it against a denominator it was never in.
        n_questions_irrelevant=sum(1 for a in judged if not a.get("relevant")),
        n_request_judged=len(judged),
        n_not_a_request=sum(1 for a in asks if a.get("is_request") is False),
        n_tasks_cut_off=len(cut_off),
        n_steps_cut_off=sum(1 for t in turns if t.get("cut_off")),
        n_overask_tasks=sum(1 for k in asked if arms.get(k) != "ask"),
        n_unreached_steps=sum(1 for t in turns if not t.get("reached")),
        mean_asks_per_task=ratio(
            sum(int(t.get("n_asks") or 0) for t in turns), len(arms)
        ),
        mean_calls_per_step=ratio(
            sum(int(t.get("n_calls") or 0) for t in reached), len(reached)
        ),
    )


def task_asks(
    items: Sequence[StepItem], asks: Sequence[Mapping[str, Any]]
) -> list[TaskAsks]:
    """One entry per task, with its open items and every question it drew.

    The open items come off the substrate, which carries them from the benchmark row. A
    task that drew no question still appears, or recall would divide by the tasks the
    policy chose to answer for.
    """
    arms: dict[str, str] = {}
    names: dict[str, frozenset[str]] = {}
    for item in items:
        arms.setdefault(item.task_key, item.arm)
        names.setdefault(
            item.task_key,
            frozenset(
                str(o.get("name") or "") for o in item.open_items if o.get("name")
            ),
        )
    drawn: dict[str, list[AskedQuestion]] = {key: [] for key in arms}
    for row in asks:
        key = str(row["task_key"])
        if key in drawn:
            drawn[key].append(
                AskedQuestion(
                    covered=frozenset(str(n) for n in (row.get("resolved") or ()))
                )
            )
    return [
        TaskAsks(
            task_key=key,
            arm=arms[key],
            open_items=names[key],
            questions=tuple(drawn[key]),
        )
        for key in arms
    ]


def _write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="a directory written by run_ask_protocol",
    )
    return parser.parse_args(argv)


def act_block(act: Any) -> dict[str, Any]:
    """The action table with each step-success rate named after the rule behind it.

    Two rules produce two numbers here, and ``step_success`` is the name of neither.
    :func:`offline_evaluation.metrics.credit_from` scans the rest of the trajectory, so
    a step can be paid by a recorded action further along; that is deliberate -- a policy
    offering alternatives may name one the run has not reached yet -- and it is looser
    than the step-for-step rule published under the plain name elsewhere. Carrying the
    plain name here would invite a reader to compare the two directly.

    Neither number is computed differently. Only the keys change: ``step_success_own_step``
    for the step-for-step rule, ``step_success_with_lookahead`` for the one that credits
    from further along.
    """
    renamed = dict(asdict(act))
    renamed["step_success_with_lookahead"] = renamed.pop("step_success")
    return renamed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir: Path = args.run_dir
    try:
        config = require_complete_run(run_dir)
    except ValueError as broken:
        raise SystemExit(f"FATAL: {broken}") from broken
    items = read_substrate(run_dir / "substrate.jsonl")
    try:
        assert_step_ids(items)
    except ValueError as broken:
        raise SystemExit(f"FATAL: {broken}") from broken
    turns = list(read_jsonl(run_dir / "turns.jsonl"))
    asks = list(read_jsonl(run_dir / "asks.jsonl"))
    try:
        assert_run_coverage(items, turns, asks, require_credit_ids="status" in config)
    except ValueError as broken:
        raise SystemExit(f"FATAL: {broken}") from broken
    # An unjudged ask has no verdict on it, and every reader here treats a missing
    # verdict as "this question earned nothing". A run scored before judging would
    # therefore report a policy that asked usefully as one that never did, and look
    # complete doing it.
    #
    # Both fields, not ``resolved`` alone. A file written before the request judge
    # existed carries ``resolved`` and no ``is_request``, so keying on ``resolved``
    # passed it straight through to be scored with ``n_request_judged = 0`` -- which
    # reads as a policy none of whose questions were requests.
    unjudged = [row for row in asks if "resolved" not in row or "is_request" not in row]
    if unjudged:
        raise SystemExit(
            f"FATAL: {len(unjudged)} of {len(asks)} asks carry no verdict. Run "
            "python -m offline_evaluation.run.judge_asks on this run first"
        )
    stage = staging(items)
    report: dict[str, Any] = {
        "run": config,
        "staging": asdict(stage) | {"staged": stage.staged},
    }

    if not stage.staged:
        # Refusing to print the rest is the point. A step posed the whole conversation
        # could act on a request the person had not made yet, and its numbers read as a
        # result. A run that is only partly staged is refused for the same reason: one
        # table over two marking conditions describes neither.
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            f"NOT STAGED: {stage.n_staged_items} of {stage.n_items} steps were posed a "
            "staged conversation.\n"
            "The action and question tables are withheld. Either this run was posed "
            "requests the person had not yet made, or its benchmark carries no "
            "conversation to stage. Both make its numbers non-comparable with a staged "
            "run's."
        )
        return 1

    questions = questions_by_step(asks)
    outcomes = [outcome(t, questions) for t in turns]
    # The parse rate is over the steps the policy was actually asked at. An unreached
    # step has a blank reply because the protocol stopped, not because the model failed.
    reached = [t for t in turns if t.get("reached")]
    if not reached:
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            f"NO REPLIES: the run reached none of its {len(turns)} steps, so there is "
            "nothing to mark. Check the policy serve was answering."
        )
        return 1

    calls = call_report(
        [str(t.get("raw_output") or "") for t in reached],
        [t.get("finish_reason") for t in reached],
    )
    act = act_report(outcomes)
    behaviour = ask_behaviour(turns, asks)
    content = score_variants(task_asks(items, asks))
    when = timing(items, asks)
    use = answer_use(items, asks, turns)

    calls_block = asdict(calls)
    gate: dict[str, int] | None = None
    if config.get("error_accounting"):
        # Only for runs that record it: adding keys to a recorded run's report would
        # break its byte-identical re-score.
        calls_block["n_errored"] = sum(1 for t in reached if t.get("error"))
        try:
            gate = statement_gate_report(read_jsonl(run_dir / "inference.jsonl"))
        except FileNotFoundError as missing:
            raise SystemExit(
                f"FATAL: {run_dir / 'inference.jsonl'} is missing; the statement-gate "
                "counts are read from it"
            ) from missing
    report |= {
        "calls": calls_block,
        "act": act_block(act),
        "ask_behaviour": asdict(behaviour),
        "ask_content": content.as_dict(),
        "ask_timing": when.as_dict(),
        "answer_use": asdict(use),
    }
    if gate is not None:
        report["statement_gate"] = gate
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write(run_dir / "outcomes.jsonl", [asdict(o) for o in outcomes])
    # Reached steps only, so a reader of this file computes the same parse rate as
    # report.json. An unreached step has a blank reply that is not an unreadable one.
    _write(
        run_dir / "emissions.jsonl",
        [
            {
                "item_id": str(t["item_id"]),
                "raw_output": str(t.get("raw_output") or ""),
                "error": t.get("error"),
                "finish_reason": t.get("finish_reason"),
            }
            for t in reached
        ],
    )
    _print(stage, calls, act, behaviour, content.as_dict(), when, use)
    if gate is not None:
        print(
            f"failed policy calls: {calls_block['n_errored']} of {calls.n} reached steps "
            "(each scored wrong; no retry)"
        )
        print(
            f"statement gate: {gate['n']} calls, {gate['n_errored']} failed, "
            f"{gate['n_unreadable']} unreadable, {gate['n_statement']} read as statements"
        )
    return 0


def statement_gate_report(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Counts over the simulated user's statement-gate calls in ``inference.jsonl``.

    ``n_errored``: the call failed (the serve raised). ``n_unreadable``: a reply came back
    but was neither word. In both cases the answerer treats the message as a question.
    ``n_statement``: read as a statement, so the user replied with the fixed refusal.
    """
    gates = [row for row in rows if row.get("role") == STATEMENT_GATE]
    return {
        "n": len(gates),
        "n_errored": sum(1 for row in gates if row.get("error")),
        "n_unreadable": sum(
            1 for row in gates if row.get("verdict") is None and not row.get("error")
        ),
        "n_statement": sum(1 for row in gates if row.get("verdict") == "statement"),
    }


def _print(
    stage: StagingReport,
    calls: Any,
    act: Any,
    behaviour: AskBehaviourReport,
    content: Mapping[str, Any],
    when: TimingSummary,
    use: AnswerUseReport,
) -> None:
    """The report as a person reads it, in the order the numbers qualify each other."""
    print(
        f"staged: {stage.n_staged_items}/{stage.n_items} steps, "
        f"{stage.n_staged_tasks}/{stage.n_tasks} tasks, "
        f"{stage.n_tasks_varying} tasks posed more than one text"
    )
    print(
        f"parse rate: {calls.parse_rate} over {calls.n} reached steps "
        f"({calls.n_unparseable} unreadable, of which {calls.n_truncated} ran out "
        f"of token budget)"
    )
    # The headline is the step-for-step rule. The tolerant one credits a step from an
    # action further along the trajectory, which is deliberate, but printing it first
    # would let a policy that lists actions ahead of itself read the same as one that
    # got each step right where it stood. So it goes underneath, as the caveat.
    print(
        f"step success: {act.step_success_own_step} over {act.n_scored} scored steps, "
        f"each credited against its own step "
        f"(exempt {act.n_exempt_ask}, inexpressible {act.n_inexpressible})"
    )
    print(
        f"  crediting from further along the trajectory as well: {act.step_success}; "
        f"{act.n_credited_ahead} of {act.n_scored} were credited that way"
    )
    print(
        f"  operation {act.operation_match}, "
        f"target {act.target_match_strict} over {act.n_target_scorable}, "
        f"text {act.text_match} over {act.n_text_scorable}"
    )
    print(
        f"asking: {behaviour.n_tasks_that_asked}/{behaviour.n_tasks} tasks asked, "
        f"{behaviour.n_questions} questions, {behaviour.n_questions_credited} credited, "
        f"{behaviour.n_overask_tasks} on tasks that left nothing open"
    )
    print(
        f"  not a request: {behaviour.n_not_a_request} of "
        f"{behaviour.n_request_judged} judged"
    )
    print(
        f"  cost: {behaviour.mean_asks_per_task} asks per task, "
        f"{behaviour.mean_calls_per_step} calls per reached step, "
        f"{behaviour.n_tasks_cut_off} tasks cut off, "
        f"{behaviour.n_unreached_steps} steps never reached"
    )
    print(f"ask content, over {content['n_gradeable']} gradeable tasks:")
    for variant in VARIANTS:
        print(
            f"  {variant:<15} credited {content[f'n_credited_{variant}']:>4}  "
            f"recall {content[f'ask_recall_{variant}']}  "
            f"precision {content[f'ask_precision_{variant}']}  "
            f"f1 {content[f'ask_f1_{variant}']}"
        )
    print(
        f"ask timing, over {when.n_items} open items "
        f"({when.n_never_committed} never committed, excluded):"
    )
    for report in when.by_source:
        print(
            f"  {report.source:<15} in time {report.n_in_time}/{report.n_bounded} "
            f"= {report.asked_in_time}  (asked at all: {report.n_asked})"
        )
    # A handful of hand-recorded windows is a thin yardstick. A timing number read off
    # the proposed windows is reproducible and evidenced, not verified, and saying so
    # here means a reader of the printed report cannot miss it. Only counts this run
    # produced are printed: a figure from an earlier run, sitting in a line of this
    # run's report, reads as this run's and there is nothing to tell the reader apart.
    recorded = next(
        (r for r in when.by_source if r.source == WINDOW_RECORDED),
        None,
    )
    if recorded is not None:
        print(
            f"  the recorded windows are n={recorded.n_bounded}; the proposed ones are "
            "a heuristic and are not independently verified"
        )
    print(
        f"answer use: {use.n_used}/{use.n_answered} = {use.used} of credited answers "
        "appear in an action at or after the asking step (reported beside the ask "
        "score, not folded into it)"
    )


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
