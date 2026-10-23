#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pose every recorded step as a turn, and let the policy ask until it acts.

At every recorded step the policy either acts or asks a clarifying question. A question
is answered by a simulated user, and the policy is posed the same step again with the
reply, until it acts.

The protocol, in full:

* **No turn has an ask as its gold action.** The recording's own clarifying question and its
  answer leave the marking path entirely. A turn is marked on the recorded *action* alone.
* **Answers come from a model playing the user.** The task's open items -- each a short name,
  a reference question and the value the recording settled on -- are handed to the answerer,
  which replies in the user's words. Nothing here decides what a question earned: that is
  :mod:`offline_evaluation.run.judge_asks`, which runs after this and writes its verdicts
  onto ``asks.jsonl``.
* **The turn index does not advance on an ask.** The policy may ask several times at one
  turn. The turn moves on only when the policy emits something other than a question.
* **Each exchange persists.** Once asked and answered, a question and its reply are part of
  the conversation, so they are posed again at every later turn.
* **A turn is cut off after :data:`MAX_CONSECUTIVE_ASKS` asks.** On a cut-off turn the task
  stops and every ask at it is marked as cut off. The policy keeps the action marks it
  earned before the loop -- nothing else.

The run is one of three stages, and it is the only one that costs GPU hours it cannot get
back. It holds the policy and the answerer, and nothing else:

===========  ======================  =================================================
stage        models                  what it does
===========  ======================  =================================================
run          policy + answerer       the policy acts or asks; the answerer replies
judge        ask judge               reads the recorded questions, says what each earned
score        none                    pure; reads the judged rows, writes the report
===========  ======================  =================================================

Three outputs, written beside the substrate that produced them:

``turns.jsonl``
    one row per step, holding what the policy finally did and whether it matches the
    recording. A step the run never reached is still a row, marked unreached and wrong, so
    getting stuck cannot shrink the denominator it is marked on.
``asks.jsonl``
    one row per ask, carrying the step it was asked at, its order within that step, the
    reply and the serve that gave it. It carries **no verdict**: run ``judge_asks`` on the
    directory before scoring it. The deadline window is **not** applied here. It is a
    property of the task, it is already on the substrate, and applying it at analysis time
    means one run can be read under more than one window.
    :mod:`offline_evaluation.ask_timing` applies it, reading the per-item windows off
    ``open_items`` rather than the single tightest bound carried on the step.
``inference.jsonl``
    one row per model call -- policy, simulated user and the simulated user's statement
    gate -- holding the exact prompt sent and the reply verbatim. Rows are in a fixed
    order (tasks in benchmark order, calls in the order each task made them), so two
    identical runs write the same file. Only the statement-gate counts in ``report.json``
    are read from it. It exists so a report can be checked against the run behind it: see
    :mod:`offline_evaluation.inference_log`.

The benchmark is read by :mod:`offline_evaluation.gold_benchmark`, which expects the
released AmbiGUI gold evaluation file. Its ``image_path`` fields are relative, so
``--image-root`` is required and names the screenshot pool.

Usage::

    python -m offline_evaluation.run.run_ask_protocol \\
        --benchmark     "$BENCH/gold_eval_set.jsonl" \\
        --image-root    "$POOL" \\
        --model         qwen3.5-4b \\
        --endpoint      http://<node>:<port>/v1 \\
        --served-model  <the serve's --served-model-name> \\
        --answerer-endpoint http://<node>:<port>/v1 \\
        --answerer-model   <a different serve> \\
        --serve-image-limit 4 \\
        --out-dir       "$OUT/ask_protocol_qwen35_4b"

Each endpoint is any OpenAI-compatible ``/v1`` base URL: a local vLLM serve, or a hosted
API. A hosted API's key is read from ``AMBIGUI_POLICY_API_KEY`` (policy) and
``AMBIGUI_ANSWERER_API_KEY`` (answerer); with neither set, no key is sent.

**A failed policy call is not retried.** An HTTP error (429, 5xx), a timeout or an
unreadable body is recorded as an empty reply with its ``error``, and that step is scored
wrong. Retrying would change behaviour against the recorded runs, which made no retry.
The score stage counts these steps as ``calls.n_errored`` in ``report.json``.

``--max-tokens`` replaces the catalogue's output-token budget for this one run. Each
open-weight run-set file lists the budget every recorded pass used
(``policy_max_tokens``), and the launcher passes it; the value is recorded in
``run.json``. Without it the catalogue entry applies.

``--screens on`` is refused at start-up for a catalogue entry with ``vision: false``:
those weights take no image, and every call would fail.

The answerer is **required**. A run without one has no user in it: the policy's questions
would go unanswered, and a conversation in which nobody replies measures nothing. The run
exits non-zero before posing anything rather than producing a file of defaults.

Two answerer failures are told apart, because they need opposite handling. A serve that
cannot be reached is an outage: it is retried, and when no try answers the run
**stops**. A serve that answers and produces nothing readable is a
property of that one question -- the same question gives the same unreadable reply on
every try -- so that question alone is recorded as unanswered and the
run goes on. See :class:`_SimulatedUser`. Either way nothing is passed off as the user's
words: an unanswered ask carries ``answered: false`` and :data:`NO_ANSWER` as its reply.

That split leaves one hole, and :func:`main` closes it at the end: if at least five
questions were asked and none were answered, the run fails. Its evidence is retained
with ``status: failed`` in ``run.json``; judging and scoring refuse that state.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from evaluation.message_kind import GateCall
from evaluation.missing_parameters import Parameter, parameters_from_details
from evaluation.serve_preflight import assert_endpoint_live
from evaluation.simulated_user import load_answerer

from offline_evaluation.calls import Candidates, ParsedCall, parse_candidates
from offline_evaluation.client import (
    Answer,
    ModelConfig,
    ServeClient,
    assert_frame_budget,
    get_model,
)
from offline_evaluation.gold_benchmark import expand
from offline_evaluation.http import (
    ANSWERER_API_KEY_ENV,
    POLICY_API_KEY_ENV,
    api_key_from_env,
)
from offline_evaluation.inference_log import (
    POLICY,
    SIMULATED_USER,
    STATEMENT_GATE,
    InferenceRow,
    InferenceWriter,
    write_run_status,
)
from offline_evaluation.jsonl import read_jsonl
from offline_evaluation.metrics import (
    assert_step_ids,
    credit_from,
    predicted_target,
    predicted_text,
)
from offline_evaluation.prompts import NO_HISTORY, build_offline_prompt
from offline_evaluation.serve_pool import (
    ServePool,
    ServeUnavailable,
    describe,
    serve_targets,
)
from offline_evaluation.substrate import StepItem, write_substrate

#: How many questions in a row end a turn. The policy that reaches this is in a loop: it has
#: asked ten times at one step without acting. Ten rather than two or three so that a task
#: with six open items can legitimately ask about each one and still have room to act.
MAX_CONSECUTIVE_ASKS = 10

#: Screens shown per step, ending at the step's own. Every run set showed one.
FRAME_CAP = 1

#: Seconds allowed for one policy call, and tasks run at once (the run sets' settings).
CALL_TIMEOUT_S = 300.0
WORKERS = 8

#: Heading for the exchanges carried into later turns. Headed and unnumbered, so the numbered
#: action list above it keeps the numbering the recording has and a dialogue line can never
#: be read as an action the policy took.
EXCHANGE_HEADING = "CLARIFICATIONS SO FAR:"

#: What an answerer call that never reached a serve is reported as to the retry pool. This
#: is the transient failure the pool exists for: the next try usually answers, and when none does the run stops rather than writing a file of defaults.
NO_REPLY = "the answerer serve could not be reached"

#: What the policy is told, and what the ask row records, when the answerer *was* reached
#: and still produced nothing a reply could be read out of -- the question was unusable, or
#: the model's JSON never parsed. That is a property of the question, not of the serve: it
#: reproduces on every try, so retrying it spends the whole pool and
#: then kills a run that is otherwise healthy. One question is recorded as unanswered
#: instead, in the user's place, so the policy is told plainly and the row says so.
#: A real reply that happened to be this exact sentence would be read as unanswered, which
#: is the conservative direction: it under-reports the user, never invents one.
NO_ANSWER = "(the user did not answer this question)"

#: The sentence the answerer's rubric gives it to return when the policy asks how to
#: operate the website rather than what the person wanted. It is a rubric instruction,
#: so the answerer follows it as well as it follows anything else: the count printed at
#: the end of a run is of replies that came back in this exact shape, and is a lower
#: bound on refusals rather than the number of method questions asked.
METHOD_REFUSAL = (
    "This is not a question related to the clarifying parameters in the instruction, "
    "therefore I cannot help you."
)


def is_method_refusal(reply: str) -> bool:
    """Whether a reply is the rubric's refusal, up to case and surrounding whitespace.

    Matched on containment, not equality: a model that puts the sentence inside a longer
    reply has still refused. Nothing here reads the question, so this counts the shape of
    the reply and not whether refusing it was right.
    """
    return METHOD_REFUSAL.casefold() in " ".join(reply.split()).casefold()


@dataclass(frozen=True)
class Exchange:
    """One question the policy asked and the reply it received."""

    question: str
    reply: str


@dataclass(frozen=True)
class Posed:
    """One call to the policy, with what the call itself cost.

    Separate from :class:`offline_evaluation.client.Answer` so the walk can be driven by a
    plain function in a test. The fields are the subset a turn record keeps.
    """

    text: str
    error: str | None = None
    finish_reason: str | None = None
    n_frames_sent: int = 0
    n_frames_unreadable: int = 0
    #: The exact text sent to the policy, kept for the inference log.
    prompt: str = ""


#: ``(item, history) -> what the policy said``. The history is passed in rather than read off
#: the item because it grows with every exchange.
Poser = Callable[[StepItem, str], Posed]


@dataclass(frozen=True)
class AskRow:
    """One question, at the step it was asked at, and what the user said back.

    ``ask_index`` counts asks within the step, from 0. ``serve`` names the answerer that
    replied.

    This row carries **no verdict**. ``resolved``, ``relevant`` and ``is_request`` are
    written onto it later by :mod:`offline_evaluation.run.judge_asks`, and the scorer
    refuses a file that has not been through that stage. Recording a verdict here would
    put the judge back in the inference loop, which is the thing this protocol stopped
    doing.
    """

    task_key: str
    step_index: int
    ask_index: int
    question: str
    reply: str
    #: ``"name at endpoint"`` of the answerer that replied.
    serve: str
    cut_off: bool
    #: False when the answerer produced nothing readable for this question and
    #: :data:`NO_ANSWER` stands in ``reply``. A question nobody answered is a different
    #: thing from a question answered badly, and the two must not read alike.
    answered: bool = True


#: How many times the answerer serve is tried, and how long the run waits between those
#: tries. A serve saturated by the policy's own traffic
#: drops requests and recovers in seconds, so a few seconds of patience is worth far more
#: than a run thrown away. Module constants rather than flags: nobody tunes these per run,
#: and a test sets them to zero.
ANSWERER_ATTEMPTS = 3
ANSWERER_RETRY_PAUSE = 5.0


@dataclass(frozen=True)
class CandidateRow:
    """One action from a step's candidate set, as recorded.

    The set is a set of **alternatives for one step**, not a plan. ``rank`` is the order
    the policy wrote them in and nothing more: rank 0 is what it would do first, and the
    rest are actions it says would be equally right. Nothing may run them in sequence.
    """

    rank: int
    action: str
    target: str
    text: str


@dataclass(frozen=True)
class TurnRow:
    """One step, and what the policy finally did at it.

    ``reached`` is false for a step the run never posed, which happens only after a cut-off.
    Such a step is still marked wrong where the recording has an action, so a policy that
    talks itself into a loop cannot shrink its own denominator.

    ``act_correct`` is ``None`` only where the recording has no action this vocabulary can
    express -- a ``read`` or a ``back``. Those steps are outside the action denominator for
    every policy equally. It is true when **any** candidate performed a recorded action the
    run had not reached yet; ``credited_item_id`` names which one.

    ``n_calls`` counts calls to the policy at this step, so it is ``n_asks`` plus the one
    call that acted. The judge's calls are not policy calls and are not counted.

    ``action``, ``predicted_target`` and ``predicted_text`` describe the FIRST candidate.
    ``candidates`` holds all of them, so a reader can see what else the policy offered and
    a re-score can use the whole set. ``rejected`` says why ``candidates`` is empty when it
    is, which separates a reply the reader refused from a reply that was blank.
    """

    item_id: str
    task_key: str
    step_index: int
    arm: str
    reached: bool
    cut_off: bool
    n_asks: int
    n_calls: int
    raw_output: str
    action: str
    predicted_target: str
    predicted_text: str
    gold_operation: str
    gold_target: str
    gold_text: str
    act_correct: bool | None
    error: str | None
    finish_reason: str | None
    n_frames_sent: int
    n_frames_unreadable: int
    candidates: tuple[CandidateRow, ...] = ()
    rejected: str = ""
    #: The id of the gold action this step was credited with, or ``""``. See `credit_from`.
    credited_item_id: str = ""


def render_exchanges(exchanges: Sequence[Exchange]) -> str:
    """The clarification block, or ``""`` when nothing has been asked yet."""
    if not exchanges:
        return ""
    lines = [EXCHANGE_HEADING]
    for exchange in exchanges:
        lines.append(f"you asked: {exchange.question}")
        lines.append(f"the user replied: {exchange.reply}")
    return "\n".join(lines)


def history_with_exchanges(history: str, exchanges: Sequence[Exchange]) -> str:
    """The step's recorded history with the conversation so far under it.

    With no exchanges this returns the history untouched, so the first pose of every step
    is the plain prompt of :mod:`offline_evaluation.prompts`.
    """
    if not exchanges:
        return history
    return f"{history.strip() or NO_HISTORY}\n\n{render_exchanges(exchanges)}"


@dataclass(frozen=True)
class Replied:
    """What the simulated user said to one text the policy produced."""

    reply: str
    #: ``"name at endpoint"`` of the answerer that gave it.
    serve: str = ""
    #: Every failed try before it, in order. Empty when the first try worked.
    retries: tuple[str, ...] = ()
    #: False when ``reply`` is :data:`NO_ANSWER` rather than the user's words.
    answered: bool = True
    #: Every statement-gate call made for this question, in order, each with the
    #: ``"name at endpoint"`` of the serve it went to. One per answerer try.
    gate_calls: tuple[tuple[str, GateCall], ...] = ()


#: ``(instruction, question, open_items) -> the user's reply``. ``open_items`` are
#: ``{name, question, answer}`` mappings, the shape
#: :func:`data_gen.underspecification.render_for_user` reads. Injected so the walk is
#: testable without a serve, and so the user can be a different model from the policy.
Answerer = Callable[[str, str, Sequence[Mapping[str, str]]], Replied]


def as_open_items(parameters: Sequence[Parameter]) -> tuple[dict[str, str], ...]:
    """The open parameter set in the shape the answerer's rubric renders.

    :class:`evaluation.missing_parameters.Parameter` already carries exactly the three
    fields ``render_for_user`` reads, so this is a projection and not a translation. It
    exists as a named function because the key names are a contract with a rubric file
    that lives outside this package.
    """
    return tuple(
        {"name": p.name, "question": p.question, "answer": p.answer} for p in parameters
    )


def answer_question(
    question: str,
    *,
    instruction: str,
    parameters: Sequence[Parameter],
    answerer: Answerer,
) -> Replied:
    """The user's reply to one text the policy produced.

    The **whole** open set is offered every time: someone asked the same thing twice
    answers it twice.

    The fixed refusal is a rule in the answerer's rubric over one class of question --
    how to operate the website -- and a model decides, question by question, whether a
    question is in that class. It is a canned sentence and goes back verbatim. A run
    prints how many of its replies came back in that shape (:data:`METHOD_REFUSAL`); the
    statement gate returns the same sentence, so that count includes gate replies.

    Nothing is decided here about what the question earned. The reply is the user's words
    and only that; :mod:`offline_evaluation.run.judge_asks` reads the recorded questions
    afterwards and says what each one covered.
    """
    return answerer(instruction, question, as_open_items(parameters))


def _unreached(item: StepItem) -> TurnRow:
    """A step the run never posed: wrong wherever the recording has an action."""
    return TurnRow(
        item_id=item.item_id,
        task_key=item.task_key,
        step_index=item.step_index,
        arm=item.arm,
        reached=False,
        cut_off=False,
        n_asks=0,
        n_calls=0,
        raw_output="",
        action="",
        predicted_target="",
        predicted_text="",
        gold_operation=item.gold_operation,
        gold_target=item.gold_target,
        gold_text=item.gold_text,
        act_correct=False if item.gold_operation else None,
        error=None,
        finish_reason=None,
        n_frames_sent=0,
        n_frames_unreadable=0,
    )


def walk_task(
    items: Sequence[StepItem],
    *,
    pose: Poser,
    answerer: Answerer,
    record: Callable[[InferenceRow], None] | None = None,
) -> tuple[list[TurnRow], list[AskRow], list[InferenceRow]]:
    """Run one task's steps in order under the protocol.

    Steps are sorted here rather than trusted to arrive sorted: the conversation grows along
    the trajectory, and reading it in file order would pose a later exchange at an earlier
    step.

    The open items come from the first step. They are a property of the task, and the
    substrate writes the same list onto every one of its steps.

    The third list is every call made, policy and answerer alike, in the order they were
    made. It is kept whole and apart from the two scored lists: see
    :mod:`offline_evaluation.inference_log`.
    """
    ordered = sorted(items, key=lambda i: i.step_index)
    parameters = parameters_from_details(ordered[0].open_items) if ordered else ()
    exchanges: list[Exchange] = []
    turns: list[TurnRow] = []
    asks: list[AskRow] = []
    calls: list[InferenceRow] = []

    def log(row: InferenceRow) -> None:
        calls.append(row)
        if record is not None:
            record(row)

    credited_golds: set[str] = set()
    stopped = False
    for index, item in enumerate(ordered):
        if stopped:
            turns.append(_unreached(item))
            continue
        turn_asks: list[AskRow] = []
        while True:
            attempt = len(turn_asks)
            try:
                posed = pose(item, history_with_exchanges(item.history, exchanges))
            except BaseException as failure:
                # A callback that raised did not return its prompt/output. Do not invent
                # them, but retain the failed attempt and all earlier call records.
                log(
                    InferenceRow(
                        task_key=item.task_key,
                        step_index=item.step_index,
                        attempt=attempt,
                        role=POLICY,
                        prompt="",
                        text="",
                        error=f"{type(failure).__name__}: {failure}",
                    )
                )
                raise
            read = parse_candidates(posed.text)
            log(_policy_row(item, attempt, posed, read))
            # An `ask` with no question is not a question. Treating it as one would let a
            # policy emitting a bare `ask` loop for free; it ends the turn instead and is
            # marked against the recorded action like any other malformed call.
            question = read.question.strip()
            if not question:
                break
            try:
                said = answer_question(
                    question,
                    instruction=item.instruction,
                    parameters=parameters,
                    answerer=answerer,
                )
            except BaseException as failure:
                for serve, gate in getattr(failure, "gate_calls", ()):
                    log(_gate_row(item, attempt, serve, gate))
                log(
                    replace(
                        _simulated_user_row(
                            item, attempt, question, Replied("", answered=False)
                        ),
                        error=f"{type(failure).__name__}: {failure}",
                    )
                )
                raise
            for serve, gate in said.gate_calls:
                log(_gate_row(item, attempt, serve, gate))
            log(_simulated_user_row(item, attempt, question, said))
            exchanges.append(Exchange(question, said.reply))
            turn_asks.append(
                AskRow(
                    task_key=item.task_key,
                    step_index=item.step_index,
                    ask_index=attempt,
                    question=question,
                    reply=said.reply,
                    serve=said.serve,
                    cut_off=False,
                    answered=said.answered,
                )
            )
            if len(turn_asks) >= MAX_CONSECUTIVE_ASKS:
                stopped = True
                break
        if stopped:
            # The turn ended in a loop. Every ask at it is marked as cut off, and the
            # judging stage credits none of them -- otherwise a policy could empty the
            # open-item set by asking its way into a cut-off and still be paid for it.
            turn_asks = [replace(ask, cut_off=True) for ask in turn_asks]
        asks.extend(turn_asks)

        # The first candidate is what the policy would do; the rest are its alternatives.
        # An emission the reader refused leaves an empty call, which marks the step wrong
        # rather than dropping it.
        first = read.first
        # Any one of the alternatives may perform a gold action the run has not reached
        # yet, and each gold action can be credited once. See `credit_from`.
        hit = (
            ""
            if stopped or not item.gold_operation
            else credit_from(ordered[index:], read.calls, credited_golds)
        )
        if hit:
            credited_golds.add(hit)
        turns.append(
            TurnRow(
                item_id=item.item_id,
                task_key=item.task_key,
                step_index=item.step_index,
                arm=item.arm,
                reached=True,
                cut_off=stopped,
                n_asks=len(turn_asks),
                n_calls=len(turn_asks) + (0 if stopped else 1),
                raw_output=posed.text,
                action=first.action,
                predicted_target=predicted_target(first),
                predicted_text=predicted_text(first),
                gold_operation=item.gold_operation,
                gold_target=item.gold_target,
                gold_text=item.gold_text,
                act_correct=(
                    None
                    if not item.gold_operation
                    else (False if stopped else bool(hit))
                ),
                error=posed.error,
                finish_reason=posed.finish_reason,
                n_frames_sent=posed.n_frames_sent,
                n_frames_unreadable=posed.n_frames_unreadable,
                candidates=tuple(
                    CandidateRow(
                        rank=rank,
                        action=call.action,
                        target=predicted_target(call),
                        text=predicted_text(call),
                    )
                    for rank, call in enumerate(read.calls)
                ),
                rejected=read.rejected,
                credited_item_id=hit,
            )
        )
    return turns, asks, calls


def _policy_row(
    item: StepItem, attempt: int, posed: Posed, read: Candidates
) -> InferenceRow:
    """The log row for one pose of the policy, before anything is decided from it."""
    return InferenceRow(
        task_key=item.task_key,
        step_index=item.step_index,
        attempt=attempt,
        role=POLICY,
        prompt=posed.prompt,
        text=posed.text,
        error=posed.error,
        finish_reason=posed.finish_reason,
        n_frames_sent=posed.n_frames_sent,
        candidates=tuple(_rendered(call) for call in read.calls),
        rejected=read.rejected,
    )


def _rendered(call: ParsedCall) -> str:
    """One alternative as the reader understood it, in the form the prompt asks for.

    The emission itself is already in the row verbatim. This is the reading of it, so a
    disagreement between what the policy wrote and what was marked is visible in one place.
    """
    target = predicted_target(call)
    text = predicted_text(call)
    return f'{call.action}(target="{target}", text="{text}")'


def _simulated_user_row(
    item: StepItem, attempt: int, question: str, said: Replied
) -> InferenceRow:
    """The log row for the user's reply to the question at ``attempt``.

    The prompt recorded is the question as the policy wrote it, not the full prompt the
    answerer built around it: that prompt is assembled inside
    :mod:`evaluation.simulated_user`, which this stage calls and does not own.

    A question the answerer could not answer carries :data:`NO_ANSWER` as its error, so
    the log says which calls reached a user and which only reached a serve.
    """
    return InferenceRow(
        task_key=item.task_key,
        step_index=item.step_index,
        attempt=attempt,
        role=SIMULATED_USER,
        prompt=question,
        text=said.reply,
        error=None if said.answered else NO_ANSWER,
        serve=said.serve,
        retries=said.retries,
    )


def _gate_row(item: StepItem, attempt: int, serve: str, gate: GateCall) -> InferenceRow:
    """The log row for one statement-gate call made for the question at ``attempt``.

    Unlike the simulated-user row, the prompt here is the full prompt the gate sent.
    """
    return InferenceRow(
        task_key=item.task_key,
        step_index=item.step_index,
        attempt=attempt,
        role=STATEMENT_GATE,
        prompt=gate.prompt,
        text=gate.reply,
        error=gate.error,
        serve=serve,
        verdict=gate.verdict,
    )


def make_poser(client: ServeClient) -> Poser:
    """A poser that asks one real serve, with the screens the model's registry entry allows."""

    def pose(item: StepItem, history: str) -> Posed:
        prompt = build_offline_prompt(
            instruction=item.instruction,
            history=history,
            # Must agree with what the client attaches, which depends on the model.
            screens=client.config.vision,
        )
        answer = client.answer(prompt, item.frames)
        return Posed(
            text=answer.text,
            error=answer.error,
            finish_reason=answer.finish_reason,
            n_frames_sent=answer.n_frames_sent,
            n_frames_unreadable=answer.n_frames_unreadable,
            prompt=prompt,
        )

    return pose


#: Stand-in registry entry for an answerer serve. :class:`_SimulatedUser` overrides
#: ``answer`` and never reaches :mod:`offline_evaluation.http`, so nothing here is sent
#: anywhere; ``ServeClient`` only requires it to exist. ``vision=False`` records the fact
#: that matters: the user is told the question in words and is shown no screen.
_USER_CONFIG = ModelConfig(
    key="simulated-user",
    hf_path="",
    vision=False,
    served_name="",
    max_image_dim=0,
    max_tokens=0,
)


class _SimulatedUser(ServeClient):
    """One answerer serve, in the shape :class:`ServePool` already drives.

    The pool owns the retries and the final raise, and it drives
    ``ServeClient``s. The simulated user is not one -- it builds its own prompt inside
    :mod:`evaluation.simulated_user` and is reached through that module rather than over
    this package's HTTP -- so it is presented as one instead of the pool being rewritten
    around it.

    Only ``answer`` differs. The prompt is the question; the instruction and the open
    items are bound per instance, because they are per question and the pool's interface
    passes only a string.

    **Two failures, told apart here.** The answerer is called directly rather than through
    a wrapper with a bare ``except``, which would return ``None`` both for a serve that
    could not be reached and for a reply whose JSON never parsed. Those two need opposite
    handling, so they are separated here:

    * the call **raised** -- no serve answered. Reported as an error, which the pool
      retries and finally raises on. A real outage still stops
      the run, which is the guarantee this class exists for.
    * the call **returned ``None``** -- a serve answered and nothing readable came out of
      it. Deterministic: the same question produces the same unreadable reply on every
      try, so retrying it only spends the pool and then throws away every
      policy call in a multi-hour run over one badly-shaped question. It is recorded as
      :data:`NO_ANSWER` for that question alone, which is a usable reply to the pool and a
      visibly unanswered row in the output.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        served_name: str,
        answer_fn: Callable[..., str | None],
        instruction: str,
        open_items: Sequence[Mapping[str, str]],
        gate_log: list[tuple[str, GateCall]] | None = None,
    ) -> None:
        super().__init__(
            endpoint=endpoint, served_name=served_name, config=_USER_CONFIG
        )
        self._answer_fn = answer_fn
        self._instruction = instruction
        self._open_items = open_items
        self._gate_log = gate_log

    def answer(self, prompt: str, frames: Sequence[str]) -> Answer:
        """The user's reply to ``prompt``. ``frames`` are ignored: the user is not shown one.

        Each statement-gate call the answerer makes is added to the shared gate log, with
        this serve's ``"name at endpoint"``, whether or not the answer call succeeds.
        """
        gates: list[GateCall] = []
        try:
            said = self._answer_fn(
                prompt,
                self._instruction,
                open_items=[dict(item) for item in self._open_items],
                gate_log=gates,
            )
        except Exception as unreachable:  # noqa: BLE001 - every failure is a dead serve
            return Answer(
                text="",
                n_frames_sent=0,
                n_frames_unreadable=0,
                error=f"{NO_REPLY} ({type(unreachable).__name__}: "
                f"{str(unreachable)[:200]})",
            )
        finally:
            if self._gate_log is not None:
                serve = f"{self.served_name} at {self.endpoint}"
                self._gate_log.extend((serve, gate) for gate in gates)
        if said is None:
            return Answer(text=NO_ANSWER, n_frames_sent=0, n_frames_unreadable=0)
        return Answer(text=said, n_frames_sent=0, n_frames_unreadable=0)


def make_answerer(
    targets: Sequence[tuple[str, str]],
    *,
    load: Callable[[str, str], Callable[..., str | None]] | None = None,
) -> Answerer:
    """An answerer backed by the online user simulator, with retries.

    ``load`` is resolved here rather than bound as a default argument so a test can
    replace this module's :func:`load_answerer`. It is called once per serve, at
    construction: loading an answerer re-reads its rubric off disk, and doing that per
    question would cost more than the call it wraps.

    The pool and its clients are built **per question**, inside the closure. The walk runs
    tasks concurrently, so the instruction and the open items -- which differ per question
    -- cannot live on a client shared between threads.
    """
    loader = load or load_answerer
    loaded = tuple((host, name, loader(host, name)) for host, name in targets)

    def answerer(
        instruction: str, question: str, open_items: Sequence[Mapping[str, str]]
    ) -> Replied:
        gate_calls: list[tuple[str, GateCall]] = []
        pool = ServePool(
            clients=tuple(
                _SimulatedUser(
                    endpoint=host,
                    served_name=name,
                    answer_fn=answer_fn,
                    instruction=instruction,
                    open_items=open_items,
                    gate_log=gate_calls,
                )
                for host, name, answer_fn in loaded
            ),
            attempts=ANSWERER_ATTEMPTS,
            pause=ANSWERER_RETRY_PAUSE,
        )
        try:
            given = pool.ask_recorded(question)
        except ServeUnavailable as failure:
            # Carried on the exception so walk_task can still log the gate calls made
            # before every try failed.
            failure.gate_calls = tuple(gate_calls)  # type: ignore[attr-defined]
            raise
        return Replied(
            reply=given.text,
            serve=given.serve,
            retries=given.failures,
            answered=given.text != NO_ANSWER,
            gate_calls=tuple(gate_calls),
        )

    return answerer


def group_by_task(items: Sequence[StepItem]) -> list[list[StepItem]]:
    """The items split per task, tasks in first-appearance order.

    Tasks are independent, so they run in parallel; the steps inside one are not, so they do
    not.
    """
    grouped: dict[str, list[StepItem]] = {}
    for item in items:
        grouped.setdefault(item.task_key, []).append(item)
    return list(grouped.values())


def answerer_targets(args: argparse.Namespace) -> tuple[tuple[str, str], ...]:
    """The answerer serve named on the command line.

    :func:`main` refuses an empty endpoint or model before anything is posed, so no
    serve is invented here.
    """
    return serve_targets(args.answerer_endpoint, args.answerer_model)


#: Fewest asks a run must hold before "every ask went unanswered" is allowed to fail it.
#: Below this the rule is reading noise: a run that asked twice and got two unreadable
#: replies is as likely to have put two hard questions as to have a broken answerer.
UNANSWERED_FLOOR = 5

#: The files this pipeline writes into its output directory: stage 1's own, the three
#: the scorer adds, and the launcher's ``run_manifest.json`` (which describes one run, so
#: an earlier run's must not survive next to a new one). Cleared before a run starts --
#: see :func:`clear_run_outputs`. ``substrate.jsonl`` and ``run.json`` are not here: both
#: are rewritten unconditionally before the walk, so neither can survive from an earlier
#: run.
RUN_OUTPUTS = (
    "turns.jsonl",
    "asks.jsonl",
    "inference.jsonl",
    "inference.partial.jsonl",
    "outcomes.jsonl",
    "emissions.jsonl",
    "report.json",
    "run_manifest.json",
)


def clear_run_outputs(out_dir: Path) -> list[str]:
    """Remove any of :data:`RUN_OUTPUTS` already in ``out_dir``, and name what went.

    ``run.json`` and ``substrate.jsonl`` are written before the walk, and every failure
    on the way to the walk raises before ``turns.jsonl`` and ``asks.jsonl`` are. So a
    second run into a directory that already held a finished one used to leave the old
    turns and asks beside a fresh ``run.json``, and the judging and scoring stages read
    that mixture as one run.

    Only files are removed, and only ones named here. A directory is never touched: a
    screenshot folder a reader may have put beside a run is not this pipeline's to delete.
    """
    removed = []
    for name in RUN_OUTPUTS:
        path = out_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


def write_run_config(
    path: Path,
    args: argparse.Namespace,
    config: ModelConfig,
    n_items: int,
    n_tasks: int,
) -> None:
    """Record what produced these turns, beside them.

    The cut-off limit and the answerer belong here as much as the seed does. Both move
    every number the run reports, and neither is recoverable from the rows.

    ``max_tokens_source`` says whether ``max_tokens`` is the catalogue's or came from
    ``--max-tokens``. ``error_accounting`` marks a run made by this runner: its turns
    record every failed policy call and its ``inference.jsonl`` holds the statement-gate
    calls, so the score stage adds ``calls.n_errored`` and ``statement_gate`` to
    ``report.json``. A run recorded before this field existed is scored with the recorded
    report layout, so its re-score stays byte-identical.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "recorded": True,
                "protocol": "ask_loop",
                "status": "running",
                "failure_reason": "",
                "candidate_eligibility": "same_staged_instruction_only",
                "model": config.key,
                "hf_path": config.hf_path,
                "vision": config.vision,
                "served_model": args.served_model,
                "endpoint": args.endpoint,
                "answerer_endpoint": args.answerer_endpoint,
                "answerer_model": args.answerer_model,
                "answerer_serves": [
                    {"endpoint": host, "served_model": name}
                    for host, name in answerer_targets(args)
                ],
                "max_asks": MAX_CONSECUTIVE_ASKS,
                "max_tokens": config.max_tokens,
                "max_tokens_source": (
                    "catalogue" if args.max_tokens is None else "--max-tokens"
                ),
                "error_accounting": True,
                "max_image_dim": config.max_image_dim,
                "seed": args.seed,
                "temperature": 0.0,
                "frame_cap": FRAME_CAP,
                "serve_image_limit": args.serve_image_limit,
                "limit": args.limit,
                "benchmark": str(args.benchmark),
                "image_root": str(args.image_root),
                "n_items": n_items,
                "n_tasks": n_tasks,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write(
    path: Path, rows: Sequence[TurnRow] | Sequence[AskRow] | Sequence[InferenceRow]
) -> None:
    """One JSON object per line, escaped to ASCII.

    ``ensure_ascii=True`` because these rows now carry free-form model text, and every
    reader here splits the file with :meth:`str.splitlines`, which breaks a line on
    ``U+2028`` as well as on ``\\n``. One of those inside a reply would turn one row into
    two unparseable halves. It also keeps stage 1 and the rewrite
    :mod:`offline_evaluation.run.judge_asks` makes of the same file byte-comparable.
    """
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def _non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def _positive(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--benchmark", type=Path, required=True, help="a benchmark half, JSONL"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="written: substrate.jsonl, turns.jsonl, asks.jsonl, inference.jsonl, run.json",
    )
    parser.add_argument("--model", required=True, help="a key from the model registry")
    parser.add_argument("--endpoint", required=True, help="http://<host>:<port>/v1")
    parser.add_argument(
        "--served-model", required=True, help="the serve's --served-model-name"
    )
    parser.add_argument(
        "--answerer-endpoint",
        default="",
        help="serve that plays the user and answers the policy's questions. Required",
    )
    parser.add_argument(
        "--answerer-model",
        default="",
        help="the answerer serve's --served-model-name",
    )
    parser.add_argument(
        "--serve-image-limit",
        type=int,
        default=0,
        help="the serve's limit_mm_per_prompt image value; asserted against the data. "
        "Required for a vision model, ignored for a text-only one",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="the screenshot pool. Every step's image_path is relative to it, and a "
        "frame that is not under it stops the run. Without this the client counts an "
        "unopenable image as unreadable and the run scores a blind policy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the serve's sampling seed; recorded in the run",
    )
    parser.add_argument(
        "--limit",
        type=_non_negative,
        default=0,
        help="run only the first N steps; 0 = every step",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive,
        default=None,
        help="the policy's output-token budget for this run, replacing the catalogue "
        "entry's (default: the catalogue's). The open-weight run-set files list the "
        "budget each recorded pass used",
    )
    parser.add_argument(
        "--screens",
        choices=("on", "off"),
        required=True,
        help="whether to attach screenshots",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalogue = get_model(args.model)
    # A text-only checkpoint takes no image: every call with a screenshot would fail and
    # be scored as a wrong step, so the run would measure nothing.
    if args.screens == "on" and not catalogue.vision:
        raise SystemExit(
            f"FATAL: {catalogue.key} is text-only (vision: false in its catalogue "
            "entry), so it cannot be run with --screens on. Use --screens off."
        )
    config = replace(catalogue, vision=args.screens == "on")
    if args.max_tokens is not None:
        config = replace(config, max_tokens=args.max_tokens)
    items = expand(
        read_jsonl(args.benchmark),
        frame_cap=FRAME_CAP,
        image_root=args.image_root,
        limit=args.limit,
    )
    if not items:
        raise SystemExit(f"FATAL: no scorable steps in {args.benchmark}")
    try:
        assert_step_ids(items)
    except ValueError as broken:
        raise SystemExit(f"FATAL: {broken}") from broken
    # The answerer is not optional. A run without one has no user in it, so a question
    # the policy asks is answered by nothing and the whole protocol measures a monologue.
    if not args.answerer_endpoint:
        raise SystemExit(
            "FATAL: --answerer-endpoint is required; the policy's questions need a user "
            "to answer them"
        )
    if not args.answerer_model:
        raise SystemExit("FATAL: --answerer-model is required with --answerer-endpoint")

    # Every guard fires before any work, because every one of these failures is silent at
    # run time.
    if config.vision:
        if not args.serve_image_limit:
            raise SystemExit(
                f"FATAL: {config.key} is a vision model; pass --serve-image-limit with "
                "the serve's limit_mm_per_prompt image value"
            )
        assert_frame_budget(
            [len(i.frames) for i in items], serve_limit=args.serve_image_limit
        )
    answerers = answerer_targets(args)
    # Every serve the run depends on answers one real request before any work starts. A
    # serve that is down, moved, or published under another name looks identical to a
    # model that answers badly.
    #
    # A hosted endpoint needs a key. Each role reads its own from the environment, so a
    # key never appears on the command line or in run.json; a local serve needs none.
    policy_key = api_key_from_env(POLICY_API_KEY_ENV)
    answerer_key = api_key_from_env(ANSWERER_API_KEY_ENV)
    assert_endpoint_live(args.endpoint, args.served_model, api_key=policy_key)
    for host, name in answerers:
        assert_endpoint_live(host, name, api_key=answerer_key)

    client = ServeClient(
        endpoint=args.endpoint,
        served_name=args.served_model,
        config=config,
        timeout=CALL_TIMEOUT_S,
        seed=args.seed,
        api_key=policy_key,
    )
    pose, answerer = make_poser(client), make_answerer(answerers)
    # Do not discard an earlier run on invalid arguments, data or failed preflight.
    stale = clear_run_outputs(args.out_dir)
    if stale:
        print(f"cleared {len(stale)} file(s) from an earlier run: {', '.join(stale)}")
    tasks = group_by_task(items)
    write_substrate(items, args.out_dir / "substrate.jsonl")
    write_run_config(args.out_dir / "run.json", args, config, len(items), len(tasks))
    try:
        result = _run_tasks(args, config, items, pose, answerer)
        write_run_status(
            args.out_dir,
            "complete" if result == 0 else "failed",
            "" if result == 0 else "inference reached no steps",
        )
        return result
    except BaseException as failure:
        write_run_status(args.out_dir, "failed", f"{type(failure).__name__}: {failure}")
        raise


def _run_tasks(
    args: argparse.Namespace,
    config: ModelConfig,
    items: Sequence[StepItem],
    pose: Poser,
    answerer: Answerer,
) -> int:
    tasks = group_by_task(items)
    answerers = answerer_targets(args)
    record = InferenceWriter(args.out_dir / "inference.partial.jsonl")
    print(
        f"[ask] {len(items)} steps over {len(tasks)} tasks, model={config.key} as "
        f"{args.served_model}, vision={config.vision}, max_asks={MAX_CONSECUTIVE_ASKS}, "
        f"workers={WORKERS}, seed={args.seed}",
        flush=True,
    )
    print(f"[ask]   answerer: {describe(answerers)}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = [
            pool.submit(
                walk_task,
                group,
                pose=pose,
                answerer=answerer,
                record=record,
            )
            for group in tasks
        ]
        walked = []
        dead: ServeUnavailable | None = None
        for future in pending:
            try:
                walked.append(future.result())
            except ServeUnavailable as failure:
                dead = dead or failure
    if dead is not None:
        # The append-only log already includes calls from failed and finished tasks.
        print(
            "[ask] the answerer died; retained durable calls in inference.partial.jsonl",
            flush=True,
        )
        raise SystemExit(f"FATAL: {dead}")
    turns = [row for group, _, _ in walked for row in group]
    asks = [row for _, group, _ in walked for row in group]
    calls = [row for _, _, group in walked for row in group]
    unanswered = [a for a in asks if not a.answered]
    _write(args.out_dir / "turns.jsonl", turns)
    _write(args.out_dir / "asks.jsonl", asks)
    # Every task finished, so ``calls`` holds every row of the partial log. It is
    # written in task order rather than in the order threads finished, so two identical
    # runs write the same file.
    _write(args.out_dir / "inference.jsonl", calls)
    record.path.unlink()
    if len(asks) >= UNANSWERED_FLOOR and len(unanswered) == len(asks):
        # A serve that is up but degraded answers every call with nothing readable. Each
        # question alone is then recorded unanswered and the run finishes, exits zero and
        # reports a policy whose every question went unanswered -- which reads as a policy
        # that asked badly. Nobody reads the stdout of a run this long, so the counts
        # printed below are not enough; the run has to fail.
        #
        # The files remain evidence, but run.json is marked failed by main so later
        # judging/scoring cannot admit them as a complete run.
        print(
            f"[ask] kept {len(turns)} turns, {len(asks)} asks and {len(calls)} model "
            f"calls -> {args.out_dir}",
            flush=True,
        )
        raise SystemExit(
            f"FATAL: none of the {len(asks)} questions this run put were answered. "
            f"The answerer ({describe(answerers)}) was reachable and returned nothing "
            "readable every time. The turn and ask files were written, but a run in "
            "which the user never answered measures nothing: do not judge or score "
            "them. Check the answerer serve, then run again."
        )

    reached = [t for t in turns if t.reached]
    scored = [t for t in turns if t.act_correct is not None]
    right = [t for t in scored if t.act_correct]
    cut = [t for t in turns if t.cut_off]
    print(f"[ask] wrote {len(turns)} turns and {len(asks)} asks -> {args.out_dir}")
    errored = [t for t in reached if t.error]
    gates = [c for c in calls if c.role == STATEMENT_GATE]
    print(f"[ask]   inference log: {len(calls)} model calls")
    print(f"[ask]   reached: {len(reached)}/{len(turns)}   cut off: {len(cut)} tasks")
    print(
        f"[ask]   failed policy calls (scored wrong, not retried): "
        f"{len(errored)}/{len(reached)} reached steps"
    )
    print(
        f"[ask]   statement-gate calls: {len(gates)}, failed: "
        f"{sum(1 for c in gates if c.error)}, unreadable: "
        f"{sum(1 for c in gates if c.verdict is None and not c.error)}"
    )
    print(
        f"[ask]   action right (with look-ahead credit): "
        f"{len(right)}/{len(scored)} marked steps"
    )
    print(
        f"[ask]   asks answered: {len(asks) - len(unanswered)}/{len(asks)} -- unjudged; "
        "run offline_evaluation.run.judge_asks next"
    )
    refused = [a for a in asks if is_method_refusal(a.reply)]
    if refused:
        # The answerer's rubric tells it to refuse a question about how to operate the
        # website with one fixed sentence. A run where most replies are that sentence
        # measured a policy asking the user to drive, and the ask scores below describe
        # something other than what they look like. Counted by matching the sentence, so
        # a reply that refused in its own words is not in here.
        print(
            f"[ask]   replies that came back as the rubric's refusal: "
            f"{len(refused)}/{len(asks)}"
        )
    if unanswered:
        # Named rather than counted silently. Every one of these is a question the policy
        # put and nobody answered, and the policy was told so; a report that does not say
        # how many there were reads them as questions the user answered badly.
        print(
            f"[ask]   asks the answerer could not answer: {len(unanswered)} "
            "(recorded unanswered, not retried: the reply was unreadable, not absent)"
        )
    # Nothing posed at all is a failed run, not an empty result.
    return 0 if reached else 1


if __name__ == "__main__":  # pragma: no cover - main() itself is covered
    raise SystemExit(main())
