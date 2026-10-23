# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The user the policy talks to, and the judge that says what it asked about.

BrowserGym gives a policy a channel to the person who set the task
(``send_msg_to_user``) but no person on the other end. This module is that person,
and the grader of what was said to them. Both were previously methods on the
project's own browser harness; they are not browser code and they outlive it.

Two halves, and they are deliberately separate:

:func:`ask_targets` decides which of the task's open parameters a question covers.
A judge does this, not an embedding: the two disagreed in a smoke test where the
judge credited *"what day do I leave?"* as the check-out date and the embedding
scored it below threshold, so the policy earned credit and was simultaneously told
the detail was already settled.

:func:`answer` produces the reply. When the run's reward grades asks, the reply is
built from the parameters the judge credited, so the policy is told exactly what it
earned credit for and the reward cannot disagree with what the policy was told.
Otherwise a served model answers in natural language from the task's own details.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
from collections.abc import Sequence
from typing import Any, Callable

from evaluation import message_kind
from evaluation.missing_parameters import (
    is_statement,
    Parameter,
    parse_targets,
    targets_prompt,
)

#: The only ask rubric this module implements: show the judge the whole open set and
#: credit the union of parameters the question covered.
ASK_JUDGE_KIND = "set"

#: New tokens the ask judge may write for one verdict.
ASK_JUDGE_MAX_TOKENS = 1024

#: Verdicts already obtained, keyed by question and parameter set. A judge outage is
#: never cached, so the next rollout retries instead of inheriting it as a verdict.
_TARGET_CACHE: dict[tuple[str, str, tuple[str, ...]], tuple[str, ...]] = {}
#: Whether the judge read the message as a statement. Keyed
#: identically to _TARGET_CACHE so the two cannot drift apart.
_STATEMENT_CACHE: dict[tuple[str, str, tuple[str, ...]], bool] = {}
#: Messages whose question/statement gate was unreadable, so the combined prompt's
#: own statement verdict was used.
GATE_UNREADABLE: list[str] = []


def _cache_key(
    question: str, instruction: str, params: Sequence[Parameter]
) -> tuple[str, str, tuple[str, ...]]:
    """What makes two judge calls the same call.

    The instruction is in the key because it is in the prompt. Without it, two
    tasks that share a slug set -- ``city`` and ``date`` recur across generated
    rows -- and receive the same message text share one verdict, so one task's
    answer is served for another task's question.
    """
    return (question, instruction, tuple(p.name for p in params))


#: Warnings already printed, so a persistent fault says so once rather than per ask.
_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Print ``message`` the first time ``key`` is seen in this process."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"[simulated_user] WARNING: {message}", flush=True)


class AskJudgeUnavailable(RuntimeError):
    """The ask judge could not be reached for this message (raised only on request)."""


def ask_targets(
    question: str,
    instruction: str,
    parameters: Sequence[Parameter],
    *,
    raise_on_outage: bool = False,
) -> tuple[str, ...]:
    """Which open parameters ``question`` asks about, per the ask judge.

    One judge call per question, not one per (question, parameter) pair: the prompt
    lists the whole set and asks which are covered, so a question that settles three
    things is credited for all three. The task text is in the prompt because the
    reward is graded with it present — a judge shown only the question and a list of
    slot names is a different judge from the one that was measured.

    Raises :exc:`RuntimeError` if the judge's identity is unset. That is a
    misconfigured run rather than an outage, and it has to raise from outside the
    ``except`` below: swallowed, it would score every ask at zero while the run
    reported normally.

    An outage returns no targets, which scores the ask as irrelevant. That is the
    conservative direction for a live reply. The phase-2 grading passes
    ``raise_on_outage=True`` instead: an outage then raises
    :exc:`AskJudgeUnavailable`, so the message is counted as a judge failure and left
    ungraded rather than counted as a real question that hit nothing.
    """
    params = tuple(p for p in parameters if p.name and p.question)
    if not question:
        return ()
    key = _cache_key(question, instruction, params)
    cached = _TARGET_CACHE.get(key)
    if cached is not None:
        return cached

    url = os.environ.get("ASK_JUDGE_URL") or ""
    if not url:
        raise RuntimeError(
            "ASK_JUDGE_URL is unset: set it to the ask judge's endpoint; there "
            "is no default."
        )
    model = os.environ.get("ASK_JUDGE_MODEL") or ""
    if not model:
        raise RuntimeError(
            "ASK_JUDGE_MODEL is unset: set it to the ask judge's served model "
            "name. There is no default, because a name that does not match the serve's "
            "--served-model-name 404s every call and the run learns nothing while "
            "reporting normally."
        )
    budget = ASK_JUDGE_MAX_TOKENS
    try:
        from cascp.common.llm import call_llm  # noqa: PLC0415

        # Asked alone first: asked together with the targets, the judge credited a
        # delivered result with the parameter it mentioned.
        outages: list[Exception] = []

        def _call(p: str) -> str:
            try:
                return call_llm(p, url, model, max_tokens=budget, temperature=0.0)
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                outages.append(exc)
                raise

        gate = message_kind.classify(question, instruction, _call)
        if outages and raise_on_outage:
            # classify() reads an unreachable judge as an unreadable verdict. In
            # the grading pass that would count the message as a question.
            raise outages[0]
        if gate == message_kind.STATEMENT:
            _STATEMENT_CACHE[key] = True
            _TARGET_CACHE[key] = ()
            return ()
        if not params:
            # Nothing open to credit, but the gate still decides whether the message
            # counts as an ask: a report on a Clear row is not over-asking.
            if gate is None:
                GATE_UNREADABLE.append(question)
            _STATEMENT_CACHE[key] = False
            _TARGET_CACHE[key] = ()
            return ()
        out = call_llm(
            targets_prompt(
                instruction=instruction, question=question, parameters=params
            ),
            url,
            model,
            max_tokens=budget,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - outage -> no credit, uncached
        if raise_on_outage:
            raise AskJudgeUnavailable(f"{type(exc).__name__}: {exc}") from exc
        # Loud, once per process. No credit is the conservative verdict and it is
        # kept, but an unreachable judge and a question that genuinely covers
        # nothing produce the identical empty result — and under a reward that
        # grades asks, that empty result is also what the policy is *told*: it
        # hears "that is already specified" for a question nobody scored. A run
        # where every ask reads as irrelevant is then indistinguishable from a
        # policy that asks badly, which is the shape of several hours already lost.
        _warn_once(
            "ask_judge_unreachable",
            f"ask judge unreachable ({type(exc).__name__}: {exc}). Every ask this "
            f"run will score as irrelevant and the policy will be told its "
            f"questions were already answered.",
        )
        return ()
    reply = str(out)
    got = parse_targets(reply, params)
    if gate == message_kind.QUESTION:
        _STATEMENT_CACHE[key] = False  # the gate decides statement-ness
    else:
        GATE_UNREADABLE.append(question)
        _warn_once(
            "gate_unreadable",
            "unreadable question/statement gate verdict; using the combined "
            "prompt's statement verdict",
        )
        _STATEMENT_CACHE[key] = is_statement(reply)
    _TARGET_CACHE[key] = got
    return got


def was_a_statement(
    question: str, instruction: str, parameters: Sequence[Parameter]
) -> bool:
    """Whether the judge read this message as a report rather than a question.

    Reads the verdict :func:`ask_targets` already obtained, so separating the two
    costs no extra judge call. Call ``ask_targets`` first; an unasked pair is
    ``False``, which keeps a message in the ask denominator rather than silently
    dropping it.
    """
    params = tuple(p for p in parameters if p.name and p.question)
    return _STATEMENT_CACHE.get(_cache_key(question, instruction, params), False)


def load_answerer(endpoint: str | None, model: str | None) -> Callable[..., str | None]:
    """The served user simulator, loaded by absolute path.

    ``from scripts.answerer import Answerer`` is shadowed by an unrelated ``scripts``
    package installed in the training venv, which raises ``ModuleNotFoundError``.
    Loading the file by path bypasses ``sys.path`` ordering.
    """
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "answerer.py"
    spec = importlib.util.spec_from_file_location("_stage_answerer", path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreadable file
        raise ImportError(f"no answerer at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    answerer: Callable[..., str | None] = module.Answerer(
        endpoint=endpoint, model=model
    ).answer
    return answerer


def answer(
    answer_fn: Callable[..., str | None],
    question: str,
    instruction: str,
    open_items: Sequence[dict[str, Any]],
    nothing_open: bool = False,
) -> str | None:
    """The simulated user's reply, or None if the answerer failed.

    None is distinct from a reply of "I don't know": it means no user was reached,
    and the caller reports that to the policy rather than passing it off as an
    answer. A silently unanswered question looks exactly like a policy that never
    asked, and a whole run can score zero on it while the logs stay clean.

    ``nothing_open`` says the empty ``open_items`` is the task's own answer rather
    than data that went missing; see :meth:`scripts.answerer.Answerer.answer`.
    """
    try:
        return answer_fn(
            question,
            instruction,
            open_items=list(open_items),
            nothing_open=nothing_open,
        )
    except Exception:  # noqa: BLE001 - must never crash a rollout
        return None
