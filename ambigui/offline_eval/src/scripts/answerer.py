#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The answerer: a language model that replies to the agent's questions as the user would.

When the agent asks a clarifying question, a model playing the user (a *simulated user*,
here called the *answerer*) replies. The answerer is never shown a recorded trajectory or
its target elements. It is given the task's instruction and, when the task records them,
the task's *open items*: the details the instruction left unstated, each with the value the
person actually chose. It is told to answer only what was asked, and not to volunteer
on-screen targets or steps, so its reply cannot leak the answer to the step being scored.

    from scripts.answerer import Answerer
    ans = Answerer(endpoint="http://localhost:8000/v1", model="glm_4_6v").answer(question, intent)
"""

from __future__ import annotations

from pathlib import Path

from collections.abc import Iterable, Mapping
from typing import Any

from cascp.common.llm import call_llm
from cascp.common.parsing import parse_judge_json
from data_gen.underspecification import render_for_user
from evaluation import message_kind

_RUBRICS = (
    Path(__file__).resolve().parents[1] / "rubrics" / "reasoning_decide" / "shared"
)

#: Conditioned on the instruction alone. The simulator has to invent a value, so a
#: question it answers well and a question it answers badly look the same to the reward.
_RUBRIC = _RUBRICS / "answer_user_intent.md"

#: Conditioned on what the person actually left open and chose. The simulator states a
#: value it was given rather than one it made up, and refuses questions about how to
#: work the site -- including when the agent phrases the question in clicking terms.
_RUBRIC_OPEN_ITEMS = _RUBRICS / "answer_from_open_items.md"

#: The reply to a message that is not a question: the rubric's own refusal, verbatim.
STATEMENT_REPLY = (
    "This is not a question related to the clarifying parameters in the instruction, "
    "therefore I cannot help you."
)


def render_intent_prompt(template: str, *, intent: str, question: str) -> str:
    """Fill the intent-conditioned answerer rubric (no gold trajectory)."""
    return template.replace("{{INTENT}}", intent).replace("{{QUESTION}}", question)


def render_open_items_prompt(
    template: str,
    *,
    intent: str,
    question: str,
    open_items: Iterable[Mapping[str, Any]],
) -> str:
    """Fill the open-items answerer rubric with what the person actually chose."""
    return (
        template.replace("{{INTENT}}", intent)
        .replace("{{QUESTION}}", question)
        .replace("{{OPEN_ITEMS}}", render_for_user(open_items))
    )


def parse_answer(raw: str) -> str | None:
    """Extract the ``answer`` string from the model's JSON reply; None if absent/blank."""
    obj = parse_judge_json(raw)
    if not obj:
        return None
    a = obj.get("answer")
    return a.strip() if isinstance(a, str) and a.strip() else None


class Answerer:
    """Serves the user-simulator: ``answer(question, intent) -> reply``."""

    def __init__(
        self,
        endpoint: str,
        model: str,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.template = _RUBRIC.read_text(encoding="utf-8")
        self.open_items_template = _RUBRIC_OPEN_ITEMS.read_text(encoding="utf-8")
        # No token budget here: cascp.common.llm takes it from its model table. A
        # constant here would override what each model needs. GLM-4.6V emits a <think>
        # trace and needs 8192; at 300 the trace exhausts the budget, no JSON is
        # ever emitted, and every answer parses to None -- the agent is told
        # "(no answer)" to every question it asks.

    def answer(
        self,
        question: str,
        intent: str,
        *,
        open_items: Iterable[Mapping[str, Any]] | None = None,
        gate_log: list[message_kind.GateCall] | None = None,
    ) -> str | None:
        """The user's short natural reply to ``question``.

        ``open_items`` is what the person left open and what they chose for each. Given
        them, the simulator states a value it was handed; without them it invents one,
        and a reward scored against an invented value cannot tell a good question from
        a bad one. A message that is not a question (a report, a summary) gets
        :data:`STATEMENT_REPLY` before any of this.

        ``gate_log``, when given, receives the statement-gate call (prompt, reply, verdict
        and any error), so the caller can write it to its inference log and count gate
        failures. A gate call that fails or gives no verdict is treated as a question.
        """
        if not question or not intent:
            return None
        # The answerer's own serve: rollouts must not depend on the ask judge.
        gate = message_kind.classify_call(
            question,
            intent,
            lambda p: call_llm(p, self.endpoint, self.model),
        )
        if gate is not None and gate_log is not None:
            gate_log.append(gate)
        if gate is not None and gate.verdict == message_kind.STATEMENT:
            return STATEMENT_REPLY
        items = list(open_items or ())
        usable = [
            item
            for item in items
            if isinstance(item, Mapping) and str(item.get("answer") or "").strip()
        ]
        if usable:
            prompt = render_open_items_prompt(
                self.open_items_template,
                intent=intent,
                question=question,
                open_items=usable,
            )
        else:
            # No open items reached the simulator: it answers from the instruction alone.
            prompt = render_intent_prompt(
                self.template, intent=intent, question=question
            )
        return parse_answer(
            call_llm(prompt, self.endpoint, self.model)
        )
