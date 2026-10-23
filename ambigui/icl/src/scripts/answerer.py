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
from data_gen.underspecification import NOTHING_OPEN, render_for_user
from evaluation import message_kind

_RUBRICS = (
    Path(__file__).resolve().parents[1] / "rubrics" / "reasoning_decide" / "shared"
)

#: Conditioned on the instruction alone. The simulator has to invent a value, so a
#: question it answers well and a question it answers badly look the same to the reward.
#: Conditioned on what the person actually left open and chose. The simulator states a
#: value it was given rather than one it made up, and refuses questions about how to
#: work the site -- including when the agent phrases the question in clicking terms.
_RUBRIC_OPEN_ITEMS = _RUBRICS / "answer_from_open_items.md"

#: The reply to a message that is not a question: the rubric's own refusal, verbatim.
STATEMENT_REPLY = (
    "This is not a question related to the clarifying parameters in the instruction, "
    "therefore I cannot help you."
)


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


def render_nothing_open_prompt(template: str, *, intent: str, question: str) -> str:
    """Fill the same rubric for a task that left nothing open at all.

    The rubric's own "not in your list" branch then answers the question, which is
    what a person with no further preference would say. Rendering the absence as
    missing data instead invites the simulator to supply the value itself.
    """
    return (
        template.replace("{{INTENT}}", intent)
        .replace("{{QUESTION}}", question)
        .replace("{{OPEN_ITEMS}}", NOTHING_OPEN)
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
        self.open_items_template = _RUBRIC_OPEN_ITEMS.read_text()
        # No token budget here: cascp.common.llm takes it from its model table. A
        # constant here would override what each model needs. GLM-4.6V emits a <think>
        # trace and needs 8192; at 300 the trace exhausts the budget, no JSON is
        # ever emitted, and every answer parses to None -- the agent is told
        # "(no answer)" to every question it asks.
        #: Asks answered from what the person actually chose.
        self.n_from_open_items = 0
        #: Asks on a task that left nothing open. Supported, not guessed: the
        #: simulator was told there is no preference to state.
        self.n_nothing_open = 0
        #: Messages the gate read as statements; they get STATEMENT_REPLY.
        self.n_statements = 0
        #: Messages whose gate verdict was unreadable; answered as questions.
        self.n_gate_unreadable = 0

    def answer(
        self,
        question: str,
        intent: str,
        *,
        open_items: Iterable[Mapping[str, Any]] | None = None,
        nothing_open: bool = False,
    ) -> str | None:
        """The user's short natural reply to ``question``.

        ``open_items`` is what the person left open and what they chose for each. Given
        them, the simulator states a value it was handed. ``nothing_open`` is the other
        case, and only a caller that has read the row can tell it from lost items: the
        task left nothing open at all, so the simulator is told there is no preference
        to state. A task with neither raises (its items failed to join): a reward
        scored against an invented value cannot tell a good question from a bad one.
        A message that is not a question (a report, a summary) gets
        :data:`STATEMENT_REPLY` before any of this. Which path was taken is counted.
        """
        if not question or not intent:
            return None
        # The answerer's own serve: rollouts must not depend on the ask judge.
        kind = message_kind.classify(
            question,
            intent,
            lambda p: call_llm(p, self.endpoint, self.model),
        )
        if kind == message_kind.STATEMENT:
            self.n_statements += 1
            return STATEMENT_REPLY
        if kind is None:
            self.n_gate_unreadable += 1
        items = list(open_items or ())
        usable = [
            item
            for item in items
            if isinstance(item, Mapping) and str(item.get("answer") or "").strip()
        ]
        if usable:
            self.n_from_open_items += 1
            prompt = render_open_items_prompt(
                self.open_items_template,
                intent=intent,
                question=question,
                open_items=usable,
            )
        elif nothing_open:
            # The task left nothing open, and the simulator is told that rather than
            # told the items are missing. The difference is the whole point: an
            # absence invites it to invent a preference the person never had.
            self.n_nothing_open += 1
            prompt = render_nothing_open_prompt(
                self.open_items_template, intent=intent, question=question
            )
        else:
            # No open items reached the simulator for a task that left something open:
            # its items failed to join. Every task of the released file has either open
            # items or nothing open, so this is a data fault, and the simulator is not
            # asked to invent a preference. The caller records the ask unanswered.
            raise ValueError(
                "no open items reached the simulated user for a task that left "
                "something open; its open items failed to join"
            )
        return parse_answer(
            call_llm(prompt, self.endpoint, self.model)
        )

    def counts(self) -> dict[str, int]:
        """How each ask was answered."""
        return {
            "from_open_items": self.n_from_open_items,
            "nothing_open": self.n_nothing_open,
            "statements": self.n_statements,
            "gate_unreadable": self.n_gate_unreadable,
        }
