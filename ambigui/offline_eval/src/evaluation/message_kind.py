# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Is a message to the user a question or a statement? Asked on its own, first.

This is the *statement gate*. A policy can send the user a result or a summary through
the same ``ask`` action it asks questions with. Answered as a question, a report that
mentions an open item would get that item's hidden value. So the simulated user first
asks its own model whether the message is a question at all; the ask judge is never
involved.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

QUESTION = "question"
STATEMENT = "statement"

PROMPT = """A user gave a web agent a task. While working, the agent sent the \
user the message below. Decide what kind of message it is.

Task: {instruction}

Message from the agent:
<<<
{message}
>>>

It is a QUESTION if it asks the user to supply something the agent needs or to make a \
decision: a missing detail (which city, which date, which size), a choice between \
options, a confirmation before acting, or a request that the user do something such as \
sign in. It is a question even when it is long, and even when it is phrased as a request \
rather than with a question mark.

It is a STATEMENT if it only tells the user something: a result or answer it found, a \
summary of a page, a report of progress, a notice that the task is done, or an \
explanation of what it did. It is a statement even when it mentions a detail the task \
left open, and even when it ends with an offer such as "let me know if you want \
anything else".

Reply with exactly one word: QUESTION or STATEMENT."""


def parse_kind(reply: str) -> str | None:
    """The first word after any closed <think> block; ``None`` if it is neither."""
    lowered = (reply or "").strip().lower()
    if "<think>" in lowered and "</think>" not in lowered:
        return None  # truncated reasoning, not a verdict
    tail = lowered.rsplit("</think>", 1)[-1]
    opening = next((x.strip() for x in tail.splitlines() if x.strip()), "")
    opening = opening.lstrip("*`\"' ")
    for kind in (QUESTION, STATEMENT):
        if re.match(rf"{kind}\b", opening):
            return kind
    return None


@dataclass(frozen=True)
class GateCall:
    """One statement-gate call: what was sent, what came back, and how it was read.

    ``verdict`` is :data:`QUESTION`, :data:`STATEMENT` or ``None``. ``None`` with an
    ``error`` means the call itself failed (the serve raised); ``None`` without one means
    the reply was readable text but neither word.
    """

    prompt: str
    reply: str
    verdict: str | None
    error: str | None = None


def classify_call(
    message: str, instruction: str, call: Callable[[str], str]
) -> GateCall | None:
    """The gate's call on ``message``, or ``None`` when ``message`` is blank (no call made).

    Never raises: a failed call is returned with its ``error`` set, so the caller can log
    and count it.
    """
    if not (message or "").strip():
        return None
    prompt = PROMPT.format(instruction=instruction, message=message)
    try:
        reply = call(prompt)
    except Exception as exc:  # noqa: BLE001 - recorded on the result; the caller counts it
        return GateCall(prompt, "", None, f"{type(exc).__name__}: {exc}"[:200])
    return GateCall(prompt, reply or "", parse_kind(reply))


def classify(message: str, instruction: str, call: Callable[[str], str]) -> str | None:
    """``call``'s verdict on ``message``; ``None`` on failure. Never raises."""
    gate = classify_call(message, instruction, call)
    return None if gate is None else gate.verdict
