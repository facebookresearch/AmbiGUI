# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Is a message to the user a question or a statement? Asked on its own, first.

Policies send results and summaries through the same ``send_msg_to_user`` channel
they ask on. Answered or judged as questions, a report that mentions an open
parameter gets that parameter's hidden value, or credit for asking about it. Each
role asks its own model, so rollouts still do not depend on the ask judge.
"""

from __future__ import annotations

import re
from collections.abc import Callable

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


def classify(message: str, instruction: str, call: Callable[[str], str]) -> str | None:
    """``call``'s verdict on ``message``; ``None`` on failure. Never raises."""
    if not (message or "").strip():
        return None
    try:
        return parse_kind(call(PROMPT.format(instruction=instruction, message=message)))
    except Exception:  # noqa: BLE001 - unreadable; the caller counts it
        return None
