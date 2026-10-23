# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Keep a multimodal prompt inside the serve's image budget.

A vLLM serve started with ``--limit-mm-per-prompt '{"image": N}'`` refuses any
completion whose prompt carries more than ``N`` images, with HTTP 400 and the message
``At most N image(s) may be provided in one prompt``. The online rollout loop appends
one screenshot per turn and never dropped any, so a rollout crossed the budget on the
turn that took it there and the whole episode ended.

That is not a hypothetical. In one 126-task baseline it ended 47 episodes -- 37% of the
set -- every one of them on the same turn, and the harness recorded them only as
``generation_error``. The policy was never at fault and the run still counted them.

The oldest images go first. What a policy needs to choose its next action is the page
in front of it; the pages it has already left are history, and the text of every turn
-- the action taken and what came back -- is kept either way, so the trajectory stays
readable. Each dropped image leaves a note in its place rather than vanishing, so the
prompt does not silently imply the turn had no screenshot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

#: How many images one policy prompt may carry (every paper run used 4).
#:
#: The serve is started with ``--limit-mm-per-prompt`` at the same number
#: (``harness/policy_serve.sh``) and the loop prunes to it. The two disagreeing costs
#: one episode per overrun, recorded only as a generation error, which is the failure
#: the rest of this module exists to stop.
IMAGE_BUDGET: Final[int] = 4

#: Stands in for a dropped screenshot. The policy is told a frame existed and is no
#: longer shown, which is a different thing from a turn that never had one.
ELIDED = "[earlier screenshot omitted to stay within the prompt's image budget]"


def count_images(messages: Sequence[Mapping[str, Any]]) -> int:
    """How many image parts a message list carries."""
    return sum(
        1
        for message in messages
        for part in _parts(message)
        if part.get("type") == "image_url"
    )


def _parts(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The structured content parts of a message, or none if its content is plain text."""
    content = message.get("content")
    return (
        [p for p in content if isinstance(p, Mapping)]
        if isinstance(content, list)
        else []
    )


def prune_images(
    messages: Sequence[Mapping[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Return ``messages`` carrying at most ``limit`` images, oldest dropped first.

    The input is not modified. Messages are returned in their original order with
    their text intact; only image parts beyond the budget are replaced, each by a
    short note in the same position.

    Raises:
        ValueError: if ``limit`` is less than 1. A budget of zero is a serve that
            accepts no images at all, which is a misconfiguration for a vision policy
            rather than a prompt this function should quietly empty.
    """
    if limit < 1:
        raise ValueError(f"image budget must be at least 1, got {limit}")

    # How many of the images, counting from the oldest, have to go. Counted up front
    # because the decision for any one image depends on how many follow it.
    excess = count_images(messages) - limit
    if excess <= 0:
        return [dict(m) for m in messages]

    out: list[dict[str, Any]] = []
    for message in messages:
        parts = _parts(message)
        if not parts:
            out.append(dict(message))
            continue
        kept: list[Any] = []
        for part in parts:
            if part.get("type") == "image_url" and excess > 0:
                kept.append({"type": "text", "text": ELIDED})
                excess -= 1
            else:
                kept.append(dict(part))
        out.append({**message, "content": kept})
    return out
