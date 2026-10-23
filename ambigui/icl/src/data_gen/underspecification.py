# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""How the simulated user is shown the details a task leaves open.

Each task row carries its *open items*: the details the instruction leaves unstated, each
with the value the person actually chose (``{name, question, answer, source}``). The
simulated user (``scripts/answerer.py``) is given them as a short list, so it states a
value it was handed rather than one it makes up, and gives the same answer on every
rollout of the same task.

Pure: renders text, no model call.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


#: Shown to the user simulator when a task records nothing open. Naming the absence
#: stops the model filling the gap with an invention of its own -- which is the whole
#: failure this rendering exists to remove.
NO_ITEMS = "(nothing was recorded for this task)"

#: Shown to the simulator when the task left nothing open at all, in place of the
#: open-items list. It is not NO_ITEMS: "nothing was recorded" is an admission that
#: the data is missing, and a simulator told that supplies the value itself. This
#: says there is no value to supply, so the rubric's "not in your list" branch
#: answers the question.
NOTHING_OPEN = (
    "Nothing. Your request was fully specified, and you had no preference "
    "beyond what it already says."
)

#: Phrasings that mark a recorded answer as help with operating the site rather than a
#: statement of what the person wanted. Kept, not deleted: a real person did give that
#: help on that recording, and erasing it would misrepresent the trajectory. What
#: changes is how it may be used.
_HOW_TO = (
    "click",
    "scroll",
    "button",
    "tab ",
    "menu",
    "icon",
    "press ",
    "select the",
    "navigate",
    "go to the",
    "top right",
    "top left",
    "bottom of the",
)


def is_how_to(answer: str) -> bool:
    """Whether a recorded answer tells the agent how to work the site."""
    low = f" {str(answer or '').strip().lower()} "
    return any(marker in low for marker in _HOW_TO)


def render_for_user(items: Iterable[Mapping[str, Any]]) -> str:
    """The open items as the user simulator sees them: parameter, then value.

    Items are ``{name, question, answer, source}``; ``parameter`` is read in place of
    ``name`` when present.

    An item whose recorded answer is really help with the site is marked inline, so the
    simulator can tell the two apart: it may give that answer when asked about that item
    and must never volunteer that kind of help anywhere else.
    """
    lines = []
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        answer = str(item.get("answer") or "").strip()
        if not answer:
            # An answerless item tells the simulator nothing and invites it to fill the
            # blank itself, which is the behaviour being removed.
            continue
        name = str(item.get("parameter") or item.get("name") or "").strip()
        label = name.replace("_", " ") if name else "something you had in mind"
        mark = (
            "   [the recording shows a person gave this help]"
            if is_how_to(answer)
            else ""
        )
        lines.append(f"- {label}: {answer}{mark}")
    return "\n".join(lines) or NO_ITEMS
