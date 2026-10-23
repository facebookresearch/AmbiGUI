# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The whole task the user asked for.

One accessor, imported by everything that needs it. Three places used to read the task
from a field of their own choosing -- the prompt builder, the environment and the
reward -- and two of them read ``instruction``, which holds only the first turn of the
dialogue. On this benchmark 539 of 779 rows are truncated that way, by a median of 151
characters and at worst by 1997: a row whose first turn is ``Hii`` and whose whole task
runs to 2000 characters.

A leaf module with no imports of its own, so importing it adds one file to a deployed
stage rather than a subtree.
"""

from __future__ import annotations

from typing import Any, Mapping

#: Where the whole task may be recorded, best first. ``canonical_instruction`` holds
#: every turn; ``instruction`` holds only the first; ``task`` is the older name.
TASK_FIELDS = ("canonical_instruction", "instruction", "task")


def task_text(row: Mapping[str, Any]) -> str:
    """The whole task, from the first of :data:`TASK_FIELDS` the row actually carries.

    The open details a good clarifying question recovers are grounded in the later
    turns, so a policy shown only the first turn cannot know what the task needs.
    """
    for field in TASK_FIELDS:
        value = row.get(field)
        if value:
            return str(value)
    return ""
