# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The per-task turn cap of a rollout.

This module is pure -- no model, no serving.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


#: Row key carrying a task's turn cap (``max_steps``, three times the length of the
#: recorded human trajectory). Read by the task-row builder and by the environment; named
#: once so the two cannot drift to different spellings of the same field.
STEP_CAP_KEY = "max_steps"


def valid_step_cap(row: Mapping[str, Any]) -> int | None:
    """The row's turn cap, or ``None`` if it carries no usable one.

    ``bool`` is excluded explicitly because ``isinstance(True, int)`` is true in Python,
    and ``True`` would otherwise read as a cap of one turn.
    """
    cap = row.get(STEP_CAP_KEY)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        return None
    return cap
