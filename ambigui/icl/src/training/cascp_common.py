#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The policy's system prompt and the task rows built from a benchmark task file.

``CASCP_SYSTEM`` is the system message every policy call starts with; its tool section is
generated from :mod:`evaluation.tool_contract`. :func:`task_rows_from_store` turns the task
file into the rows the environment is reset with.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from evaluation.task_text import task_text
from evaluation.tool_contract import tool_prompt

# System rubric shown to the policy: act from the page via the browse tool, and ask
# when the instruction is ambiguous (a missing parameter, or which of several targets
# is meant) instead of guessing. The tool section is generated from
# evaluation.tool_contract, so what the policy is told it may call is the same
# vocabulary the environment executes and the parser reads.
CASCP_SYSTEM = (
    "You are a web agent. Look at the page and carry out the user's instruction.\n\n"
    + tool_prompt()
    + "\n\nIf the instruction is ambiguous — for example a missing size or date, or "
    "which of several targets is meant — send a message to the user with "
    "`send_msg_to_user` asking for what is missing, instead of guessing. Use "
    "`report_infeasible` when the task cannot be done on this site. When you have "
    "carried the instruction out, call `report_task_complete` with a one-line "
    "summary of what you did; that is the only way to end the episode having "
    "succeeded, and an episode that runs out of turns instead scores nothing."
)


#: Ambiguous rows that reached the simulator with no open items. A task labelled
#: Ambiguous leaves something unsaid by definition, so an empty set on one means the
#: items never arrived -- and the row then reads as "this task left nothing open",
#: which is indistinguishable from a task that really was fully specified. The user
#: simulator invents a value and the reward is scored against the invention. Collected
#: rather than raised so one bad row does not cost a run, surfaced so it cannot pass
#: unnoticed.
#:
#: A Clear row with no items is *correct* and is not counted. Counting it conflated the
#: two: on the 783-row training store the old counter reported 345, which is exactly the
#: number of Clear rows, while the number of Ambiguous rows missing items was 0.
OPEN_ITEM_MISSING_ON_AMBIGUOUS: list[str] = []


def _carry_open_fields(row: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Copy the fields the user simulator needs onto the built row.

    These were absent from both builders' allowlists, so ``open_items`` never reached
    the environment and every clarifying question took the guessing path. Both ends were
    wired correctly and the wire between them was never connected.

    ``ambiguity`` travels with them because the count-only ask reward reads nothing else:
    without it on the row, every task scores as Clear and no ask is ever credited.
    """
    items = source.get("open_items") or source.get("open_details") or []
    row["open_items"] = [i for i in items if isinstance(i, Mapping)]
    row["ambiguity"] = source.get("ambiguity")
    # The per-task turn cap. The environment treats a row without one as fatal, and
    # both allowlists dropped it, so every task in an evaluation ended as a harness
    # error before the policy was asked anything.
    row["max_steps"] = source.get("max_steps")


def _record_missing_open_items(rows: list[dict[str, Any]]) -> None:
    """Record Ambiguous rows that carry no open items."""
    OPEN_ITEM_MISSING_ON_AMBIGUOUS.extend(
        str(r.get("task_id"))
        for r in rows
        if str(r.get("ambiguity")) == "Ambiguous" and not r.get("open_items")
    )


class UnbuildableTaskRow(ValueError):
    """Raised for a store row that cannot be turned into a task, named by its id.

    Skipping such a row shrinks the task set without saying so, and a smaller set
    still produces a score. The id is in the message because a count of dropped rows
    tells nobody which rows to go and fix.

    Every row of the file is used, so every row is checked.
    """


def task_rows_from_store(store_path: Path) -> list[dict[str, Any]]:
    """Task rows from a benchmark task file (JSONL): each line is
    ``{task_id or sample_id, instruction, start_url, ambiguity?, severity?, open_items?, max_steps}``.
    Adds the text ``prompt``. Every row of the file is returned, in file order.

    **No stored screenshot is read.** An online task's turn-0 observation is its live start page,
    photographed by the harness after it navigates; a frame recorded during the offline demonstration a
    row was built from is a different page on a different day. The prompt built here is text only, and
    the rollout adds the live frame. A row that cannot be built raises
    :class:`UnbuildableTaskRow` rather than being skipped.
    """
    rows: list[dict[str, Any]] = []
    for line in Path(store_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tid = str(r.get("task_id") or r.get("sample_id") or "")
        instr, start_url = task_text(r), r.get("start_url")
        if not instr or not start_url:
            missing = " and ".join(
                name
                for name, value in (("instruction", instr), ("start_url", start_url))
                if not value
            )
            raise UnbuildableTaskRow(
                f"{store_path}: row {tid or '(no sample_id)'} has no {missing}"
            )
        built = {
            "task_id": tid,
            "instruction": str(instr),
            "start_url": str(start_url),
            "severity": r.get("severity"),  # gold stakes -> env's side-effect guard
            "prompt": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": CASCP_SYSTEM}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Instruction: {instr}"}],
                },
            ],
        }
        _carry_open_fields(built, r)
        rows.append(built)
    _record_missing_open_items(rows)
    return rows
