# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The teacher-forced substrate: one item per recorded step, and its file format.

**What teacher forcing is and why it is here.** A recorded trajectory is a chain: step 3 only
makes sense after steps 1 and 2 happened. If an agent is scored by letting it drive the whole
chain, one wrong move at step 1 makes every later step unanswerable, and the score stops
measuring the agent's ability and starts measuring how early it slipped. Mind2Web's offline
protocol removes that: each step is scored on its own, and the agent is handed the *recorded*
history of what came before, never its own.

:class:`StepItem` is one such step. ``offline_evaluation.gold_benchmark`` builds the items from
the benchmark file; :func:`write_substrate` and :func:`read_substrate` store them, so the
scorer rebuilds exactly the items the runner posed. :func:`browse_operation` maps a recorded
tool to the ``browse`` action the policy is shown; a step whose tool the policy cannot express
(``read``, ``enter``, ``back``) is left unscorable rather than counted as a miss.

Pure: no model, no network, no randomness.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


#: Recorded tool -> the ``browse`` action a policy would emit for it. ``input`` and ``clear`` are
#: both text entry, and a swipe is how a phone recording scrolls; mapping them lets a correct
#: answer count on the real steps that carry one.
BROWSE_EQUIVALENT: Mapping[str, str] = {
    "click": "click",
    "type": "type",
    "input": "type",
    "clear": "type",
    "select": "select",
    "swipe": "scroll",
    "scroll": "scroll",
    "goto": "goto",
    "stop": "stop",
}


DEADLINE_NONE = "none"

#: Dialogue roles that are never posed to the policy. The protocol under test is the agent
#: asking its own question and being answered from the open items. Posing the recording's
#: own question and its answer would settle the open item before the agent asks, and the
#: ask measurement would collapse.
WITHHELD_ROLES = frozenset({"clarifying_question", "clarifying_answer"})
#: Recorded speaker -> how the turn is labelled in the prompt. ``system`` is the agent side
#: of the recording, which is the assistant from the policy's point of view.
SPEAKERS: Mapping[str, str] = {"user": "user", "system": "assistant"}
#: Opens the posed conversation. ``TASK:`` is followed by the dialogue rather than a single
#: sentence, so the policy is told what it is reading and which line is the live request.
DIALOGUE_LEAD = (
    "the user and the assistant have been talking. the latest request is last."
)


def browse_operation(tool: str | None) -> str:
    """``tool`` in the ``browse`` vocabulary, or ``""`` when the policy cannot express it.

    ``read`` is an observation rather than an action, and ``enter``/``back``/``press``/``hotkey``
    are key presses the advertised tool has no argument for. Returning ``""`` marks the step
    unscorable for operation; scoring it would charge the policy for a call it was never offered.
    """
    return BROWSE_EQUIVALENT.get(str(tool or "").strip().lower(), "")


@dataclass(frozen=True)
class StepItem:
    """One step to be scored, with everything needed to pose it and to mark it."""

    source: str
    sample_id: str
    step_index: int
    instruction: str
    #: The recorded actions before this step, one per line. The agent's own choices never
    #: enter it -- that is the whole point of the protocol.
    history: str
    frames: tuple[str, ...]
    #: ``ask`` or ``act`` -- which behaviour the benchmark says is correct for this task.
    arm: str
    gold_operation: str
    gold_target: str
    gold_text: str
    gold_question: str
    gold_parameter: str
    #: ``None`` on an act row: no question was due, so no question can be late.
    ask_deadline: int | None
    deadline_source: str
    #: Always empty: the runner poses no page listing. Kept so the substrate file has the
    #: same fields as the recorded runs.
    grounding: str = ""
    #: What the task left unspecified, carried from the row rather than joined later.
    #: Every number about asking is computed against these, and a join at scoring time
    #: that matched nothing reported "this task left nothing open" -- indistinguishable
    #: from a task that really was fully specified. Carrying them removes the join.
    open_items: tuple[dict[str, Any], ...] = ()

    @property
    def task_key(self) -> str:
        """The trajectory this step belongs to.

        Keyed on the source as well as the id: ids are unique only within one source, so
        keying on the id alone could merge two different tasks.
        """
        return f"{self.source}::{self.sample_id}"

    @property
    def item_id(self) -> str:
        """This step's identity, unique across the whole run."""
        return f"{self.task_key}__s{self.step_index}"


def write_substrate(items: Sequence[StepItem], out: Path) -> int:
    """Write ``items`` as JSONL and return how many were written.

    ``item_id`` and ``task_key`` are written out rather than left to be recomputed, so a scorer
    reading the file joins on the same key the runner emitted against.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in items:
            record = asdict(item) | {
                "item_id": item.item_id,
                "task_key": item.task_key,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(items)


def read_substrate(path: Path) -> list[StepItem]:
    """The items in a substrate file, in the order they were written.

    The inverse of :func:`write_substrate`. The scorer rebuilds the exact items the runner
    posed, rather than re-deriving them from the benchmark, so a change to the expansion rules
    can never silently re-pose a finished run against different items.
    """
    fields = set(StepItem.__dataclass_fields__)
    items = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record["frames"] = tuple(record.get("frames") or ())
            # JSON has no tuple, so a round trip would otherwise return a list and the
            # rebuilt item would compare unequal to the one that was written.
            record["open_items"] = tuple(record.get("open_items") or ())
            items.append(StepItem(**{k: v for k, v in record.items() if k in fields}))
    return items
