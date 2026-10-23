# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Every call made during a run, kept whole and apart from the marking.

A scoring report says what a number is. It does not say what the models were shown or what
they said back, and once a run is scored those two are the only things that cannot be
reconstructed: the runner asks for temperature zero, but a served model decoding greedily
is still not bit-for-bit reproducible, so re-running it produces different text. A report
disagreeing with the run behind it is then unanswerable.

One row per model call, in ``inference.jsonl``:

* **the exact prompt**, including the question-and-answer prefix as it stood at that call,
  so a reader can see what the policy had been told by then;
* **the reply, verbatim**, before any reading of it;
* **what was read out of it** -- the alternatives, or the phrase saying why none was;
* **which serve answered and what it took**, so a reply that arrived on the third try is
  not indistinguishable from one that arrived first time.

Several calls share one step. The policy is posed again after every question it asks, and
``attempt`` counts those re-poses from 0. Writing only the last of them would hide the
whole exchange and leave the asks file holding questions with no prompts behind them.

No model or network calls: this module records evidence and inference admission state.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

#: A call to the policy being marked.
POLICY = "policy"

#: A call to the independent ask judge, which reads the policy's questions and says which
#: open details each one asks about. It answers nothing to the policy -- the rows under it
#: are written by :mod:`offline_evaluation.run.judge_asks`, after the run has finished.
JUDGE = "judge"

#: A call to the served user simulator that answers the policy's questions in words.
SIMULATED_USER = "simulated_user"

#: A call the simulated user makes to its own model before it answers, asking whether the
#: policy's message is a question or a statement (:mod:`evaluation.message_kind`). Logged
#: before the :data:`SIMULATED_USER` row of the same question, one row per try.
STATEMENT_GATE = "statement_gate"


@dataclass(frozen=True)
class InferenceRow:
    """One call to one model, with everything needed to read it back."""

    task_key: str
    step_index: int
    #: 0 for the first call at this step, and one more for each re-pose after a question.
    #: Judge rows carry the attempt of the policy call whose question they judge.
    attempt: int
    #: :data:`POLICY`, :data:`SIMULATED_USER`, :data:`STATEMENT_GATE` or :data:`JUDGE`.
    role: str
    #: The exact text sent, including the question-and-answer prefix as it stood.
    prompt: str
    #: What came back, verbatim and unread.
    text: str
    error: str | None = None
    finish_reason: str | None = None
    n_frames_sent: int = 0
    #: One string per alternative read out of ``text``. Policy rows only.
    candidates: tuple[str, ...] = ()
    #: Why no action was read out of ``text``, or ``""``.
    rejected: str = ""
    #: ``"name at endpoint"`` of the serve that answered. Not set on policy rows.
    serve: str = ""
    #: Every failed try before this reply, in order. Empty when the first try worked.
    retries: tuple[str, ...] = ()
    #: How a :data:`STATEMENT_GATE` reply was read: ``"question"``, ``"statement"``, or
    #: ``None`` (unreadable, or the call failed and ``error`` says why). ``None`` on every
    #: other role.
    verdict: str | None = None


class InferenceWriter:
    """Persist each completed call before another model can fail, across task threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = Lock()
        path.touch()

    def __call__(self, row: InferenceRow) -> None:
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def write_run_status(run_dir: Path, status: str, reason: str = "") -> None:
    """Atomically update inference admission state without discarding its evidence."""
    path = run_dir / "run.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(status=status, failure_reason=reason)
    tmp = path.with_name("run.json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(config, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def require_complete_run(run_dir: Path) -> dict[str, Any]:
    """Admit complete new runs; legacy configs without status retain coverage checks."""
    config: dict[str, Any] = json.loads(
        (run_dir / "run.json").read_text(encoding="utf-8")
    )
    if "status" in config and config["status"] != "complete":
        raise ValueError(
            f"inference run is {config['status']!r}, not complete: "
            f"{config.get('failure_reason') or 'inference has not finished'}"
        )
    return config
