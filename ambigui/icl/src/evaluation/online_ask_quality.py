# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Phase 2 of the online evaluation: ask quality, scored after the rollouts.

An online rollout has no steps to be early or late against and no per-row reference
question. What it has is a trajectory, the questions asked in it, and the parameters the
task left open. That is what this scores.

**Why this is a separate pass.** An evaluation pays nothing for an ask, so it has no reason
to consult the ask judge inside the rollout, where a judge outage would cost browser work.
The rollouts answer through the simulated user, record every question
(``ask_tasks.json``), and the questions are scored here once the rollouts are done, so this
pass can be rerun under another judge without repeating the browser work. Without a judge
there is no precision to report, and this says that rather than reporting zero: a zero
reads as "every question was irrelevant", which is a claim about the policy that nobody
made.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from evaluation import simulated_user
from evaluation.ask_f1_metrics import Rollout, summarise
from evaluation.missing_parameters import Parameter

#: The environment variables naming the ask judge.
ENV_VARS: Final[tuple[str, ...]] = (
    "ASK_JUDGE_URL",
    "ASK_JUDGE_MODEL",
)

#: How many judge failures to quote in the block. Enough to recognise the fault,
#: short enough that a scores file stays readable.
_EXAMPLES: Final[int] = 3


@dataclass(frozen=True)
class AskedTask:
    """One evaluated task: what it left open, and what the policy asked about it."""

    task_id: str
    instruction: str
    #: The task's open parameters. Empty means the task had nothing to ask about, so
    #: any question on it is over-asking rather than a missed or hit blocker.
    parameters: tuple[Parameter, ...]
    #: Every question the policy sent, in order. Empty when it never asked.
    questions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """The task as the handover file carries it."""
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "parameters": [
                {"name": p.name, "question": p.question, "answer": p.answer}
                for p in self.parameters
            ],
            "questions": list(self.questions),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AskedTask:
        """One task back out of the handover file."""
        return cls(
            task_id=str(raw["task_id"]),
            instruction=str(raw["instruction"]),
            parameters=tuple(
                Parameter(
                    name=str(p["name"]),
                    question=str(p["question"]),
                    answer=str(p.get("answer", "")),
                )
                for p in raw.get("parameters", ())
            ),
            questions=tuple(str(q) for q in raw.get("questions", ())),
        )


def write_tasks(path: Path, tasks: Sequence[AskedTask]) -> None:
    """Hand the run's questions to the judging pass.

    Every evaluated task is written, including the ones that asked nothing: they are
    the denominator, and a file holding only the askers would describe a run that
    always asked.
    """
    path.write_text(json.dumps([t.as_dict() for t in tasks], indent=2))


def read_tasks(path: Path) -> tuple[AskedTask, ...]:
    """The tasks a run handed over, in the order it evaluated them."""
    return tuple(AskedTask.from_dict(raw) for raw in json.loads(path.read_text()))


def judge_unavailable(env: dict[str, str] | None = None) -> str | None:
    """Why the ask judge cannot score this run, or ``None`` if it can.

    A sentence rather than a boolean, because the ways this fails are not
    interchangeable and the reader of a scores file cannot see the environment the
    job ran in.
    """
    source = os.environ if env is None else env
    for name in ENV_VARS:
        if not (source.get(name) or "").strip():
            return (
                f"{name} is unset. Set the ask judge's endpoint and model; "
                f"question quality is not measured without it."
            )
    return None


def counts(tasks: Sequence[AskedTask]) -> dict[str, int]:
    """Figures that hold whether or not a judge ran.

    Public because the run writes them into its own scores file before any judge
    has seen the questions, and two places counting the same asks differently is
    the defect this module exists to avoid.
    """
    return {
        "n_tasks": len(tasks),
        "n_asks": sum(len(t.questions) for t in tasks),
        "n_tasks_that_asked": sum(1 for t in tasks if t.questions),
    }


#: Marks a message the judge read as a report rather than a question.
_STATEMENT: Final[str] = "\x00statement"


def score(
    tasks: Sequence[AskedTask],
    credit: Callable[..., Sequence[str]] | None = None,
    workers: int = 1,
    statement: Callable[..., bool] | None = None,
) -> dict[str, Any]:
    """The ask-quality block for a scores file.

    ``credit`` is the judge call, defaulting to
    :func:`evaluation.simulated_user.ask_targets`. It is a parameter so the pass can
    be tested without a serve, and so a caller can substitute a recorded verdict.

    ``statement`` is the companion verdict, defaulting to
    :func:`evaluation.simulated_user.was_a_statement`. It reads the reply ``ask``
    already obtained, so it costs no extra call -- but a caller that substitutes
    ``credit`` and not ``statement`` classifies nothing, and every message lands
    in the ask denominator.

    ``workers`` is how many questions are in flight at once. Each is one network
    round trip to the judge and they do not depend on each other, so the pass is
    bound by the serve rather than by the number of questions. The verdicts are
    reassembled in the order the run asked them, so the block does not depend on
    how many ran at once.

    The block always carries the counts. It carries a report only when a judge
    actually scored the questions; otherwise ``applicable`` is ``False`` and
    ``reason`` says what was missing.
    """
    figures = counts(tasks)
    if not tasks:
        return {
            "applicable": False,
            "reason": "no task reached the ask-quality pass; there is nothing to score",
            **figures,
        }
    unavailable = judge_unavailable()
    if unavailable is not None:
        return {"applicable": False, "reason": unavailable, **figures}

    ask = credit or simulated_user.ask_targets
    statement = statement or simulated_user.was_a_statement

    def verdict(job: tuple[AskedTask, str]) -> tuple[list[str], str | None]:
        """One message's credited parameters, and the failure that cost it."""
        task, question = job
        try:
            names = [
                str(name) for name in ask(question, task.instruction, task.parameters)
            ]
            # A message the judge read as a report rather than a question is not
            # an ask, and counting it as one puts a delivered finding in the
            # denominator of every ask ratio. Distinct from a question that
            # covered nothing, which is over-asking and stays counted.
            if statement(question, task.instruction, task.parameters):
                return [], _STATEMENT
            return names, None
        except Exception as exc:  # noqa: BLE001 - one bad call is not the pass
            # Counted, not just dropped. An unscored question is indistinguishable
            # from an irrelevant one inside the ratio, so the number of them has
            # to travel beside the ratio.
            return [], f"{task.task_id}: {type(exc).__name__}: {exc}"[:200]

    jobs = [(task, question) for task in tasks for question in task.questions]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        verdicts = list(pool.map(verdict, jobs))

    per_task: dict[str, list[Rollout]] = {}
    per_task_detail: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    seen = 0
    statements: dict[str, list[str]] = {}
    for task in tasks:
        credited: dict[str, list[str]] = {}
        # Counted, not inferred from ``len(credited)``: that dict is keyed on the
        # question's text, so a policy that asked the same thing three times
        # reported one ask. Repeated asking is the thing the over-asking rate and
        # the precision denominator exist to measure.
        n_ask = 0
        n_rel = 0
        # Parameters already settled by an earlier question on this task. A second
        # ask at one of them is a repeat, which `evaluation.missing_parameters`
        # already defines as earning nothing: counting it relevant let a policy
        # that spams one good question score perfect precision.
        spent: set[str] = set()
        for question in task.questions:
            names, failure = verdicts[seen]
            seen += 1
            if failure == _STATEMENT:
                statements.setdefault(task.task_id, []).append(question[:120])
                continue
            if failure is not None:
                failures.append(failure)
            n_ask += 1
            fresh = [name for name in names if name not in spent]
            if fresh:
                n_rel += 1
                spent.update(fresh)
            credited[question] = names
        # One relevant *ask*, not one per parameter: a question settling three things
        # is still one question, and the aggregator refuses a task whose relevant
        # count exceeds its ask count.
        per_task[task.task_id] = [Rollout(n_ask=n_ask, n_rel=n_rel)]
        per_task_detail[task.task_id] = {"n_ask": n_ask, "n_rel": n_rel}
        if credited:
            # Keyed on the question text, so repeated questions collapse here.
            # `n_ask` above counts them; this shows what the judge said about
            # each distinct one.
            per_task_detail[task.task_id]["credited"] = credited

    gold_ids = {t.task_id for t in tasks if t.parameters}
    return {
        "applicable": True,
        **figures,
        "report": summarise(per_task, gold_ids).as_dict(),
        "per_task": per_task_detail,
        "n_judge_failures": len(failures),
        "judge_failure_examples": failures[:_EXAMPLES],
        # Counted and shown, not dropped quietly: the policy sent these, and how
        # often it reports through the ask channel is itself a measurement.
        # The counts above include statements, because the policy sent them. These
        # two say how many were not questions and what the ratios were computed
        # over, so the block can be reconciled without reading this source.
        "n_statements": sum(len(v) for v in statements.values()),
        "n_asks_scored": sum(r[0].n_ask for r in per_task.values()),
        "statement_examples": [q for qs in statements.values() for q in qs][:_EXAMPLES],
    }
