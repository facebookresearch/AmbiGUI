# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Score a pass's clarifying questions by what they asked about, not by when they came.

A timing figure alone never reads the text of a question: a model that stops at the
right step and then asks about the wrong thing would score exactly as well as one that
asks for the information the instruction left out.

This module scores four content rules of increasing strictness over the same tasks. Each
is reported as its own recall, precision and F1, so the four sit side by side in one
table and the gap between them says where a model fails: it asked but about nothing
relevant, or it found one open item and stopped, or it found them all but also asked
three questions nobody needed.

Timing is reported separately (:mod:`offline_evaluation.ask_timing`). Nothing here reads
a deadline, so the content figures and the timing figures move independently.

This module reads a verdict, it does not make one. Each
:class:`AskedQuestion` arrives with the open items a judge already said the question
pins down. For a run of the ask-and-answer protocol that judge is the judge stage
(``offline_evaluation.run.judge_asks``), and its verdict is on the run's own
``asks.jsonl``. So the score
needs no second judge pass and no network, and it is reproducible from the artefacts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

#: The four rules, loosest first. They are nested: a task credited under one is credited
#: under every rule before it. Reporting them in this order makes a drop between two
#: columns readable as the one thing that rule adds.
VARIANTS: tuple[str, ...] = ("any", "at_least_one", "full_set", "full_set_clean")


@dataclass(frozen=True)
class AskedQuestion:
    """One clarifying question, as the open items the judge said it pins down.

    The question's own text is not held here. Nothing in these four rules reads it: the
    judge has already decided what the question covered, and this module scores that
    decision against the task's open items.
    """

    covered: frozenset[str]


@dataclass(frozen=True)
class TaskAsks:
    """Every question one task drew, against the open items that task leaves unsaid.

    ``open_items`` is the set of names the benchmark records for this task. It is empty
    on a task whose set was never recorded, which is a gap in the benchmark rather than
    a property of the model, so such a task is reported apart and kept out of every rate.
    """

    task_key: str
    arm: str
    open_items: frozenset[str]
    questions: tuple[AskedQuestion, ...]

    @property
    def asked(self) -> bool:
        """Whether the model put any question at all on this task."""
        return bool(self.questions)

    @property
    def gradeable(self) -> bool:
        """Whether this task can carry a content score.

        An act task cannot: it has nothing to ask about, so it appears only as a cost to
        precision when the model asks anyway.
        """
        return self.arm == "ask" and bool(self.open_items)

    @property
    def covered(self) -> frozenset[str]:
        """The open items the model's questions reached, over the whole task.

        Restricted to this task's own items. A verdict naming something else is a join
        error, and counting it would let the full-set rule pass on the wrong evidence.
        """
        reached: frozenset[str] = frozenset()
        for question in self.questions:
            reached |= question.covered
        return reached & self.open_items

    @property
    def n_stray(self) -> int:
        """Questions that pinned down none of this task's open items."""
        return sum(1 for q in self.questions if not (q.covered & self.open_items))


def credited(task: TaskAsks, variant: str) -> bool:
    """Whether ``task`` counts as a true positive under ``variant``.

    The two full-set rules require a non-empty ``open_items``. Without that, a task with
    no recorded set and no coverage would satisfy ``covered == open_items`` as two empty
    sets and be credited under the strictest rule while failing the loosest, which is
    the nesting this whole scheme rests on. Such a task is filtered out as ungradeable
    before it reaches here, so the test is a guard, not a code path with a result.
    """
    if variant == "any":
        return task.asked
    if variant == "at_least_one":
        return bool(task.covered)
    if variant == "full_set":
        return bool(task.open_items) and task.covered == task.open_items
    if variant == "full_set_clean":
        return (
            bool(task.open_items)
            and task.covered == task.open_items
            and task.n_stray == 0
        )
    raise ValueError(f"unknown ask variant {variant!r}; expected one of {VARIANTS}")


@dataclass(frozen=True)
class VariantScore:
    """One rule's recall, precision and F1, with the count behind each."""

    variant: str
    n_true_positive: int
    recall: float | None
    precision: float | None
    f1: float | None


@dataclass(frozen=True)
class AskVariantReport:
    """The four content rules over one pass, with every denominator named."""

    n_ask_tasks: int
    n_act_tasks: int
    n_gradeable: int
    n_ungradeable: int
    #: The precision denominator: gradeable ask tasks that asked, plus act tasks that
    #: asked. An ask task with no recorded open items is not in it, because nothing says
    #: whether its question was right, and charging the model for a gap in the benchmark
    #: would read as a model that asked badly.
    n_asked_tasks: int
    #: Ask tasks that asked but cannot be graded, so are absent from the line above.
    #: Non-zero means precision rests on fewer questions than the model actually spent.
    n_asked_ungradeable: int
    n_overask_tasks: int
    #: Every question in the pass, over every task.
    n_questions: int
    #: Questions on a gradeable task that pinned down none of its open items. Counted
    #: over the gradeable tasks only, so read it against ``n_questions_gradeable``.
    n_stray_questions: int
    #: Questions on a gradeable task, the denominator ``n_stray_questions`` belongs to.
    n_questions_gradeable: int
    by_variant: dict[str, VariantScore]

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_ask_tasks": self.n_ask_tasks,
            "n_act_tasks": self.n_act_tasks,
            "n_gradeable": self.n_gradeable,
            "n_ungradeable": self.n_ungradeable,
            "n_asked_tasks": self.n_asked_tasks,
            "n_asked_ungradeable": self.n_asked_ungradeable,
            "n_overask_tasks": self.n_overask_tasks,
            "n_questions": self.n_questions,
            "n_stray_questions": self.n_stray_questions,
            "n_questions_gradeable": self.n_questions_gradeable,
        }
        for variant, score in self.by_variant.items():
            out[f"n_credited_{variant}"] = score.n_true_positive
            out[f"ask_recall_{variant}"] = score.recall
            out[f"ask_precision_{variant}"] = score.precision
            out[f"ask_f1_{variant}"] = score.f1
        return out


def score_variants(tasks: Iterable[TaskAsks]) -> AskVariantReport:
    """The four rules over every task in one pass.

    Recall divides by the gradeable ask tasks. Precision divides by the tasks the model
    asked on at all, gradeable ask tasks and act tasks together, so over-asking costs
    precision exactly as it does in the timing figure. A rate with no denominator is
    ``None`` and not ``0.0``: a pass with nothing to grade has not been measured, and a
    zero in that cell reads as a model that never asked correctly.

    An ask task whose open items were never recorded is in neither denominator, even if
    the model asked on it. ``n_asked_ungradeable`` reports how many questions that hides
    from precision, so the omission is visible rather than silent.
    """
    rows = list(tasks)
    if not rows:
        raise ValueError("score_variants needs at least one task; got none")
    ask = [t for t in rows if t.arm == "ask"]
    act = [t for t in rows if t.arm != "ask"]
    gradeable = [t for t in ask if t.gradeable]
    overask = [t for t in act if t.asked]
    n_asked = sum(1 for t in gradeable if t.asked) + len(overask)
    return AskVariantReport(
        n_ask_tasks=len(ask),
        n_act_tasks=len(act),
        n_gradeable=len(gradeable),
        n_ungradeable=len(ask) - len(gradeable),
        n_asked_tasks=n_asked,
        n_asked_ungradeable=sum(1 for t in ask if t.asked and not t.gradeable),
        n_overask_tasks=len(overask),
        n_questions=sum(len(t.questions) for t in rows),
        n_stray_questions=sum(t.n_stray for t in gradeable),
        n_questions_gradeable=sum(len(t.questions) for t in gradeable),
        by_variant={
            variant: _score(variant, gradeable, n_asked) for variant in VARIANTS
        },
    )


def _score(variant: str, gradeable: Sequence[TaskAsks], n_asked: int) -> VariantScore:
    """One rule, over the tasks that can carry a content score."""
    hits = sum(1 for t in gradeable if credited(t, variant))
    recall = round(hits / len(gradeable), 4) if gradeable else None
    precision = round(hits / n_asked, 4) if n_asked else None
    return VariantScore(
        variant=variant,
        n_true_positive=hits,
        recall=recall,
        precision=precision,
        f1=_f1(recall, precision),
    )


def _f1(recall: float | None, precision: float | None) -> float | None:
    """The harmonic mean, or ``None`` when either side was not measured."""
    if recall is None or precision is None:
        return None
    if recall + precision == 0:
        return 0.0
    return round(2 * recall * precision / (recall + precision), 4)
