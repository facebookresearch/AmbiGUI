# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Guards that stop a run rather than let it report a plausible-looking zero.

Every number this project has had to withdraw was produced by a run that completed
successfully while measuring nothing. The pattern repeats:

* a tool name the environment could not dispatch returned the string ``"unknown tool"``,
  so 182 rollouts never left the landing page and still yielded a task-success rate;
* a judge that answered ``/v1/models`` but returned nothing to a real completion made
  ``score_trajectory`` raise, which a bare ``except`` turned into ``success = False`` on
  every task of seven checkpoints — an exact ``0.0000`` that read as a model result;
* a 1500-token budget meant for non-reasoning models truncated a reasoning model's reply,
  so every row came back ``no_candidates`` and a baseline scored 0 of 182.

The common shape is a failure that degrades to a *valid-looking value* instead of an
error. These guards convert each one back into a loud stop. They are deliberately blunt:
a false abort costs a re-run, while a false pass costs a day of analysis on numbers that
mean nothing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field



class RunUnhealthy(RuntimeError):
    """A run cannot produce a trustworthy number and should stop now."""


@dataclass
class ScoringHealth:
    """Tally of how a run's scoring is actually going, checked as it goes.

    ``judge_errors`` counts tasks where the verdict came from an exception rather than
    from the judge. That distinction is invisible in the output — both arrive as a
    failed task — so it has to be counted at the point it happens.

    The tally is written from the evaluator's worker threads, so the increments are
    taken under a lock. Unlocked, a run whose concurrency deadline fired while workers
    were still finishing could lose an increment and report an ``n_scored`` that
    disagrees with its own task count — in the one field a reader checks to confirm
    those two agree.
    """

    n_scored: int = 0
    n_success: int = 0
    judge_errors: int = 0
    _reasons: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, *, success: bool, judge_error: str | None = None) -> None:
        with self._lock:
            self.n_scored += 1
            self.n_success += bool(success)
            if judge_error:
                self.judge_errors += 1
                if len(self._reasons) < 5:
                    self._reasons.append(judge_error)

    @property
    def judge_error_rate(self) -> float:
        return self.judge_errors / self.n_scored if self.n_scored else 0.0


def assert_scoring_healthy(
    health: ScoringHealth,
    *,
    max_judge_error_rate: float = 0.05,
    min_scored: int = 10,
) -> None:
    """Raise if the judge is failing often enough that the scores are not the model's.

    Below ``min_scored`` nothing is checked — a handful of tasks cannot distinguish a
    broken judge from a hard batch.

    Two conditions stop the run:

    ``judge_error_rate`` over the threshold
        Most verdicts came from an exception handler. Whatever the run reports is the
        failure rate of the judge, not of the policy.
    every task failed *and* any judge error occurred
        A clean sweep of zeros is the signature of a judge that stopped responding. A
        genuinely hopeless policy produces zeros with no judge errors at all, so the
        conjunction is what separates the two.
    """
    if health.n_scored < min_scored:
        return
    examples = "; ".join(health._reasons[:3])
    if health.judge_error_rate > max_judge_error_rate:
        raise RunUnhealthy(
            f"{health.judge_errors}/{health.n_scored} tasks "
            f"({health.judge_error_rate:.1%}) were scored from a judge exception, over "
            f"the {max_judge_error_rate:.0%} threshold. These are judge failures, not "
            f"model failures. Examples: {examples}"
        )
    if health.n_success == 0 and health.judge_errors:
        raise RunUnhealthy(
            f"every one of {health.n_scored} tasks scored a failure, and "
            f"{health.judge_errors} of them hit a judge exception. A clean sweep of "
            f"zeros with judge errors present is a dead judge, not a dead policy. "
            f"Examples: {examples}"
        )
