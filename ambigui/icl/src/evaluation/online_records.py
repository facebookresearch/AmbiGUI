# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The per-task record an online rollout produces, and how a rollout ended.

:class:`RolloutResult` is everything one rollout produced; :func:`task_record` is the single
writer of the per-task row in ``scores.json``, so every path through the evaluation writes
the same fields. An absent field would read as a measured zero to anything downstream.

``end_reason`` exists because a turn count is not interpretable on its own. A rollout
that stopped because it finished and one the harness cut off at the step budget both
report a number, and only the reason distinguishes them. The two harness cutoffs --
``max_steps`` and ``repeat_break`` -- right-censor the distribution; see
:mod:`evaluation.turn_metrics`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

#: The policy reported the task impossible. Deliberate, and not success: this is
#: ``report_infeasible`` and nothing else.
END_STOP = "stop"
#: The policy reported the task complete -- ``report_task_complete``. Deliberate, and
#: a claim rather than a verdict: the judge still decides whether the task was done.
#: Kept apart from ``stop`` because a run cannot be read at all if "I finished" and "I
#: cannot do this" land in one bucket.
END_FINISHED = "finished"
#: A generation the harness could not read a tool call out of.
END_NO_TOOL_CALL = "no_tool_call"
#: The serve stopped the generation at the token limit, so there was no tool call
#: to read because the policy had not finished writing one.
#:
#: Separated from ``no_tool_call``, which is "we could not parse what it wrote".
#: A truncated generation is a parsing failure whose cause we know: raise the
#: budget. Filing the two together put a policy that declined to act and a policy
#: cut off mid-sentence in one bucket -- 93 of 246 rollouts in one run -- and the
#: two need different work.
END_TRUNCATED = "truncated"
#: The generation call raised.
END_GENERATION_ERROR = "generation_error"
#: The step budget ran out. Censors the turn count.
END_MAX_STEPS = "max_steps"
#: The loop-breaker fired on a repeated call. Censors the turn count.
END_REPEAT_BREAK = "repeat_break"
#: The start page never loaded, so no turn was ever taken.
END_ENV_UNREACHABLE = "env_unreachable"
#: The harness itself raised. Distinct from ``generation_error``: that one means the
#: serve refused a completion, this one means our code broke. Filing the second as the
#: first is the mistake the ``env_unreachable`` branch was added to stop making.
END_HARNESS_ERROR = "harness_error"
#: The start page loaded but a policy could not act on it: a bot wall, a CloudFront
#: denial, an interstitial, or a page offering nothing to click or type. Distinct from
#: ``env_unreachable``, where nothing loaded at all. A page we were never allowed to use
#: is not a task the policy failed, so this is attributed to the world and the row is
#: set aside from the headline score rather than counted as a failure.
END_ENV_BLOCKED = "env_blocked"
#: The policy pressed the control that commits the task, and the harness did not.
#:
#: A commit cannot be executed on a live site -- an order placed, a donation given, a
#: subscription taken out is real. So the harness absorbs the press and ends the
#: episode here. Deliberate, and like ``finished`` a claim rather than a verdict: the
#: agent got as far as the thing that would have completed the task, and the judge
#: still decides whether what it did up to that point was right. It is its own ending
#: because the alternative was the episode carrying on past the commit until the step
#: budget ran out, which filed a trajectory that reached the goal as ``max_steps``.
END_COMMIT_WITHHELD = "commit_withheld"
#: The task never returned a result -- the concurrency deadline hit first.
END_UNFINISHED = "unfinished"

END_REASONS: frozenset[str] = frozenset(
    {
        END_STOP,
        END_FINISHED,
        END_NO_TOOL_CALL,
        END_TRUNCATED,
        END_GENERATION_ERROR,
        END_MAX_STEPS,
        END_REPEAT_BREAK,
        END_ENV_UNREACHABLE,
        END_ENV_BLOCKED,
        END_HARNESS_ERROR,
        END_COMMIT_WITHHELD,
        END_UNFINISHED,
    }
)

#: The endings the policy chose. Not a grading: each is deliberate, and what makes
#: them one set is that the rollout ran to the length the policy meant it to, so its
#: turn count is a measurement rather than a floor.
#:
#: Used instead of a bare ``== END_STOP`` by everything that asks whether a rollout
#: ran to its natural end. That comparison was written in two readers while ``stop``
#: was the only deliberate ending there was; left alone, each would have gone on
#: reporting every completed rollout as cut short.
SELF_ENDED: frozenset[str] = frozenset({END_STOP, END_FINISHED, END_COMMIT_WITHHELD})


#: The benchmark flag on a row whose recording completed a commitment the agent
#: must not reach. The curated set's own guidance: the asking is fully measurable
#: on these -- the missing details are search parameters the agent has to notice
#: before it filters -- but the end state is deliberately out of reach, so task
#: success is not. Excluded from the success rate and reported beside it.
STOPS_BEFORE_COMMITMENT: Final[str] = (
    "Stops before the commitment the recording completed"
)


def task_key(row: Mapping[str, Any]) -> str:
    """The identifier this row's records are filed under.

    Two benchmarks name the same thing differently: the training store writes
    ``task_id``, the curated online set writes ``sample_id`` and no ``task_id``
    at all. The environment has always read either, but the evaluator
    subscripted ``task_id`` in the two places that open and close a task, so a
    set carrying only ``sample_id`` raised on its first row -- after the serves
    were up and the browser had launched.

    The ``__rN`` suffix distinguishes repeated rollouts of one task; records are
    filed under the task, so it is stripped.

    Raises :class:`ValueError` on a row naming neither. Returning ``""`` would
    file every unnamed row under one key and report them as a single task.
    """
    key = str(row.get("task_id") or row.get("sample_id") or "").split("__r")[0]
    if not key:
        raise ValueError(
            "row carries neither 'task_id' nor 'sample_id', so it names no task"
        )
    return key


def scorable_for_success(row: Mapping[str, Any]) -> bool:
    """Whether this row's task success means anything.

    A row that stops before the commitment cannot reach the end state its
    recording did, so a failure on it measures the benchmark's design and not
    the policy.
    """
    return STOPS_BEFORE_COMMITMENT not in (row.get("flags") or ())


def asks_in_actions(actions: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """The questions the policy asked, in the order it asked them.

    A question the environment failed to record still counts, as the empty string: an
    ask the list dropped would make the question list disagree with the ask count.
    """
    return tuple(
        str((a.get("args") or {}).get("question", ""))
        for a in actions
        if a.get("tool") == "ASK"
    )


def assert_known_end_reason(end_reason: str) -> None:
    """Refuse an ending the run counts under nothing.

    One definition, because both writers need it: a reason absent from
    :data:`END_REASONS` is tallied by no reader, so a row carrying one disappears
    from the totals while still looking like a scored row.
    """
    if end_reason not in END_REASONS:
        raise ValueError(
            f"unknown end_reason {end_reason!r}; expected one of {sorted(END_REASONS)}"
        )


@dataclass(frozen=True)
class PreflightFailure:
    """A task the policy could never be given a first turn on, named.

    An online task's opening observation is its live start page, so a page that will
    not open is not a hard rollout -- it is a rollout that cannot begin. Returned as a
    bare ``None`` it is indistinguishable from a task the policy attempted and failed,
    and the id of the task nobody ran is lost, which lets a sweep report a clean score
    over a set it never ran.

    Detected before the episode rather than during it: a start page that fails halfway
    through turn one spends the policy's budget on an error page and is recorded as the
    policy failing.
    """

    #: The store row this task came from. Online rows are keyed by ``sample_id``.
    sample_id: str
    #: The address that would not open.
    start_url: str
    #: How the run counts it. One of :data:`END_REASONS`.
    end_reason: str
    #: What went wrong, in the words of whatever reported it.
    reason: str

    def __post_init__(self) -> None:
        assert_known_end_reason(self.end_reason)
        if not self.sample_id:
            raise ValueError("a pre-flight failure with no sample_id names no task")

    def __str__(self) -> str:
        return (
            f"[preflight] {self.end_reason} sample_id={self.sample_id} "
            f"start_url={self.start_url or '(none given)'} reason={self.reason}"
        )


@dataclass(frozen=True)
class RolloutResult:
    """Everything one rollout produced.

    A record rather than the tuple this used to be. The tuple had grown to seven
    members, and its unreachable-start-page branch returned three of them while the only
    caller unpacked all seven -- so every task whose start page was unreachable raised
    inside a blanket ``except`` and was logged as a harness crash, on exactly the rows
    the blocked/clean cut exists to separate out.

    ``frozen=True`` fixes the fields, not the dicts inside ``actions``; those come
    straight from the environment and are not copied.

    ``n_turns`` is ``None`` only for a task whose turn count is genuinely unknown -- one
    that never returned a result, or that died in the harness partway through. A rollout
    that took no turn because the page was unreachable reports ``0``: that is measured.
    """

    success: bool | None = False
    n_ask: int = 0
    n_dispatched: int = 0
    judge_error: str | None = None
    n_turns: int | None = None
    end_reason: str = END_UNFINISHED
    #: What the ending was, when the reason alone does not say. ``generation_error``
    #: was the commonest ending of two baseline evaluations -- 52 and 51 of 110 --
    #: and the category carried nothing, so establishing that every one of them was
    #: an HTTP 400 from the policy serve, most of them the 32,768-token context
    #: limit, meant grepping a serve log that happened not to have rotated yet. The
    #: exception belongs on the row that it ended.
    end_detail: str | None = None
    shots: tuple[str, ...] = ()
    actions: tuple[Mapping[str, Any], ...] = ()
    #: What the policy wrote on each turn, and why the serve stopped writing.
    #:
    #: ``no_tool_call`` was the commonest ending of a 246-task run -- 93 of them --
    #: and carried nothing, so there was no way to tell a policy that declined to
    #: act from one truncated mid-sentence by its token budget. Those need
    #: different work. The text is the completion's ``content``. A serve running a
    #: reasoning parser puts the thinking in ``reasoning_content`` instead, and that
    #: is not read here -- so on such a serve this carries the visible answer only.
    turns: tuple[Mapping[str, Any], ...] = ()
    # Appended: preserve all original positional constructors.
    judge_rationale: str | None = None
    judge_version: str = "judge_v1"
    judge_journal_path: str | None = None
    #: Which route reached the start page: ``web_transport.DIRECT`` or the
    #: fallback proxy's label. None when no browser was opened.
    transport: str | None = None
    #: When the rollout ran (UTC, taken by this run at reset). Judge v2 reads it as
    #: the task's time evidence for relative dates such as "today".
    task_time_evidence: Mapping[str, Any] | None = None
    #: Questions on an Ambiguous row with no curated open items. The simulated user
    #: has no intended value for them and gives no answer.
    n_unanswerable_no_curated_intent: int = 0

    @property
    def asks(self) -> tuple[str, ...]:
        return asks_in_actions(self.actions)


def task_record(
    *, task_id: str, ambiguity: str | None, result: RolloutResult
) -> dict[str, Any]:
    """The per-task row written into a scores file's ``results`` array.

    Every path through the evaluator builds its record here, so the key set is identical
    whether the task finished, crashed or never returned.
    """
    assert_known_end_reason(result.end_reason)
    return {
        "task_id": task_id,
        "ambiguity": ambiguity,
        "success": result.success,
        "judge_rationale": result.judge_rationale,
        "judge_version": result.judge_version,
        "judge_journal_path": result.judge_journal_path,
        "n_ask": result.n_ask,
        "n_dispatched": result.n_dispatched,
        "judge_error": result.judge_error,
        "n_turns": result.n_turns,
        "end_reason": result.end_reason,
        "end_detail": result.end_detail,
        "asks": list(result.asks),
        "transport": result.transport,
        "task_time_evidence": dict(result.task_time_evidence) if result.task_time_evidence else None,
    }
