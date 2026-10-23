# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Persist a rollout so it can be re-judged without being re-run.

A task judge's verdict is a function of the instruction, the ordered screenshots and the
action list. The frames are written to disk as the rollout runs; the actions live in the
environment and die with the process. Without this record, a judge failure could only be
recovered by running the rollout again.

A trajectory written here is self-sufficient: everything the judge reads, beside the
frames it refers to. Re-judging then costs a few minutes and, importantly,
produces a verdict comparable to the original because the judge sees the identical
prompt. Screenshot paths are stored relative to the trajectory file so the whole
directory can be moved or archived without breaking.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: The file written beside a task's frames.
TRAJECTORY_FILE = "trajectory.json"


class IncompleteTrajectory(ValueError):
    """The record cannot reproduce the judge's input, so it is not worth writing."""


@dataclass(frozen=True)
class Trajectory:
    """One rollout, in the form the judge consumes it."""

    #: The benchmark row this rollout came from, with any ``__rN`` repeat suffix
    #: stripped. A record whose id does not match a benchmark row cannot be re-judged
    #: against anything, so this is the one field both writers must agree on.
    task_id: str
    instruction: str
    #: Screenshot filenames in order, relative to the trajectory file's directory.
    shots: tuple[str, ...]
    #: The env's action dicts, in order, exactly as the judge formats them.
    actions: tuple[dict[str, Any], ...] = ()
    #: Clarifying questions the policy asked, kept separately for ask-quality grading.
    asks: tuple[str, ...] = ()
    #: What the original run concluded, so a re-judge can be compared against it.
    original_success: bool | None = None
    original_judge_error: str | None = None
    #: The full id including a repeat suffix, when the rollout is one of several of the
    #: same task. Kept apart from ``task_id`` so matching a benchmark row stays exact.
    rollout_id: str | None = None
    #: Training opens an episode segment per rollout; the online eval does not. Promoted
    #: out of ``meta`` because it identifies the record, and a second writer used to
    #: drop it.
    episode_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # Appended: no positional shift and no invented historical rationale.
    original_rationale: str | None = None
    original_judge_version: str | None = None


def build_trajectory(
    *,
    task_id: str,
    instruction: str,
    shot_paths: list[str],
    actions: list[dict[str, Any]],
    asks: list[str] | None = None,
    success: bool | None = None,
    judge_error: str | None = None,
    judge_rationale: str | None = None,
    judge_version: str | None = None,
    rollout_id: str | None = None,
    episode_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> Trajectory:
    """Assemble a :class:`Trajectory`, reducing absolute shot paths to filenames.

    ``task_id`` is normalised to the benchmark row: a ``__rN`` repeat suffix is moved
    into ``rollout_id``. The two writers disagreed on this -- the environment kept the
    suffix, the online eval stripped it -- so the same rollout could be filed under two
    ids depending on which path wrote last.
    """
    base, _, repeat = str(task_id).partition("__r")
    return Trajectory(
        task_id=base,
        rollout_id=rollout_id or (task_id if repeat else None),
        episode_id=episode_id,
        instruction=instruction,
        shots=tuple(Path(p).name for p in shot_paths),
        actions=tuple(actions),
        asks=tuple(asks or ()),
        original_success=success,
        original_judge_error=judge_error,
        original_rationale=judge_rationale,
        original_judge_version=judge_version,
        meta=dict(meta or {}),
    )


def write_trajectory(directory: Path, trajectory: Trajectory) -> Path:
    """Write ``trajectory.json`` into ``directory`` and return its path.

    Every referenced screenshot must exist. A record naming a frame that is not there
    cannot be re-judged faithfully, and discovering that at re-judge time — long after
    the rollout is unrepeatable — is the situation this module exists to prevent.
    """
    directory.mkdir(parents=True, exist_ok=True)
    missing = [s for s in trajectory.shots if not (directory / s).exists()]
    if missing:
        raise IncompleteTrajectory(
            f"{trajectory.task_id}: {len(missing)} referenced screenshots are not in "
            f"{directory}, e.g. {missing[:3]}. The record could not be re-judged."
        )
    path = directory / TRAJECTORY_FILE
    path.write_text(json.dumps(asdict(trajectory), indent=2, sort_keys=True))
    return path


def persist_rollout(directory: Path, trajectory: Trajectory) -> str | None:
    """Write ``trajectory``, merging with any record already there. Never raises.

    **One record per rollout, written by whoever knows most.** Two call sites write
    this file: the rollout tool after each reward, and the online eval after scoring.
    They never collide -- one writes during training, the other during evaluation, and
    neither runs while the other does. No record has been overwritten.

    The defect was that the two paths produced records of **different shape**, with
    nothing comparing them: training kept the ``__rN`` repeat suffix on ``task_id`` and
    carried ``episode_id`` but no ``judge_error``; the eval stripped the suffix and
    carried ``judge_error`` but no ``episode_id``. A re-judge spanning both stores saw
    the same rollout under two ids with different fields present.

    Merging rather than overwriting is what keeps that true if the paths ever do
    converge: each writer contributes what it knows and neither can erase the other. A
    field is only replaced when the new record actually carries one.

    Returns ``None`` on success, or a one-line description of what went wrong. The
    caller is expected to count it: a rollout must not be lost over bookkeeping, but
    the loss must not be silent either -- that combination is how a sweep came back
    with frames and no records and still reported a clean run.
    """
    try:
        existing_path = directory / TRAJECTORY_FILE
        if existing_path.exists():
            trajectory = _merge(read_trajectory(existing_path), trajectory)
        write_trajectory(directory, trajectory)
    except Exception as exc:  # noqa: BLE001 - reported, never raised at the caller
        return f"{trajectory.task_id}: {type(exc).__name__}: {exc}"
    return None


def _merge(old: Trajectory, new: Trajectory) -> Trajectory:
    """``new`` wins wherever it says something; ``old`` fills the rest."""

    def pick(a: Any, b: Any) -> Any:
        return a if a not in (None, "", (), {}) else b

    return Trajectory(
        task_id=pick(new.task_id, old.task_id),
        instruction=pick(new.instruction, old.instruction),
        shots=pick(new.shots, old.shots),
        actions=pick(new.actions, old.actions),
        asks=pick(new.asks, old.asks),
        # A versioned unresolved new verdict must not inherit an old boolean.
        original_success=(new.original_success if new.original_judge_version is not None else pick(new.original_success, old.original_success)),
        original_judge_error=pick(new.original_judge_error, old.original_judge_error),
        original_rationale=pick(new.original_rationale, old.original_rationale),
        original_judge_version=pick(new.original_judge_version, old.original_judge_version),
        rollout_id=pick(new.rollout_id, old.rollout_id),
        episode_id=pick(new.episode_id, old.episode_id),
        meta={**old.meta, **new.meta},
    )


def read_trajectory(path: Path) -> Trajectory:
    """Load a trajectory written by :func:`write_trajectory`."""
    raw = json.loads(path.read_text())
    return Trajectory(
        task_id=str(raw["task_id"]),
        instruction=str(raw["instruction"]),
        shots=tuple(raw.get("shots") or ()),
        actions=tuple(raw.get("actions") or ()),
        asks=tuple(raw.get("asks") or ()),
        original_success=raw.get("original_success"),
        original_judge_error=raw.get("original_judge_error"),
        original_rationale=raw.get("original_rationale"),
        original_judge_version=raw.get("original_judge_version"),
        rollout_id=raw.get("rollout_id"),
        episode_id=raw.get("episode_id"),
        meta=dict(raw.get("meta") or {}),
    )


def missing_trajectories(root: Path) -> list[str]:
    """Directories that hold frames but no trajectory record, relative to ``root``.

    Reported rather than skipped: a partially persisted sweep would otherwise re-judge a
    subset and present the result as though it covered everything.

    Paths are relative rather than bare directory names because an episode directory is
    only identifiable with its task segment — ``ep0001`` alone names nothing, and two
    tasks can hold an episode of the same name.
    """
    return sorted(
        str(d.relative_to(root))
        for d in root.rglob("*")
        if d.is_dir() and any(d.glob("*.png")) and not (d / TRAJECTORY_FILE).exists()
    )


def episode_dir(root: Path, task_id: str, episode_id: str | None = None) -> Path:
    """Where one episode's frames and trajectory live.

    ``<root>/<task_id>/<episode_id>`` — the episode segment is what keeps concurrent
    rollouts of the same task apart. Training wrote ``<root>/<task_id>`` with no episode
    segment, so on one 24-cell sweep 28,678 rollouts shared 352 directories, up to 128 of
    them overwriting one ``step_00.png``. The action and reward records stayed
    per-rollout, so the logs looked complete while the frames they referenced belonged to
    whichever rollout happened to finish last.

    Without an ``episode_id`` the old flat layout is returned, which is correct for a
    surface that runs each task once — the online evaluation does.
    """
    base = root / (task_id or "task")
    return base / episode_id if episode_id else base


def directory_for_shots(shot_paths: list[str], *, fallback: Path) -> Path:
    """The directory a trajectory must be written to, given where its frames landed.

    :func:`write_trajectory` stores shot *filenames* and verifies each one beside the
    record, so the record has to sit in the same directory as the frames. Inferring that
    from the frames themselves is the only way to stay correct when the caller does not
    know whether the env opened an episode segment: the online evaluation passed
    ``<root>/<task_id>`` while the env wrote ``<root>/<task_id>/<episode_id>``, so every
    rollout that produced a frame raised :class:`IncompleteTrajectory` and was discarded.
    The only records that survived were the frameless ones, which passed the check
    vacuously — leaving a store that looked populated and contained nothing re-judgeable.

    ``fallback`` is used when there are no frames to infer from; such a rollout is still
    a real outcome and must be recorded.
    """
    parents = {Path(p).parent for p in shot_paths}
    if not parents:
        return fallback
    if len(parents) > 1:
        raise IncompleteTrajectory(
            f"frames span more than one directory ({len(parents)}), so no single "
            f"location can resolve the record's relative shot names: "
            f"{sorted(str(p) for p in parents)[:3]}"
        )
    return parents.pop()


def new_episode_id() -> str:
    """A short unique id for one rollout.
    Random rather than derived from the step or rank: an episode needs to be unique
    across concurrent workers, retries and cells, and nothing downstream reads meaning
    from it. Results stay reproducible because this names a directory, not a sample.
    """
    return uuid.uuid4().hex[:12]
