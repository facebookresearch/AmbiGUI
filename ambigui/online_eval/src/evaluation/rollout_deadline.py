#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A wall-clock bound on one rollout.

Every other bound in the online loop is scoped to something smaller than an episode: an
action timeout, a browser-call timeout, a per-task turn cap. A rollout can stay under all of
them and still run for a very long time, so the episode itself gets a deadline
(600 s: ``EPISODE_DEADLINE_S`` in ``scripts/eval_cascp_online.py``).

The environment (``training/bg_web_env.py``) starts the clock when a rollout begins
(:meth:`RolloutDeadlines.start`), registers a closer that ends the browser's process tree
(:meth:`RolloutDeadlines.on_expire`, :func:`kill_process_tree`), and checks the clock at every
turn boundary (:meth:`RolloutDeadlines.check`). A rollout past its deadline has its closers
run and ends as ``unfinished``. A closer ends the *process* rather than calling the browser
library's close method, because the library's synchronous API is bound to the thread that
created it and raises if touched from another one.

The deadline is keyed on a per-rollout object held by weak reference: when the rollout ends
and the object is dropped, its entry is cleaned up. There is no default deadline: how long a
rollout may take is a cost decision, so every run states it.
"""

from __future__ import annotations

import os
import signal
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


def child_pids(pid: int) -> list[int]:
    """The direct child process ids of ``pid``, from ``/proc``.

    Read per-thread, because that is where the kernel keeps the list: a process's
    children are the union of its threads' children, and a browser driver spawned from a
    worker thread appears under that thread, not under the main one. An unreadable or
    absent entry yields nothing rather than raising -- this is called while tearing down
    a process that may already be dying.

    Some kernels have no ``children`` file at all. Then the parent pid of every process
    is read from ``/proc/<pid>/stat`` instead, which finds the same children.
    """
    found: list[int] = []
    try:
        tasks = sorted(Path(f"/proc/{pid}/task").iterdir())
    except OSError:
        return found
    readable = False
    for task in tasks:
        try:
            raw = (task / "children").read_text()
        except OSError:
            continue
        readable = True
        found.extend(int(p) for p in raw.split())
    if not readable:
        found = _children_from_stat(pid)
    return found


def _children_from_stat(pid: int) -> list[int]:
    """The direct children of ``pid``: every process whose parent pid field is ``pid``."""
    found: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return found
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as handle:
                stat = handle.read()
        except OSError:
            continue
        # The command name is in parentheses and may itself hold spaces or ')'.
        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) > 1 and fields[1] == str(pid):
            found.append(int(name))
    return sorted(found)


def kill_process_tree(pid: int) -> list[int]:
    """SIGKILL ``pid`` and everything under it, deepest first. Returns the pids signalled.

    This, and not a library ``close()``, is what frees a wedged rollout. The thread
    blocked in the browser call is waiting to read from the driver process; killing that
    process closes the pipe, which makes the pending read raise, which ends the turn.
    Signalling is safe from any thread, which the browser library's own API is not.

    Deepest first so a supervisor cannot restart a child between the two kills. Children
    are enumerated before the parent dies, because once it is gone they are reparented
    and no longer findable from here.
    """
    signalled: list[int] = []
    for child in child_pids(pid):
        signalled.extend(kill_process_tree(child))
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        # Already gone, or never ours to signal. Either way there is nothing to free.
        return signalled
    signalled.append(pid)
    return signalled


class RolloutExpired(RuntimeError):
    """A rollout outlived its deadline and was reaped."""


@dataclass
class _Running:
    """One rollout being timed."""

    #: Weak reference to the rollout object. Its death is the only signal that the
    #: episode ended, and the only way to tell a live rollout from a dead one whose
    #: ``id()`` has since been reused.
    ref: weakref.ref[object]
    #: When :meth:`RolloutDeadlines.start` first saw it, on the injected clock.
    started: float
    #: What to run when this rollout is reaped -- the teardown that unblocks whatever it
    #: is stuck inside. See :meth:`RolloutDeadlines.on_expire`.
    closers: list[Callable[[], None]] = field(default_factory=list)


class RolloutDeadlines:
    """Wall-clock deadlines for the rollouts running in this worker process.

    Args:
        deadline_s: How long one rollout may take, in seconds. Required and positive:
            there is no length that is right for every run, so there is no default.
        clock: Monotonic time source, injected for tests.
        on_event: Called with one flat dict per event (``start``, ``finish``,
            ``expired``, ``close_failed``), for the run's log.
    """

    def __init__(
        self,
        *,
        deadline_s: float,
        clock: Callable[[], float] = time.monotonic,
        on_event: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        if deadline_s <= 0:
            raise ValueError(f"deadline_s must be > 0; got {deadline_s!r}")
        self._deadline_s = float(deadline_s)
        self._clock = clock
        self._on_event = on_event
        # Keyed by id() and not by the object: the rollout need not be hashable, and
        # holding it as a key would keep it alive forever -- which is precisely the
        # signal being waited on. ``_Running.ref`` guards the id reuse this invites.
        self._running: dict[int, _Running] = {}
        # A closer runs outside the lock (see ``_drop``); the lock guards ``_running``.
        self._lock = threading.RLock()

    # ── what a turn calls ─────────────────────────────────────────────────────

    def start(self, rollout: object) -> float:
        """Begin timing ``rollout``, or return the deadline it already has.

        Calling it again for the same rollout returns the deadline it already has.
        Deliberately **not** a refresh -- a deadline that each turn pushed back would
        bound the gap between turns rather than the episode, and a rollout stuck in a
        fast loop would never reach it.

        Args:
            rollout: The environment's per-rollout episode token, dropped when the
                rollout ends. Must support weak references.

        Returns:
            The absolute time, on this object's clock, at which it expires.

        Raises:
            TypeError: ``rollout`` forbids weak references, so the episode's end could
                not be observed and its entry would outlive it.
        """
        self._reap_finished()
        key = id(rollout)
        with self._lock:
            running = self._running.get(key)
            if running is not None and running.ref() is rollout:
                return running.started + self._deadline_s
            running = _Running(ref=self._weak(rollout), started=self._clock())
            self._running[key] = running
        self._emit("start", deadline_s=self._deadline_s)
        return running.started + self._deadline_s

    def on_expire(self, rollout: object, closer: Callable[[], None]) -> None:
        """Run ``closer`` when this rollout expires.

        Ending the browser process the rollout uses makes any pending browser call raise,
        so the rollout ends. See :func:`kill_process_tree` for why that means a signal
        rather than a ``close()``.

        Registering against a rollout that is not being timed is a no-op.
        """
        running = self._running.get(id(rollout))
        if running is not None and running.ref() is rollout:
            with self._lock:
                running.closers.append(closer)

    def check(self, rollout: object) -> None:
        """Raise :class:`RolloutExpired` if this rollout is already past its deadline.

        It costs a clock read at a turn boundary. A rollout that is not being timed
        passes, because a turn must not fail for arriving before :meth:`start`.
        """
        with self._lock:
            running = self._running.get(id(rollout))
            if running is None or running.ref() is not rollout:
                return
            over = self._clock() - running.started - self._deadline_s
        if over >= 0:
            self._drop(id(rollout), "expired", over_s=round(over, 1))
            raise RolloutExpired(
                f"rollout exceeded its {self._deadline_s:g}s deadline by {over:.1f}s"
            )

    def finish(self, rollout: object) -> None:
        """Stop timing a rollout that ended normally. Idempotent."""
        self._drop(id(rollout), "finish", run_closers=False)

    # ── internals ─────────────────────────────────────────────────────────────

    def _reap_finished(self) -> None:
        # Over a snapshot, under the lock: never iterate the live dict.
        with self._lock:
            dead = [k for k, r in self._running.items() if r.ref() is None]
        for key in dead:
            self._drop(key, "finish", run_closers=False)

    def _drop(
        self,
        key: int,
        event: str,
        *,
        run_closers: bool = True,
        **fields: object,
    ) -> None:
        with self._lock:
            running = self._running.pop(key, None)
        if running is None:
            return
        if run_closers:
            # Outside the lock, deliberately: a closer kills a process tree and can take
            # a moment, and holding the lock across it would stall every other rollout's
            # turn behind the one hang this is freeing.
            for closer in running.closers:
                try:
                    closer()
                except Exception as exc:  # noqa: BLE001 - a teardown fault is not fatal
                    # A closer that throws must not stop the ones after it.
                    self._emit(
                        "close_failed", err=type(exc).__name__, msg=str(exc)[:200]
                    )
        self._emit(event, **fields)

    def _weak(self, rollout: object) -> weakref.ref[object]:
        try:
            return weakref.ref(rollout)
        except TypeError as exc:
            raise TypeError(
                f"{type(rollout).__name__} cannot be weak-referenced, so a deadline "
                "cannot be scoped to it; pass the per-rollout object the agent loop "
                "threads through its turns"
            ) from exc

    def _emit(self, ev: str, **fields: object) -> None:
        # ``active`` on every event: the question a stalled run has to answer is how many
        # rollouts were alive at once, which no per-event field on its own can give.
        if self._on_event is not None:
            self._on_event({"ev": ev, "active": len(self._running), **fields})
