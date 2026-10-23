# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Call one of several interchangeable serves, with retries, and stop when none answers.

Two stages of the ask protocol put one text to a model that is not the policy: the run
stage asks the simulated user to reply to a question, and the judge stage asks the ask
judge which open details a recorded question covers. The two have nothing in common except
their failure mode, which is why they share this module and why nothing here names either
of them. A pool is one or more interchangeable serves and the rule for giving up on them.

:class:`offline_evaluation.client.ServeClient` never raises. That is right for the policy --
a row that failed is a row the policy did not answer, and it is scored as one. It is wrong
for a serve the measurement depends on. A dead one returns a blank reply, the reader of
that reply takes the blank for an answer, and the run finishes, exits zero, and reports a
policy that never asked usefully. Nothing in the output says the serve was down.

So this module makes that case loud:

* a failed or unusable reply is **retried**, because a serve under load drops requests;
* when every try has failed, the caller **stops** with :class:`ServeUnavailable`
  naming every attempt, instead of writing a complete-looking report of defaults.

Stopping is the point. The operator brings a serve up and runs again; a run that went on
without one would have to be thrown away, and only after someone noticed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from offline_evaluation.client import ServeClient


class ServeUnavailable(RuntimeError):
    """No serve in the pool produced a usable reply, so the run cannot be scored."""


def usable(text: str) -> bool:
    """Whether a reply can be read at all.

    Blank is not a reply. Neither is reasoning that never closed: a reply holding an
    opening ``<think>`` and no closing one spent its whole budget deliberating, and the
    scratchpad weighs every option in turn, so reading it as an answer credits all of
    them. Both are transient -- the same prompt usually answers on the next try -- so both
    are retried rather than accepted.
    """
    stripped = (text or "").strip()
    return bool(stripped) and not ("<think>" in stripped and "</think>" not in stripped)


@dataclass(frozen=True)
class ServeReply:
    """One usable reply, with what it took to get it."""

    text: str
    #: ``"name at endpoint"`` of the serve that answered.
    serve: str
    #: Every failed try before it, in order. Empty when the first try worked.
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServePool:
    """One or more interchangeable serves, tried in order until one gives a usable reply.

    ``clients`` are the serves, in preference order. ``attempts`` is how many times each is
    tried before the next is reached. ``pause`` is the wait between tries, which exists so
    a serve briefly saturated by the policy's own traffic is given time to drain; it is a
    constructor argument so a test runs with no wait.
    """

    clients: tuple[ServeClient, ...]
    attempts: int = 3
    pause: float = 2.0
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("ServePool needs at least one serve; got none")
        if self.attempts < 1:
            raise ValueError(f"attempts must be at least 1; got {self.attempts}")

    def ask_recorded(self, prompt: str) -> ServeReply:
        """One reply, or raise :class:`ServeUnavailable` having tried everything.

        The reply carries which serve answered and what the tries before it cost. A
        reply that arrived on the third try is usable and a warning sign. Returning only the text makes those two
        cases look like a serve that answered immediately.
        """
        failures: list[str] = []
        last = (len(self.clients), self.attempts)
        for index, client in enumerate(self.clients, start=1):
            for attempt in range(1, self.attempts + 1):
                answer = client.answer(prompt, ())
                if answer.error is None and usable(answer.text):
                    return ServeReply(
                        text=answer.text,
                        serve=f"{client.served_name} at {client.endpoint}",
                        failures=tuple(failures),
                    )
                failures.append(
                    f"  {client.served_name} at {client.endpoint}, try {attempt}: "
                    f"{answer.error or 'unusable reply'}"
                )
                # The wait is there to let a serve drain before it is tried again. After
                # the last try of the last serve there is nothing left to try, so it
                # would only delay the failure below.
                if (index, attempt) != last:
                    self.sleep(self.pause)
        raise ServeUnavailable(
            "no serve replied, so this run cannot continue. Bring the serve up and run "
            "again.\n" + "\n".join(failures)
        )


def serve_targets(endpoint: str, model: str) -> tuple[tuple[str, str], ...]:
    """The ``(endpoint, served name)`` pair to call, as a one-serve pool.

    Neither argument has a default. Every caller refuses an empty one before anything is
    posed, and a pool built out of nothing would be a pool pointed at the policy itself.
    One serve only: a comma-separated list is refused rather than read as one URL.
    """
    if "," in endpoint or "," in model:
        raise ValueError(
            f"one serve only; got a comma-separated list: {endpoint!r} / {model!r}"
        )
    return ((endpoint.strip(), model.strip()),)


def describe(targets: Sequence[tuple[str, str]]) -> str:
    """The serves, as one line for the run's log."""
    return ", ".join(f"{name} at {host}" for host, name in targets)
