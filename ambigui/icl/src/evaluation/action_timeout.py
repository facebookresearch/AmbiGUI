# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Put a floor under how long a browser action may take.

BrowserGym's action primitives do not take their timeout from the Playwright
context. Each one writes it out: ``elem.click(..., timeout=500)``,
``elem.fill(..., timeout=500)``, and so on through every primitive in
``browsergym.core.action.functions``. ``set_default_timeout`` cannot reach an
explicit argument, so the value a run configures is ignored and every action gets
half a second to find its element, scroll it into view and complete.

That is not enough on a live third-party page. In one 126-task baseline, 391 of
557 failed actions failed with ``Timeout 500ms exceeded`` -- and the failure is
quiet: the action does nothing, the page is unchanged, the policy sees the same
screenshot and repeats itself, and the episode ends on the loop breaker. Read off
the metrics that looks like a policy that cannot make progress.

Rather than fork the action primitives, this raises the floor one level down, on
the Playwright locator they all call. A timeout below the floor becomes the floor;
anything at or above it, including ``0`` for *no limit*, is left as the caller
asked.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from playwright.sync_api import Locator

#: The floor, in milliseconds, when a run configures none.
#:
#: BrowserGym's own task default for the page context is 5000ms, so this is the
#: budget the harness was written against rather than a number chosen here. It is
#: a floor on one action in a deadlined rollout, not on the rollout.
DEFAULT_ACTION_TIMEOUT_MS = 5000

#: The locator methods an action primitive calls with a written-out timeout.
#:
#: Taken from the bodies in ``browsergym.core.action.functions`` and
#: ``browsergym.core.action.utils``. A method missing here is an action that still
#: fails in 500ms, and nothing in a run would say which one.
TIMED_METHODS = (
    "check",
    "clear",
    "click",
    "dblclick",
    "fill",
    "focus",
    "hover",
    "press",
    "scroll_into_view_if_needed",
    "select_option",
    "set_input_files",
    "type",
    "uncheck",
)

#: Marks a method this module has already wrapped, so booting a second environment
#: in the same process does not wrap the wrapper.
_FLOOR_ATTR = "_action_timeout_floor_ms"

_F = TypeVar("_F", bound=Callable[..., Any])


def raise_action_timeout(
    floor_ms: int = DEFAULT_ACTION_TIMEOUT_MS,
    *,
    locator_cls: type = Locator,
) -> None:
    """Make every timed locator call wait at least ``floor_ms`` milliseconds.

    Idempotent and re-callable: a second call with a different floor replaces the
    floor in place rather than adding a second wrapper.

    Args:
        floor_ms: the smallest timeout any action may run with.
        locator_cls: the class to patch. Defaults to Playwright's ``Locator``,
            which is what BrowserGym's action primitives call; the argument exists
            so the behaviour can be tested without a browser.

    Raises:
        ValueError: if ``floor_ms`` is less than 1. Playwright reads ``0`` as *wait
            indefinitely*, and a floor of zero would turn every action into an
            unbounded wait -- one wedged call then stops the whole process.
    """
    if floor_ms < 1:
        raise ValueError(f"action timeout floor must be at least 1ms, got {floor_ms}")

    for name in TIMED_METHODS:
        method = getattr(locator_cls, name, None)
        if method is None:
            continue
        if hasattr(method, _FLOOR_ATTR):
            setattr(method, _FLOOR_ATTR, floor_ms)
            continue
        setattr(locator_cls, name, _with_floor(method, floor_ms))


def _with_floor(method: _F, floor_ms: int) -> _F:
    """``method`` with its ``timeout`` argument held at or above ``floor_ms``."""

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        timeout = kwargs.get("timeout")
        floor = getattr(wrapper, _FLOOR_ATTR)
        # None means Playwright would fall back to the context default, which is
        # the value this floor replaces; 0 means the caller asked for no limit.
        if timeout is None or 0 < timeout < floor:
            kwargs["timeout"] = floor
        return method(*args, **kwargs)

    setattr(wrapper, _FLOOR_ATTR, floor_ms)
    return wrapper  # type: ignore[return-value]
