# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The page half of the canonical form: what changed, and nothing that did not.

CASCP's canonicalizer keeps a running restatement of *the conversation* — what is
being asked and what has been established — and folds each turn into the prior form
rather than restating it. This is the same idea applied to *the page*, and together
they are what the policy is shown.

The page half is deliberately not a language model. "Only add what is genuinely
new" is set algebra over the elements of a tree, and an exact answer is available:
a model asked to do it could omit something the agent then cannot reach, or invent
a handle that does not exist. It is also run once per turn inside the rollout, where
an extra generation costs latency and reproducibility.

**Why stating only the difference is sound.** The conversation is append-only, so
anything said on an earlier turn is still in front of the model. Repeating the whole page every turn re-states what
it can already see, and on one commercial site that is two thousand tokens a turn
of mostly unchanged text.

**Why a difference is not simply an addition.** A bid is a handle on one observation,
not an identifier: BrowserGym re-assigns them as the page changes, and in two real
episodes 12% of bids named more than one element over the episode's life. So three
things have to be said, not one:

* what **appeared** — the agent could not have known of it;
* what **moved** — still on the page, but under a new handle, so the one the agent
  is holding is now wrong;
* what **expired** — gone, so any handle for it must not be used.

The last two are what makes stating a difference safe. Without them the agent keeps
a handle that silently resolves to something else; that was observed twice in ten
live trajectories, and both times the click succeeded on the wrong element.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: One addressable line of a rendered tree: ``[191] link 'Carnival Home Page'``.
_ELEMENT = re.compile(r"\[(\w+)\]\s+(\S+)\s+'([^']*)'")

#: Roles a policy can actually do something with. Everything else is page
#: furniture: it describes the page but cannot be clicked, filled or chosen, and
#: carrying it costs tokens the episode needs for turns.
ACTIONABLE: frozenset[str] = frozenset(
    {
        "link",
        "button",
        "textbox",
        "searchbox",
        "combobox",
        "listbox",
        "option",
        "menuitem",
        "menuitemcheckbox",
        "menuitemradio",
        "checkbox",
        "radio",
        "switch",
        "slider",
        "spinbutton",
        "tab",
        "treeitem",
    }
)


@dataclass(frozen=True)
class PageState:
    """The addressable surface of one page, as the policy would act on it.

    ``elements`` maps each element — identified by what it *is*, not by the handle
    it currently has — to the handles it currently has. An element can hold more
    than one when a page repeats it.
    """

    url: str = ""
    title: str = ""
    elements: Mapping[tuple[str, str], frozenset[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class PageChange:
    """What is different about the page since the policy was last shown it."""

    appeared: tuple[tuple[str, str, str], ...] = ()
    moved: tuple[tuple[str, str, str], ...] = ()
    expired: tuple[tuple[str, str], ...] = ()
    url_changed: tuple[str, str] | None = None
    title_changed: tuple[str, str] | None = None

    @property
    def nothing_changed(self) -> bool:
        return not (
            self.appeared
            or self.moved
            or self.expired
            or self.url_changed
            or self.title_changed
        )


def page_state(url: str, title: str, tree: str) -> PageState:
    """The addressable elements of ``tree``, keyed by what each element is.

    Only actionable roles are kept. A page's headings and paragraphs are visible in
    the screenshot and do not need a handle; what the policy needs from the text is
    the set of things it can name.
    """
    elements: dict[tuple[str, str], set[str]] = {}
    for bid, role, name in _ELEMENT.findall(tree or ""):
        if role not in ACTIONABLE:
            continue
        elements.setdefault((role, name), set()).add(bid)
    return PageState(
        url=url,
        title=title,
        elements={key: frozenset(bids) for key, bids in elements.items()},
    )


def changes(previous: PageState | None, current: PageState) -> PageChange:
    """What the policy has not already been told about ``current``.

    An element whose handle is unchanged is omitted: it was stated on an earlier
    turn and that turn is still in the model's context. An element whose handle has
    changed is reported as moved, with the handle that no longer works, because a
    stale handle does not fail — it resolves to whatever now carries that number.
    """
    if previous is None:
        return PageChange(
            appeared=tuple(
                (bid, *key)
                for key, bids in sorted(current.elements.items())
                for bid in sorted(bids)
            ),
            url_changed=("", current.url) if current.url else None,
            title_changed=("", current.title) if current.title else None,
        )

    appeared: list[tuple[str, str, str]] = []
    moved: list[tuple[str, str, str]] = []
    for key, bids in sorted(current.elements.items()):
        before = previous.elements.get(key)
        if before is None:
            appeared.extend((bid, *key) for bid in sorted(bids))
            continue
        for bid in sorted(bids - before):
            # Same element, different handle. The agent may be holding the old one.
            moved.append((bid, *key))
    expired = [
        (role, name)
        for (role, name), bids in sorted(previous.elements.items())
        if (role, name) not in current.elements
    ]
    return PageChange(
        appeared=tuple(appeared),
        moved=tuple(moved),
        expired=tuple(expired),
        url_changed=(
            (previous.url, current.url) if previous.url != current.url else None
        ),
        title_changed=(
            (previous.title, current.title) if previous.title != current.title else None
        ),
    )


def describe(change: PageChange, *, limit: int = 60) -> str:
    """``change`` as the policy is told it.

    ``limit`` bounds how many elements of each kind are listed. A page that
    replaces its whole body — a navigation — produces hundreds, and listing them
    all would give back the cost this exists to avoid. The count is always stated,
    so a truncated list reads as truncated rather than as the whole page.
    """
    if change.nothing_changed:
        return "The page is unchanged."

    lines: list[str] = []
    if change.url_changed:
        was, now = change.url_changed
        lines.append(f"The page is now {now}" + (f" (was {was})" if was else ""))
    if change.title_changed:
        _, now = change.title_changed
        lines.append(f"Its title is {now!r}")
    if change.expired:
        shown = ", ".join(
            f"{role} {name!r}" if name else role
            for role, name in change.expired[:limit]
        )
        lines.append(
            f"Gone from the page ({len(change.expired)}); any handle you were "
            f"holding for these no longer refers to them: {shown}"
            + (" ..." if len(change.expired) > limit else "")
        )
    if change.moved:
        shown = ", ".join(
            f"[{bid}] {role} {name!r}" for bid, role, name in change.moved[:limit]
        )
        lines.append(
            f"Now under a different handle ({len(change.moved)}); use these and "
            f"not the earlier ones: {shown}"
            + (" ..." if len(change.moved) > limit else "")
        )
    if change.appeared:
        shown = "\n".join(
            f"  [{bid}] {role} {name!r}" for bid, role, name in change.appeared[:limit]
        )
        lines.append(
            f"New on the page ({len(change.appeared)}"
            + (f", first {limit} shown" if len(change.appeared) > limit else "")
            + f"):\n{shown}"
        )
    return "\n".join(lines)


def history_line(action: dict[str, Any]) -> str:
    """One past turn, as the fact the harness recorded about it.

    The element's *name* and not its handle: a handle expires with the page, and a
    history full of expired handles is the thing that gets reused by mistake.
    """
    tool = str(action.get("tool") or "?")
    args = action.get("args") or {}
    name = str(args.get("name") or "")
    result = str(action.get("result") or "").split("\n")[0]
    if tool == "ASK":
        return f"asked the user: {str(args.get('question') or '')!r}"
    if tool == "STOP":
        return "declared the task finished"
    where = f" on {name!r}" if name else ""
    return f"{tool}{where} -> {result}" if result else f"{tool}{where}"
