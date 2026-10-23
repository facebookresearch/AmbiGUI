# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The one tool contract: what a policy may emit, and what the environment performs.

The vocabulary is BrowserGym's, not ours. This module does not define actions; it
reads them off :class:`browsergym.core.action.highlevel.HighLevelActionSet` and
presents them to the policy. BrowserGym parses what comes back.

**Why the project's own vocabulary was withdrawn.** It offered seven actions —
``click``, ``type``, ``select``, ``scroll``, ``goto``, ``ask``, ``stop`` — addressed
by the visible text of a target, and resolved that text against a list of elements
shown to the policy. Two faults followed from the design rather than from any one
bug:

* Text is not a handle. Five fields on one page were all called ``on``; a rendered
  ``SAIL TO`` is ``Sail To`` to the browser. Resolution then had to guess, and the
  guessing grew — substring matches, case folding, collapsing duplicates — until the
  harness was quietly repairing the policy's aim and reporting success for actions
  that landed somewhere else.
* Seven actions could not express the benchmark's own reference solutions, which
  contain hover steps. The policy was graded on tasks whose demonstrated answer used
  an action it was never offered.

BrowserGym addresses elements by ``bid``, a handle it stamps on every node of the
accessibility tree it shows the policy. A bid either exists or it does not, so there
is nothing to guess and no room for the harness to compensate. Its action set is
seventeen actions wide and is the one the published web-agent benchmarks use, which
makes results here comparable to theirs.

**The one addition.** BrowserGym has no verb for *the task is done*. One is added, in
:mod:`evaluation.finish_action`, through BrowserGym's own ``custom_actions`` hook --
so it is read off its definition exactly as the other seventeen are read off theirs,
and there is still no second place where an action is written down. It is listed in
:data:`CUSTOM_ACTIONS` rather than buried in the builder, so what this project added
to somebody else's vocabulary can be read in one line.

**The wire shape.** The policy speaks a single tool, ``browse``, carrying one
argument: a line of BrowserGym action code.

    {"name": "browse", "arguments": {"action": "click('191')"}}

The tool exists only because the trainer dispatches tool calls. It adds no vocabulary
of its own: the string inside is BrowserGym's, is described to the policy by
BrowserGym's own :meth:`~HighLevelActionSet.describe`, and is executed by passing it
to a BrowserGym environment. There is no second place where an action name is
written down, so the drift this module was originally created to end cannot recur —
and it can no longer be ended by agreeing with ourselves, because the definition is
upstream.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping
from typing import Any, Final

from browsergym.core.action.highlevel import HighLevelActionSet

from evaluation.finish_action import report_task_complete

#: The BrowserGym action subsets offered to the policy.
#:
#: ``chat`` carries ``send_msg_to_user``, which is how a BrowserGym agent addresses
#: the person who set the task — it is the action this project measures, because
#: asking a clarifying question is a message to the user and nothing else.
#: ``bid`` is the page-acting set, addressed by bid. ``nav`` is history and
#: navigation. ``infeas`` carries ``report_infeasible``, for a task that cannot be
#: done, which a web agent needs and which a ``stop`` action conflates with success.
#:
#: ``coord`` and ``tab`` are deliberately absent: coordinate actions bypass the
#: accessibility tree, which is the thing that makes an action checkable, and no task
#: in this benchmark spans tabs.
ACTION_SUBSETS: Final[tuple[str, ...]] = ("chat", "bid", "nav", "infeas")

#: The actions this project adds to BrowserGym's, through BrowserGym's own extension
#: point. One of them: a terminal verb for success, which upstream does not have. See
#: :mod:`evaluation.finish_action` for why it is a verb of its own rather than a second
#: meaning for ``send_msg_to_user``.
#:
#: They go in through ``custom_actions`` rather than being described here, so an added
#: action is read off its definition exactly as an upstream one is read off its own.
#: ``"custom"`` is BrowserGym's subset name for them and is kept out of
#: :data:`ACTION_SUBSETS`, which names the upstream subsets this project takes.
CUSTOM_ACTIONS: Final[tuple[Callable[..., Any], ...]] = (report_task_complete,)

#: The single name every tool call carries. The environment has no method of this
#: name: the call carries BrowserGym action code and is never dispatched by
#: ``getattr``.
TOOL_NAME: Final[str] = "browse"

#: The only argument ``browse`` takes.
ACTION_SLOT: Final[str] = "action"


@functools.lru_cache(maxsize=1)
def action_set() -> HighLevelActionSet:
    """BrowserGym's action set, plus :data:`CUSTOM_ACTIONS`, built once.

    ``strict=False`` is BrowserGym's own default for agent use: it accepts an action
    whose arguments are close enough to parse rather than rejecting the step. A
    rejected step costs a turn and teaches the policy nothing about the page.
    """
    return HighLevelActionSet(
        subsets=[*ACTION_SUBSETS, "custom"],
        custom_actions=list(CUSTOM_ACTIONS),
        strict=False,
    )


@functools.lru_cache(maxsize=1)
def wire_actions() -> tuple[str, ...]:
    """Every action name the policy may use, in BrowserGym's spelling.

    Eighteen of them: BrowserGym's seventeen, including ``noop``, which it allows
    unconditionally, and this project's one added terminal verb.
    """
    return tuple(sorted(action_set().action_set))


@functools.lru_cache(maxsize=1)
def action_description() -> str:
    """The description of the action space shown to the policy.

    Generated by BrowserGym, with examples and without the long per-action prose:
    the long form runs to several thousand characters and crowds out the page
    itself, and every action here has an example that shows its argument shape.

    Nothing in this project paraphrases it. A paraphrase is a second declaration of
    the vocabulary, which is the fault this module exists to prevent.
    """
    described: str = action_set().describe(
        with_long_description=False, with_examples=True
    )
    return described


def _top_level_split(text: str) -> list[str]:
    """``text`` split on commas that are not inside brackets or quotes.

    A parameter list contains commas that do not separate parameters:
    ``button: Literal['left', 'middle']`` has two. Splitting naively invents
    parameters that do not exist.
    """
    parts: list[str] = []
    depth = 0
    quote = ""
    current: list[str] = []
    for char in text:
        if quote:
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if "".join(current).strip():
        parts.append("".join(current))
    return parts


#: Annotations BrowserGym writes for a parameter that must go into the call
#: unquoted. Everything else is quoted — a bid is ``str`` upstream, and
#: ``click(821)`` is not the same call as ``click('821')``.
_NUMERIC_ANNOTATIONS: Final[frozenset[str]] = frozenset({"int", "float"})


def required_parameters(signature: str) -> tuple[tuple[str, str], ...]:
    """The parameters of ``signature`` that have no default, in order.

    Each is returned with its declared annotation, which decides whether the value
    is quoted when the call is built.

    ``signature`` is BrowserGym's own declared form, e.g.
    ``click(bid: str, button: Literal['left', ...] = 'left', ...)``. Only the
    ones without a default are asked of the policy: every optional one has a
    sensible default upstream, and a model made to name them all writes more
    tokens for the same action.

    A return annotation after the parameter list is read past rather than refused.
    Upstream declares none, but :data:`CUSTOM_ACTIONS` is typed like the rest of this
    project, and refusing those signatures would return no parameters for them -- so
    ``compose`` would build the verb with no arguments at all and the summary would be
    dropped on the floor.
    """
    opened = signature.find("(")
    closed = signature.rfind(")")
    if opened < 0 or closed < opened:
        return ()
    inner = signature[opened + 1 : closed]
    found: list[tuple[str, str]] = []
    for part in _top_level_split(inner):
        if "=" in part:
            continue
        name, _, annotation = part.partition(":")
        name = name.strip()
        if name and name not in ("self", "*", "/"):
            found.append((name, annotation.strip()))
    return tuple(found)


@functools.lru_cache(maxsize=1)
def signatures() -> dict[str, tuple[tuple[str, str], ...]]:
    """Each action's required parameters, in order, from BrowserGym's own
    declared signatures.

    Not a table written here. A hand-kept copy of somebody else's signatures is a
    second declaration of the vocabulary and it goes stale silently: the argument
    simply stops being passed and the action does something subtly different.
    """
    return {
        name: required_parameters(getattr(action, "signature", "") or "")
        for name, action in action_set().action_set.items()
    }


def compose(action: str, arguments: Mapping[str, Any]) -> str:
    """One line of BrowserGym action code from a verb and its named arguments.

    ``compose("click", {"bid": "821"})`` is ``"click('821')"``. This is encoding,
    not interpretation: the names come from BrowserGym's own signatures, so there
    is nothing to guess about which argument is which. A missing one is passed as
    an empty string rather than dropped, so the action fails at BrowserGym naming
    the parameter instead of silently becoming a different call.

    Whether a value is quoted is decided by the annotation BrowserGym declared,
    not by whether the value happens to look like a number. A bid is a ``str``
    upstream and is usually all digits, so quoting by appearance writes
    ``click(821)`` — which is not the call ``click('821')``.
    """
    verb = (action or "").strip()
    parameters = signatures().get(verb)
    if parameters is None:
        return verb
    rendered = []
    for key, annotation in parameters:
        text = str(arguments.get(key, ""))
        rendered.append(text if annotation in _NUMERIC_ANNOTATIONS else repr(text))
    return f"{verb}({', '.join(rendered)})"


def call(action: str, **arguments: str) -> str:
    """One assistant tool call, in the shape the schema declares.

    ``call("click", bid="191")`` returns::

        {"name": "browse", "arguments": {"action": "click", "bid": "191"}}
    """
    payload = {ACTION_SLOT: action, **arguments}
    return json.dumps({"name": TOOL_NAME, "arguments": payload}, ensure_ascii=False)


#: What each of BrowserGym's argument names holds, in the policy's terms. Keyed by
#: the name BrowserGym itself uses, so a parameter cannot be documented under a name
#: the environment does not accept.
SLOT_DESCRIPTIONS: Final[dict[str, str]] = {
    "bid": "The element's handle, shown in square brackets in the accessibility "
    "tree: in `[191] link 'Home'` the bid is 191",
    "value": "The text to put into the field",
    "options": "The option to choose",
    "text": "What to say to the user",
    "reason": "Why the task cannot be done",
    "summary": "What you did, in one line, now that the task is complete",
    "key_comb": "A key or combination, such as Enter or Control+a",
    "url": "An absolute address to open",
    "delta_x": "How far to scroll sideways, in pixels",
    "delta_y": "How far to scroll down, in pixels; negative scrolls up",
    "from_bid": "The handle of the element to drag",
    "to_bid": "The handle of the element to drop it on",
    "file": "The path of the file to upload",
}


def tool_json_schema() -> dict[str, Any]:
    """The tool schema to advertise to a policy, in OpenAI function-tool shape.

    One tool whose ``action`` names the verb and whose other properties are
    BrowserGym's own argument names. The policy emits one parameter per argument,
    which is what the chat templates render and what a model trained on them
    produces; a schema that asked for the whole call as one string got the
    arguments split across separate parameters anyway, and 31 of 31 actions in a
    real training step arrived unparseable.
    """
    properties: dict[str, Any] = {
        ACTION_SLOT: {
            "type": "string",
            "enum": list(wire_actions()),
            "description": "Which action to take.",
        }
    }
    for _verb, parameters in sorted(signatures().items()):
        for key, _annotation in parameters:
            if key in properties:
                continue
            takes = sorted(
                v
                for v, ps in signatures().items()
                if any(name == key for name, _ in ps)
            )
            properties[key] = {
                "type": "string",
                "description": f"{SLOT_DESCRIPTIONS.get(key, key)} "
                f"({'/'.join(takes)}).",
            }
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Take one action on the live web page.\n\n" + action_description()
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [ACTION_SLOT],
            },
            "return": {
                "type": "string",
                "description": (
                    "The resulting page: its accessibility tree with a bid on every "
                    "element, a screenshot, and any error the action raised."
                ),
            },
        },
    }


def tool_prompt() -> str:
    """The tool section of a system prompt, generated from the contract itself.

    Written here rather than by hand in each harness: a prompt advertising a
    different shape from the one the environment executes is the divergence this
    module exists to close.
    """
    return "\n".join(
        [
            "You have exactly one tool, `browse`. Its `action` argument names "
            "which action to take, and the action's own arguments go beside it:",
            "",
            "    " + call("click", bid="191"),
            "    " + call("fill", bid="303", value="Miami"),
            "    " + call("send_msg_to_user", text="Which port would you like?"),
            "",
            "The actions listed below are NOT tools and must not be called by "
            'name: `{"name": "click", ...}` will be refused. Name the action in '
            "`browse`'s `action` argument instead.",
            "",
            "Elements are named by the bid in square brackets in the accessibility "
            "tree: in `[191] link 'Skip to Main Content'` the bid is 191.",
            "",
            action_description(),
        ]
    )
