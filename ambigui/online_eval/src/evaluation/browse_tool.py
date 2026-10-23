# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""One tool call from a policy, resolved into a line of BrowserGym action code.

Training advertises a single ``browse(action)`` tool whose argument is a line of
BrowserGym action code, so a policy trained here emits
``browse(action="click(\'191\')")``. Some callers still present the action under a
differently named tool. This module accepts either and returns the code, raising
:class:`UnknownTool` on a call it cannot place.

Refusing is the point. The online harness once dispatched with
``getattr(env, name, lambda **_: "unknown tool")``, and the environment had no
``browse`` method, so every call the policy made hit that fallback and did nothing.
Across 182 online tasks, 177 produced exactly one frame — the initial page — and
``_ask_count`` never incremented, which is why the ask rate read exactly 0.0000. It
was a dispatch failure wearing the costume of a behavioural finding.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final

from evaluation.tool_contract import compose, wire_actions
from training.bg_web_env import verb_of


class UnknownTool(ValueError):
    """The policy emitted a tool call the environment cannot perform."""


#: Tool names that carry an action rather than being one.
_CARRIERS: Final[frozenset[str]] = frozenset({"browse", "web", "browser"})


def action_code(name: str, args: Mapping[str, Any]) -> str:
    """The line of BrowserGym action code a tool call carries.

    Two shapes are accepted, because two are produced and both are well defined:

    * **a verb and its named arguments** — ``browse(action="click", bid="191")``.
      This is what the schema declares and what the chat templates render, so it
      is what a model emits. The code is built from BrowserGym's own signature,
      which is encoding rather than interpretation: the argument names are
      upstream's, so there is nothing to guess about which is which.
    * **the whole call as one string** — ``browse(action="click('191')")``.

    Neither is a repair of the other. What is *not* accepted is the action as the
    tool name (``click(bid="191")``): the policy was given one tool, and silently
    accepting a different shape is how a harness starts deciding what the policy
    meant.

    Raises:
        UnknownTool: if the call carries no action, names a verb outside the
            action set, or arrives under a tool name that is not the one offered.
    """
    called = (name or "").strip().lower()
    if called not in _CARRIERS:
        raise UnknownTool(
            f"{name!r} is not a tool. The policy is given one tool, browse, whose "
            f"`action` names the verb and whose other arguments are that verb's."
        )
    action = str(args.get("action") or "").strip()
    if not action:
        raise UnknownTool(
            "browse was called with no action. Name one of BrowserGym's actions, "
            "such as action='click' with bid='191'."
        )
    if verb_of(action):
        return action  # already a whole call
    if action not in wire_actions():
        raise UnknownTool(
            f"{action!r} is not an action. Known: {', '.join(wire_actions())}."
        )
    return compose(action, args)


#: A qwen3_coder tool call:
#:
#: .. code-block:: text
#:
#:     <tool_call>
#:     <function=click>
#:     <parameter=target>
#:     Sign in
#:     </parameter>
#:     </function>
#:     </tool_call>
#:
#: Neither the Gemma text parser (``call:name{...}``) nor the OpenAI/Hermes JSON parser
#: reads this, so a policy trained with ``multi_turn.format=qwen3_coder`` had every call
#: parsed as "no tool call at all" and the rollout ended on the first turn.
_FUNCTION = re.compile(r"<function=([A-Za-z_][\w]*)>", re.IGNORECASE)
_PARAMETER = re.compile(
    r"<parameter=([A-Za-z_][\w]*)>(.*?)</parameter>", re.DOTALL | re.IGNORECASE
)


def parse_qwen3_coder_call(text: str) -> tuple[str | None, dict[str, Any]]:
    """Extract ``(name, args)`` from a qwen3_coder tool call; ``(None, {})`` if absent.

    Reasoning before ``</think>`` is discarded, and the **last** call in the reply wins —
    a model that talks through two options before committing should be read as choosing
    the one it ended on.

    A call whose closing tags were cut off by the token budget still parses if the
    function name arrived: a truncated ``<parameter=`` simply contributes nothing, which
    surfaces as an empty target rather than as silence. Silence was the failure mode that
    made 182 rollouts look deliberate.
    """
    body = (text or "").split("</think>")[-1]
    names = list(_FUNCTION.finditer(body))
    if not names:
        return None, {}
    last = names[-1]
    args = {key: value.strip() for key, value in _PARAMETER.findall(body[last.end() :])}
    return last.group(1), args
