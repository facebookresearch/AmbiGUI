# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The one action this project adds to BrowserGym's: saying the task is done.

BrowserGym's action set has no terminal verb for success. ``report_infeasible`` ends
an episode, and it means *this cannot be done*; ``send_msg_to_user`` does not end
anything, and in this harness it is the clarifying question the whole experiment
measures. So the only deliberate ending a policy could reach was "impossible". An
agent that completed its task had no way to say so: it carried on until the step cap,
and a capped rollout scores zero however the judge read its final screen. Every task-success number taken that way is
a measurement of a harness that could not record success.

**Why a new verb rather than an overload.** Reusing ``send_msg_to_user`` was the
cheaper change and the wrong one: every completion would arrive on the same channel
as every question, and the ask reward is the one signal this experiment turns on.

**Why it is declared here and not written into a prompt.**
:mod:`evaluation.tool_contract` reads the vocabulary off BrowserGym rather than
restating it, because a restatement is a second declaration that goes stale silently.
BrowserGym's own extension point -- ``HighLevelActionSet(subsets=[..., "custom"],
custom_actions=[...])`` -- keeps that property for an added action: it takes the
signature from :func:`inspect.signature`, and the description and the example from the
docstring below. There is still exactly one place the verb is written down, and the
prompt, the tool schema and the executed code all come from it.

The body is the body BrowserGym executes. ``send_message_to_user`` is one of the names
BrowserGym binds into the namespace it runs an action in (see
``browsergym.core.action.functions``), which is also how upstream's own
``send_msg_to_user`` is written. Putting the summary on the chat channel is what lets
the environment read it back off the transcript the episode actually holds, rather
than off anything the harness keeps on the side.

This module deliberately does not use ``from __future__ import annotations``, which
every other module here does. BrowserGym renders the signature it shows the policy
with :func:`inspect.signature`, and under that import every annotation becomes a
string: the prompt would read ``report_task_complete(summary: 'str') -> 'None'``
beside seventeen upstream actions written ``report_infeasible(reason: str)``, and
:func:`evaluation.tool_contract.compose` decides whether to quote an argument by
comparing the annotation against a set of names that a quoted one never matches.
"""

from collections.abc import Callable
from typing import Final

#: Bound by BrowserGym inside the namespace an action is executed in. Declared and not
#: assigned: nothing in this process calls it, and a module-level ``None`` would make a
#: mistake here look like a working action that silently did nothing.
send_message_to_user: Callable[[str], None]


def report_task_complete(summary: str) -> None:
    """
    Report that the task is complete and end the episode. Use this once the user's
    instruction has been carried out, giving a one-line summary of what you did.

    Examples:
        report_task_complete('Added the blue mug to the cart and placed the order.')
    """
    send_message_to_user(summary)  # noqa: F821 - bound by BrowserGym; see above


#: The verb, spelled once. Everything that has to recognise it -- the environment's
#: dispatch, the evaluator's loop, the tests -- imports this rather than typing the
#: string, so the name cannot drift from the function that defines it.
FINISH_VERB: Final[str] = report_task_complete.__name__
