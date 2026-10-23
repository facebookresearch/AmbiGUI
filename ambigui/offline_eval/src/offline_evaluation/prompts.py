# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The user message one offline item is posed with.

Kept apart from the parsing and from the serve call because it is the part that decides what
the policy is actually asked, and every published number is a number about *this* text. A
change here is not refactoring -- it invalidates comparability with every earlier run. The text
as it stands now is what the published runs were asked, not what earlier runs were asked: this text last changed when the model was offered the
option of a second action, so a number measured before that change was measured on a
different question and is not comparable with one measured after it.

The governing rule throughout: announce only what is really attached. A model told to look at
a screen it was not given can refuse, describe an imagined page, or spend its budget hunting
for one. Saying so explicitly is no better -- measured on this benchmark, adding
"(no screenshots are available)" halved a text-only model's parse rate, because it answered
by explaining that it could not see the page. So an absent screen is simply omitted.

The worked examples are generic on purpose. A label lifted off a benchmark row would hand
the policy one of the strings it is about to be marked against, inside its own prompt.

Two readings of this text look like improvements and are not, both measured over the same
51 steps at temperature 0 with a 2048-token budget. Writing the line cap as "At most 4
lines in total" instead of "Up to 4" dropped the parse rate from 0.9216 to 0.5882, and
every lost reply had run out of budget rather than come back malformed. Moving the ``ask``
example out of the single list of examples into a block of its own cost a further 0.0785
on top of the better wording. Both read to the model as a demand for more output, and it
deliberates until the budget ends. Keep the cap terse, and keep one list.
"""

from __future__ import annotations

from offline_evaluation.calls import BROWSE_ACTIONS, MAX_ALTERNATIVES

_PROMPT = """You are a web agent. Below is the task{screens}, and everything you have done \
so far.

TASK: {instruction}

WHAT YOU HAVE DONE SO FAR:
{history}

Answer with ONE of these two, and nothing else:
  * one clarifying question, or
  * the action to take now, as one browse call per line.

If, and only if, a second action is just as right for THIS step, put it on the next \
line. Up to {max_alternatives}. They are alternatives, never a plan. \
Think it through once, then give every line at once. Do not explain your choice. Never \
give a question and an action together.

browse(action, target="", text="", question="")
  action   one of: {actions}
  target   visible text / label / URL the action applies to (click/type/select/scroll/goto)
  text     text to type, or the option to select (type and select actions)
  question the clarifying question to ask the user (ask action only)

Examples of the exact output format:
  browse(action="click", target="Sign in")
  browse(action="type", target="Search", text="london")
  browse(action="select", target="Sort by", text="Newest")
  browse(action="ask", question="Which address should I ship to?")

Two alternatives for one step, where either would be right:
  browse(action="click", target="Sign in")
  browse(action="click", target="Log in")

Use action="ask" when the task genuinely does not determine what to do next and guessing \
wrong would be costly. Otherwise act."""

#: Written out rather than left blank: a blank block reads as "the agent did nothing",
#: which is a different claim from "this is the first step".
NO_HISTORY = "(nothing yet - this is the first step)"
#: What the opening line says about what accompanies the task. Announcing something that
#: is not there sends the model looking for it, and saying so explicitly is no better:
#: measured on this benchmark, "(no screenshots are available)" halved a text-only
#: model's parse rate, because it answered by explaining that it could not see the page.
#: So an absent screen is simply omitted, and the prompt neither promises it nor draws
#: attention to its absence.
_SCREENS_CLAUSE = "the most recent screenshots"


def build_offline_prompt(*, instruction: str, history: str, screens: bool = True) -> str:
    """The user message for one offline item.

    ``screens`` says whether screenshots accompany this prompt, and must match what the
    caller actually attaches. A model told to look at a screen it was not given can
    refuse, describe an imagined page, or spend its budget hunting for one.

    An empty history is written out rather than left blank: a blank block reads as "the
    agent did nothing", which is a different claim from "this is the first step". The
    output format is shown by example, not just named — a 2B model given a bare list of
    action names echoes the list back, which is what three of the first pass's failures
    literally were.
    """
    return _PROMPT.format(
        screens=f", {_SCREENS_CLAUSE}" if screens else "",
        instruction=instruction,
        history=history.strip() or NO_HISTORY,
        actions=", ".join(BROWSE_ACTIONS),
        max_alternatives=MAX_ALTERNATIVES,
    )
