# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""An ask judge for the offline protocol that first tests whether the text is a request.

A plain parameter judge (such as the online evaluation's set judge) asks one thing:
which open parameters does the agent's question ask about? It presupposes the text is a
question, and then tests only whether the text is *about* a parameter. A statement that
names the value is about it.

That is not a small gap. On one recorded trajectory the policy produced a finished
weather report whose only question mark is in the closing "Is there anything else you'd
like to know?". Such a judge credited it against all three open items of the task. An
open item is a detail the person never stated, which the agent has to settle before it
can act.

This module puts a gate in front of the parameter test. The judge must first say the text
is a request to the user for information the agent does not have. If it is not, no
parameter is credited, whatever the rest of the reply says.

Only the offline protocol's matcher,
:func:`offline_evaluation.run.judge_asks.make_matcher`, calls this module.

A second change comes with the gate: the reply is a per-parameter yes or no, not a list
of names. The list form invites a weak judge to name everything it recognises. That is
what happened on one observed task: the text asks about the date, and the judge returned
three other parameters. A yes or no per line makes each parameter a separate decision.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from evaluation.missing_parameters import Parameter


@dataclass(frozen=True)
class AskVerdict:
    """What the judge said about one text the policy produced.

    ``targets`` is empty whenever ``is_request`` is false, so a caller that only reads the
    targets behaves exactly as it did before this gate existed.
    """

    is_request: bool
    targets: tuple[str, ...]
    #: How the verdict was obtained, carried so the run can be read back. Nothing decides
    #: anything from these four: the exact prompt sent, the reply verbatim, ``"name at
    #: endpoint"`` of the serve that gave it, and every failed try before it.
    prompt: str = ""
    raw: str = ""
    serve: str = ""
    retries: tuple[str, ...] = ()


_PROMPT = """A user gave a web agent a task. Some details the agent needs are not \
stated. They are listed below as open parameters.

Task: {instruction}

Open parameters:
{parameters}

The agent then produced this text:
{text}

Decide two things.

First: is the text a request to the user for information the agent does not have? It is \
NOT a request when it states the information, reports a result, refuses the task, or \
only offers further help. "Is there anything else?" is an offer of help, not a request. \
A text can end in a question mark and still be a statement.

Second: only if the text is a request, decide for each open parameter whether the text \
asks the user to supply that value. Naming the value is not asking for it.

Reply in this form and nothing else:

REQUEST: yes
<name>: yes
<name>: no

If the text is not a request, reply with one line:

REQUEST: no"""


def request_prompt(
    *, instruction: str, text: str, parameters: Sequence[Parameter]
) -> str:
    """The judge prompt: is this text a request, and if so what does it ask for?

    ``text`` is whatever the policy put in the question argument of an ``ask`` call. It is
    named ``text`` rather than ``question`` on purpose: calling it a question is the
    assumption this judge exists to test.
    """
    if not parameters:
        raise ValueError("request_prompt needs at least one open parameter")
    listed = "\n".join(f"- {p.name}: {p.question}" for p in parameters)
    return _PROMPT.format(instruction=instruction, parameters=listed, text=text)


def parse_verdict(reply: str, parameters: Sequence[Parameter]) -> AskVerdict:
    """Read a judge reply in the form the prompt asks for.

    A reader that credits any parameter whose name appears in the text cannot read this
    reply: it names every parameter on both sides of the decision.

    A parameter is credited only by its own line ending in ``yes``. A missing line is a
    no, so a judge that lists two of three parameters has refused the third.
    """
    lowered = (reply or "").strip().lower()
    # Reasoning that never closed is a truncated reply, not an answer. A scratchpad that
    # weighs each parameter in turn names all of them, and reading it as a verdict is the
    # over-return this judge exists to stop.
    if not lowered or ("<think>" in lowered and "</think>" not in lowered):
        return AskVerdict(False, ())
    tail = lowered.rsplit("</think>", 1)[-1]
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not any(re.match(r"request\s*:\s*yes\b", line) for line in lines):
        return AskVerdict(False, ())
    # Whole word on the name, and ``yes`` as the whole value. "date" sits inside
    # "check_in_date", and "no, yes would be wrong" is a refusal that contains "yes".
    return AskVerdict(
        True,
        tuple(
            p.name
            for p in parameters
            if any(
                re.fullmatch(rf"{re.escape(p.name.lower())}\s*:\s*yes", line)
                for line in lines
            )
        ),
    )
