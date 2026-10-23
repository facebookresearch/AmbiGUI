# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The set of details (*parameters*) a task leaves open, and the ask judge's prompt for it.

A task's open parameters are read off its row (:func:`parameters_from_row`). Each has a
name, a reference question and, when the data records one, the answer the person gave.
The ask judge is shown the instruction, one message from the agent and the parameter set
(:func:`targets_prompt`), and says which parameters the message asks about, or that it is
a statement rather than a question (:func:`parse_targets`, :func:`is_statement`). One
question may cover several parameters.

Pure: no I/O and no model calls. The caller owns the judge serve.
"""

from __future__ import annotations

from typing import Final

import json
import re

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from annotation.core.schema import canonicalize


@dataclass(frozen=True)
class Parameter:
    """One thing the task does not determine.

    ``name`` is a short slug used as the credit key (``check_out_date``). ``question`` is
    the reference phrasing for it, kept so a judge can be shown a concrete example rather
    than a bare slot name.
    """

    name: str
    question: str
    #: What this user would answer. Empty when the data does not carry one, in which case
    #: the caller must fall back to a model and accept an invented answer.
    answer: str = ""


_PROMPT = """A user gave a web agent this task. Some details the agent needs are not \
stated in the task, and are listed below as open parameters.

Task: {instruction}

Open parameters:
{parameters}

The agent sent: {question}

First decide what the message is.

A **question** asks the user for information the agent does not have: which city, \
which date, which of two results. A **statement** is not a question -- it reports \
what the agent found, summarises a page, says the task is done, or tells the user \
something. A message can be long and still be a question, and can end in a question \
mark and still be a statement.

If the message is not a question, reply with the single word {statement}.

Otherwise, which open parameters does the question ask about? A single question may \
cover more than one. Reply with their names from the list separated by commas, or the \
word none if it covers none of them. Reply with names only."""


def targets_prompt(
    *, instruction: str, question: str, parameters: Sequence[Parameter]
) -> str:
    """The judge prompt: which open parameters does this question target?

    This asks a different thing from the current rubric. The current one asks whether a
    question matches one specific reference, so a question about a different but equally
    unspecified parameter scores as irrelevant. This asks which parameters are targeted, so
    every open parameter is reachable and a question covering several is credited for all
    of them.
    """
    if not parameters:
        raise ValueError("targets_prompt needs at least one open parameter")
    listed = "\n".join(f"- {p.name}: {p.question}" for p in parameters)
    return _PROMPT.format(
        instruction=instruction,
        parameters=listed,
        question=question,
        statement=STATEMENT,
    )


#: The judge's verdict that the message was not a question at all.
#:
#: The ask channel carries both. ``send_msg_to_user`` is BrowserGym's, its only
#: worked example is a delivered finding, and that example is generated into the
#: policy's prompt verbatim because the vocabulary is declared in exactly one
#: place -- so the policy reports results through it and the ask counter cannot
#: tell those from questions. Splitting them here costs nothing the judge was not
#: already reading.
#:
#: Distinct from ``none``, which means a real question that covered no open
#: parameter. That is over-asking and belongs in the denominator; a statement is
#: not an ask at all.
STATEMENT: Final[str] = "statement"


def is_statement(text: str) -> bool:
    """Whether the judge said this message was not a question.

    Read off the reply's opening word, the same way a refusal is, so a judge that
    explains itself afterwards is still understood.
    """
    lowered = (text or "").strip().lower()
    if "<think>" in lowered and "</think>" not in lowered:
        return False
    tail = lowered.rsplit("</think>", 1)[-1]
    opening = next((x.strip() for x in tail.splitlines() if x.strip()), "")
    return bool(re.match(rf"{STATEMENT}\b", opening))


def parse_targets(text: str, parameters: Sequence[Parameter]) -> tuple[str, ...]:
    """Every parameter named in a judge reply, in the order they appear in ``parameters``.

    Matching is on the parameter name appearing in the reply, so a judge that answers with
    a sentence rather than a bare list is still read.

    ``none`` is a refusal only when the reply *opens* with it. The prompt offers "none" as
    a whole answer, and "none of them relate to check_out_date" is one -- but a judge that
    answers "guests; none of the others apply" has named a target, and reading that as a
    refusal silently loses recall on the control that defines ``blockers_found``.
    """
    lowered = (text or "").strip().lower()
    if not lowered:
        return ()
    # A judge that explains itself -- "Statement: the agent reported the city and
    # the date." -- names parameters in its explanation, and reading those as
    # targets pays credit for a message that is not a question. One verdict, read
    # the same way wherever it is read.
    if is_statement(text):
        return ()
    # Reasoning that never closed is a truncated reply, not an answer. Parsing it reads
    # the model's scratchpad as a verdict -- and a scratchpad weighing each parameter in
    # turn names all of them, which is an over-return the reward would pay for.
    if "<think>" in lowered and "</think>" not in lowered:
        return ()
    tail = lowered.rsplit("</think>", 1)[-1]
    opening = next((x.strip() for x in tail.splitlines() if x.strip()), "")
    # Whole word. "nonetheless" opens with "none", and reading that as a refusal
    # discards every correct target in the same reply.
    if re.match(r"none\b", opening):
        return ()
    # Whole word again, for the opposite reason: "date" sits inside
    # "check_out_date", so a plain substring test credits a parameter the agent
    # never asked about and inflates the recall it is scored on.
    return tuple(
        p.name
        for p in parameters
        if re.search(rf"(?<![a-z0-9_]){re.escape(p.name.lower())}(?![a-z0-9_])", tail)
    )


def parameters_from_details(
    details: Iterable[object] | None,
) -> tuple[Parameter, ...]:
    """Read the open-parameter set off a row's ``open_details``.

    ``open_details`` entries carry ``name``, ``question``, ``answer`` and ``source``.
    A parameter needs a name and a question -- the name is the credit key and the
    question is what the judge is shown. It does not need an answer: a parameter the
    task leaves open is worth asking about even where no answer was recorded, and
    dropping it would understate the underspecification.

    Names must be unique (each is a credit key), so a repeated name
    keeps its first entry. Malformed entries are skipped rather than raised on: these
    lists are model-generated and one bad entry must not lose the whole row. A ``details``
    that is not a sequence of entries at all *is* raised on, because iterating a mapping
    yields its keys, every one of which is skipped -- the row would read as "nothing left
    open" when it carries a full set.
    """
    if isinstance(details, Mapping):
        raise TypeError("open_details must be a sequence of entries, not a mapping")
    out: list[Parameter] = []
    seen: set[str] = set()
    for entry in details or []:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "").strip()
        question = str(entry.get("question") or "").strip()
        if not name or not question or name in seen:
            continue
        seen.add(name)
        out.append(
            Parameter(
                name=name,
                question=question,
                answer=str(entry.get("answer") or "").strip(),
            )
        )
    return tuple(out)


#: Row fields an open-parameter set may arrive in, most recent name first. ``open_items``
#: is what the curated store writes; ``open_details`` is the older name, still present on
#: rows built before the rename; ``open_details_json`` is the same list encoded as a JSON
#: string, which is the only shape a flat parquet ``Features`` schema can carry.
OPEN_PARAMETER_FIELDS = ("open_items", "open_details_json", "open_details")


def open_items_from_row(row: Mapping[str, object]) -> list[dict[str, object]]:
    """The task's open details as raw dicts, from whichever field the row carries.

    The same three fields :func:`parameters_from_row` reads, including the JSON string
    a parquet row carries. Two readers of one thing drift: the environment read only
    the two list fields, so on a parquet row it found none and the simulated user was
    given nothing to answer from -- while the reward, reading the JSON string, scored
    the questions against a set the user could not see.

    Each item keeps its ``answer``, which is what the user states. A malformed string
    yields an empty list, for the reason stated in :func:`parameters_from_row`.
    """
    for field_name in OPEN_PARAMETER_FIELDS:
        raw = row.get(field_name)
        if isinstance(raw, str):
            if not raw.strip():
                continue
            try:
                raw = json.loads(raw)
            except ValueError:
                return []
        if isinstance(raw, Mapping) or raw is None or not isinstance(raw, Iterable):
            continue
        items = [dict(i) for i in raw if isinstance(i, Mapping)]
        if items:
            return items
    return []


def row_left_nothing_open(row: Mapping[str, object]) -> bool:
    """Whether this task left nothing open, as opposed to losing what it left open.

    A row with no open items is ambiguous evidence on its own: a Clear task genuinely
    had nothing to record, while an Ambiguous one whose items failed to join has
    something to record and lost it. Only the first may be told to the user simulator
    as "no preference"; telling it to the second hides a guess. A label outside the
    frozen vocabulary -- a row still in review, say -- is neither, and answers False
    so that the guess stays counted.
    """
    if open_items_from_row(row):
        return False
    try:
        return canonicalize("ambiguity", str(row.get("ambiguity") or "")) == "Clear"
    except ValueError:
        return False


def parameters_from_row(row: Mapping[str, object]) -> tuple[Parameter, ...]:
    """The task's open-parameter set, from whichever field and encoding the row carries.

    Empty for a row that leaves nothing open -- an act row, or an ask row whose items
    never made it onto it. The caller cannot tell those apart here and must not try: a
    row with no set is scored as having no blockers either way.

    A JSON string that does not decode yields an empty set rather than raising. These
    strings are written by a build and read inside a rollout, where a malformed one must
    cost that row its credit and not the run.

    A field holding a *mapping* is the opposite case and is raised on, by falling through
    to ``parameters_from_details``. The two differ in what is known at the point of
    failure. A string that will not decode may have held nothing; there is no way to tell
    from here, so the row loses its credit and the run continues. A mapping is visibly a
    populated set in the wrong shape, and iterating it yields its keys, every one of which
    is skipped -- the row would read as "this task leaves nothing open", which is
    indistinguishable from a genuinely specified task and is the most expensive thing that
    can happen here. Returning an empty set would make this wrapper the one place the
    guard cannot be reached from.
    """
    for field_name in OPEN_PARAMETER_FIELDS:
        raw = row.get(field_name)
        if isinstance(raw, str):
            if not raw.strip():
                continue
            try:
                raw = json.loads(raw)
            except ValueError:
                return ()
        # Absent, or a scalar: the wrong field, so try the next one. A mapping is not
        # skipped here -- it goes on to parameters_from_details, which raises.
        if raw is None or not isinstance(raw, Iterable):
            continue
        found = parameters_from_details(raw)
        if found:
            return found
    return ()
