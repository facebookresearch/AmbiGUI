# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The set of details (*parameters*) a task leaves open.

A task's open parameters are read off its row (:func:`parameters_from_details`). Each has a
name, a reference question and, when the data records one, the answer the person gave.

Pure: no I/O and no model calls.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


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


def parameters_from_details(
    details: Iterable[object] | None,
) -> tuple[Parameter, ...]:
    """Read the open-parameter set off a task's ``open_items``.

    ``open_items`` entries carry ``name``, ``question``, ``answer`` and ``source``.
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
        raise TypeError("open_items must be a sequence of entries, not a mapping")
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
