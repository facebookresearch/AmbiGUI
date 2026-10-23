# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Read the gold benchmark's recorded action, which is a sentence rather than a call.

Every other substrate in this package is handed a structured ``gold_action`` with a tool
name and an argument dictionary. The AmbiGUI gold evaluation file records the step as one
English string instead -- ``click "Proton Calendar"``, ``type "Mexican" into "Search"`` --
so a reader has to parse it.

Three rules decide what a shape means.

**A quoted operand is a name the user could see; an angle-bracketed one is not.** The file
writes a developer symbol as ``<imageButton>`` precisely so it cannot be mistaken for a
caption. The gold file's own documentation says not to string-match an agent's output
against one. 141 steps of the released 3,115-step file write their operand that way.
Others keep the operand in quotes but declare a weak ``element_source``; see
:data:`WEAK_ELEMENT_SOURCES`.

**Where there is a separator, the left side is typed and the right side is acted on.**
``type "Mexican" into "Search"`` types the first and targets the second;
``select "Large" from "Size"`` does the same with a different verb.

**A quoted operand after a typing verb is text, not a target.** Steps whose
``element_source`` says no element was recorded can still carry a quoted string, because
the string is what was typed. Reading it as a target would invent an element the recording
never named.

Operations map through :data:`offline_evaluation.substrate.BROWSE_EQUIVALENT`, so a verb
the policy has no way to emit yields an empty operation and leaves the step unscorable
rather than counted as a miss the policy could never avoid. In the released 3,115-step
file that is 443 steps, over three verbs: ``read`` 400, ``enter`` 38, ``back`` 5.

Pure: no model, no network, no randomness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from offline_evaluation.substrate import browse_operation

#: ``element_source`` values naming the element with a developer symbol or a generated
#: phrase rather than text a person saw. 169 steps carry one. The target is dropped on
#: them, so the step is unscorable for target rather than a miss.
WEAK_ELEMENT_SOURCES = frozenset(
    {"view identifier", "element tag only", "field name"},
)

#: Verbs whose single operand is the text that was entered, not the element acted on.
TEXT_VERBS = frozenset({"input", "type", "clear"})

#: Verbs whose single operand names a key, so it is neither an element nor typed text.
#: ``press "Enter"`` acts on no element and enters no text. Reading ``Enter`` as a target
#: would ask an agent to click a button of that name, which was never on the screen.
KEY_VERBS = frozenset({"press", "hotkey"})

#: Bare words that separate the typed value from the element it went into.
SEPARATORS = frozenset({"into", "from"})

#: A quoted string with backslash escapes, an angle-bracketed symbol, or a bare word, in
#: that order. Quoted comes first because three recorded actions carry an escaped quote
#: inside the target -- ``click "33\\" - 49\\""`` -- and splitting on the quote character
#: would cut one of them in half.
_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|<[^<>]*>|\S+')


@dataclass(frozen=True)
class Operand:
    """One argument of a recorded action, with how the file wrote it."""

    text: str
    #: ``quoted`` is a name a person could see, ``angle`` a developer symbol, ``word`` a
    #: bare token such as a verb or a separator.
    kind: str


@dataclass(frozen=True)
class GoldAction:
    """A recorded step, read into the three fields a step is marked on.

    ``operation`` is empty when the policy has no way to express the recorded verb.
    ``target`` is empty when the recording named no element, named it with a developer
    symbol, or the operand was the typed text. ``text`` is empty when nothing was typed.
    An empty field means *not scorable here*, never *the policy got it wrong*.
    """

    operation: str
    target: str
    text: str


def _unescape(value: str) -> str:
    """The contents of a quoted operand, with its backslash escapes resolved."""
    return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")


def tokenise(action: str) -> list[Operand]:
    """Split a recorded action into its verb and operands.

    Quoted operands keep their contents and lose their quotes. Angle-bracketed ones keep
    their brackets, so a caller can tell a developer symbol from a caption without
    consulting ``element_source`` as well.
    """
    out = []
    for match in _TOKEN.finditer(action or ""):
        raw = match.group(0)
        if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
            out.append(Operand(_unescape(raw), "quoted"))
        elif raw.startswith("<") and raw.endswith(">"):
            out.append(Operand(raw, "angle"))
        else:
            out.append(Operand(raw, "word"))
    return out


def parse_gold_action(action: str, element_source: str = "") -> GoldAction:
    """Read one recorded action, dropping anything that cannot be marked.

    ``element_source`` is the file's own statement of where the element name came from.
    A name from :data:`WEAK_ELEMENT_SOURCES` is dropped even when it is quoted, because
    those 169 steps name the element with a symbol rather than with what was on screen.
    """
    tokens = tokenise(action)
    if not tokens:
        return GoldAction("", "", "")
    verb = tokens[0].text.lower()
    operation = browse_operation(verb)
    operands = tokens[1:]
    split = next(
        (
            i
            for i, operand in enumerate(operands)
            if operand.kind == "word" and operand.text.lower() in SEPARATORS
        ),
        None,
    )
    if split is None:
        lone = operands[0] if operands and verb not in KEY_VERBS else None
        typed = lone if lone and verb in TEXT_VERBS else None
        acted = lone if lone and verb not in TEXT_VERBS else None
    else:
        # ``split > 0``, not ``split``: a separator at position 0 means nothing was typed
        # before it, so there is no left operand to read.
        typed = operands[0] if split > 0 else None
        acted = operands[split + 1] if split + 1 < len(operands) else None
    # Only a quoted operand is a value. A bare word here would be a fragment of the
    # sentence, and an angle-bracketed one is a symbol no agent can be asked to produce.
    text = typed.text if typed and typed.kind == "quoted" else ""
    target = (
        acted.text
        if acted
        and acted.kind == "quoted"
        and element_source not in WEAK_ELEMENT_SOURCES
        else ""
    )
    return GoldAction(operation, target, text)
