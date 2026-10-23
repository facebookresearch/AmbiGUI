# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Read one emission back into the action the policy meant, and match it to the recording.

A policy answers with a line of text. Everything downstream -- the per-step mark and the
ask-or-act decision -- first has to turn that line into an action and its
content, then decide whether that action names the same thing the recording did. Both jobs
live here, so every consumer applies the same rule and a change to the rule moves every
number together.

Three emission shapes are accepted, because all three are shapes a policy actually produces
rather than shapes it was promised: the advertised ``browse(action=...)`` call, the JSON a
hermes-format serve emits, and a bare ``click OK``. Rejecting the last two would be
strictness about syntax the model was never shown.

The action word is read the same way. This benchmark measures whether an agent asks when
the task underdetermines what to do, not whether it can reproduce a tool name, so a named
list of synonyms -- ``tap`` for click, ``swipe`` for scroll -- is accepted inside a call.
See :data:`BROWSE_SYNONYM`.

Target matching is deliberately crude and deliberately reported twice. Strict understates --
a policy writes "the current location button" where the recording says "Around current
location" -- and tolerant overstates. Publishing one without the other would hide which.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


#: Words too common to carry a match on their own.
_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
        "button",
        "link",
    }
)

_CALL = re.compile(r"^\s*([a-z_]+)\s*\(\s*(.*?)\s*\)\s*$", re.I | re.S)
#: Where one keyword argument's value starts: its name, and the quote that opens it. The
#: value itself is read by :func:`_read_quoted` rather than by this pattern, because a
#: regular expression cannot tell a closing quote from an escaped one. The closing quote
#: must also MATCH the opening one: accepting either ends the value at the first inner
#: apostrophe, so a question like "Which day's forecast?" arrives as "Which day".
#: Clarifying questions and product names contain apostrophes constantly.
_ASSIGN = re.compile(r"""([a-z_]+)\s*=\s*(["'])""")
#: The escape sequences a policy writes inside a quoted argument. Anything else after a
#: backslash is that character itself, which is what makes ``\"`` a quote and ``\\`` a
#: backslash.
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}


#: The ``action`` values the ``browse`` tool accepts: one tool with an ``action`` argument,
#: addressing elements by their visible text. This is the tool the offline prompt shows the
#: policy (``offline_evaluation.prompts``).
BROWSE_ACTIONS: tuple[str, ...] = (
    "click",
    "type",
    "select",
    "scroll",
    "goto",
    "ask",
    "stop",
)


#: Action words a policy writes that mean one of :data:`BROWSE_ACTIONS`. This benchmark
#: measures whether an agent asks when the task underdetermines what to do. It does not
#: measure whether the agent can reproduce a tool name, so a policy that writes ``tap``
#: has made the click decision and is read as having made it. Recording it as a policy
#: that emitted nothing would answer a question nobody asked.
#:
#: The list is named rather than inferred. A rule that mapped any unknown verb to the
#: nearest action would read ``read`` or ``back`` -- things a policy asks for that this
#: harness does not offer -- as a decision the policy never made.
#:
#: Every verb here is unambiguous as a user-interface action. Common English words that
#: double as action words -- ``open``, ``write``, ``choose``, ``finish`` -- are left out,
#: because :func:`_parse_one` also reads a bare ``click OK`` line, and ``Navigate to the
#: checkout page.`` has that shape. The bare shape is restricted to the canonical verbs
#: for the same reason; see :func:`_parse_one`.
#:
#: Unambiguous means *the same action under another name*, not *near enough*. ``clear``
#: was here and is not a synonym for ``type``: it empties a field, and read as a type it
#: becomes a type with no text, which is a different action from the one the policy asked
#: for. A policy that writes ``clear`` is now read as having emitted nothing, which is the
#: honest reading -- the harness offers no way to express it.
#:
#: The recording's side has its own map, :data:`~offline_evaluation.substrate.
#: BROWSE_EQUIVALENT`. The two overlap, and must never disagree about a shared verb.
BROWSE_SYNONYM: Mapping[str, str] = {
    "tap": "click",
    "left_click": "click",
    # Arity is dropped on purpose. The benchmark records a click and offers one click, so
    # there is nothing finer to be right or wrong about; reading a double or right click
    # as "no action at all" would mark a decision the policy did make as one it did not.
    "double_click": "click",
    "right_click": "click",
    "triple_click": "click",
    "input": "type",
    "enter_text": "type",
    "set_text": "type",
    "select_option": "select",
    "swipe": "scroll",
    "navigate": "goto",
    "open_url": "goto",
    "finished": "stop",
    "terminate": "stop",
    # Asking is the behaviour this benchmark exists to measure. A question read as
    # nothing does not merely lose a step -- it moves the headline ask rate down, which
    # is the number the whole comparison turns on.
    "ask_user": "ask",
    "ask_question": "ask",
    "clarify": "ask",
    "request_clarification": "ask",
}


def canonical_action(written: str) -> str:
    """The action ``written`` names, or ``""`` when it names none.

    Accepts the canonical verb and every synonym in :data:`BROWSE_SYNONYM`.
    """
    name = (written or "").strip().lower()
    name = BROWSE_SYNONYM.get(name, name)
    return name if name in BROWSE_ACTIONS else ""


#: Wrapper tokens a serve puts around the final answer, which are not part of the answer.
#: GLM emits ``<|begin_of_box|>...<|end_of_box|>`` around the call it settled on. Left in
#: place the call does not parse and the step records as unreadable, which scores a model
#: that answered correctly as one that emitted nothing -- on one recorded pass, almost
#: every call was valid and the parse rate still read 0.000. Removing them cannot change any other
#: model's reading: a reply that does not contain the tokens is returned unaltered.
_SENTINELS: tuple[str, ...] = ("<|begin_of_box|>", "<|end_of_box|>")


def strip_sentinels(text: str) -> str:
    """``text`` without the wrapper tokens a serve adds around the final answer."""
    for token in _SENTINELS:
        text = text.replace(token, "")
    return text


def strip_reasoning(text: str) -> str:
    """Whatever follows the last reasoning block.

    A thinking model writes its deliberation first and the call last. Parsing the whole reply
    finds no call and records the row as unparseable -- which scores as a failure to answer
    rather than as a wrong answer, and silently deflates every thinking model. Measured on one
    run, 169 of 170 replies carried a valid call after the marker.
    """
    return text.split("</think>")[-1] if "</think>" in text else text


@dataclass(frozen=True)
class ParsedCall:
    """One emission, broken into its parts.

    ``action`` is ``""`` when nothing could be read. ``args`` holds the named arguments the
    policy actually wrote, which is what a tool-call check needs: whether the argument the
    action requires was supplied, and whether any argument outside the schema was invented.
    ``bare`` marks the ``click OK`` shape, which names no arguments at all, so an argument
    check on it would report a schema violation that the syntax makes impossible.
    """

    action: str
    args: Mapping[str, str]
    payload: str
    bare: bool


#: What every reader returns when it could read nothing.
_EMPTY = ParsedCall("", {}, "", False)


#: How many alternative actions one emission may offer. On many steps more than one action
#: is defensible -- "Sign in" and "Log in" are the same intent against two labels -- and
#: forcing a single guess scores the policy on which synonym it picked. A cap is still
#: needed: an uncapped list lets a policy name every clickable element and be credited for
#: whichever one the recording happens to hold, which measures nothing.
MAX_ALTERNATIVES = 4


@dataclass(frozen=True)
class Candidates:
    """One emission, read as either a question or a set of alternative actions.

    ``calls`` is empty when nothing could be read, and ``rejected`` then says why in one
    phrase, so a blank row can be told apart from a row the reader refused.

    The actions are **alternatives for one step**, not a plan. Nothing downstream may run
    them in sequence, and their order carries no meaning beyond the order they were written.
    """

    calls: tuple[ParsedCall, ...]
    rejected: str

    def __bool__(self) -> bool:
        return bool(self.calls)

    @property
    def asked(self) -> bool:
        """Whether this emission is a clarifying question rather than an action set."""
        return bool(self.calls) and self.calls[0].action == "ask"

    @property
    def question(self) -> str:
        """The question asked, or ``""`` when this emission is an action set."""
        return self.calls[0].payload if self.asked else ""

    @property
    def first(self) -> ParsedCall:
        """The action written first, or the empty call when nothing could be read.

        For callers that record one action per step. The alternatives are a set, so the
        order means only the order they were written; the first is the one the policy
        would take, which is what a single-action column has to hold.
        """
        return self.calls[0] if self.calls else _EMPTY


def parse_candidates(text: str) -> Candidates:
    """One emission as a question or as 1 to :data:`MAX_ALTERNATIVES` alternative actions.

    The format is deliberately exclusive. A question and an action together state two
    different intentions for one step, and either reading of it invents a decision the
    policy did not make, so the pair is refused rather than resolved.

    Repeated identical calls collapse to one. A policy that writes the same call six times
    has still named one action, and reading that as six alternatives would both breach the
    cap and misreport what it offered. More than :data:`MAX_ALTERNATIVES` distinct actions
    is refused whole rather than truncated: keeping the first four would credit a policy for
    a guess it did not commit to and hide that it named more.

    The three single-call shapes (a ``browse(...)`` call, a JSON tool call and a bare
    ``click OK``) are all accepted, under the same rules: a reasoning block and the serve's wrapper tokens come off first, and the bare
    ``click OK`` shape is read only from a one-line reply, because an ordinary sentence
    opening with an action word has the same shape. Formatted JSON is accepted whole --
    see :func:`_units` -- so a pretty-printed tool call is not destroyed by the split.
    """
    raw = strip_sentinels(strip_reasoning(text or "")).strip()
    if not raw:
        return Candidates((), "the reply was empty")
    units = _units(raw)
    schema = [unit for unit in units if _is_schema_call(unit)]
    if not schema:
        bare = _parse_one(units[0]) if len(units) == 1 else _EMPTY
        if not bare.action:
            return Candidates((), "the reply held no call")
        return Candidates((bare,), "")
    calls = _dedupe(_parse_one(unit) for unit in schema)
    if any(call.action == "ask" for call in calls) and len(calls) > 1:
        return Candidates((), "a question and another call were given together")
    if len(calls) > MAX_ALTERNATIVES:
        return Candidates((), f"more than {MAX_ALTERNATIVES} alternatives were given")
    return Candidates(calls, "")


def _dedupe(calls: Iterable[ParsedCall]) -> tuple[ParsedCall, ...]:
    """``calls`` with repeats removed, in the order they were written."""
    seen: dict[tuple[str, str, tuple[tuple[str, str], ...]], ParsedCall] = {}
    for call in calls:
        seen.setdefault(
            (call.action, call.payload, tuple(sorted(call.args.items()))), call
        )
    return tuple(seen.values())


def _parse_one(raw: str) -> ParsedCall:
    """One already-isolated line as a call, or :data:`_EMPTY` if it is not one."""
    args = _as_args(raw)
    if args is not None:
        action = canonical_action(str(args.get("action") or ""))
        if not action:
            return _EMPTY
        named = {str(k): str(v) for k, v in args.items() if k != "action"}
        return ParsedCall(action, named, _payload(action, args), False)
    head, _, rest = raw.partition(" ")
    # The canonical verbs only. A synonym is honoured inside a call, where naming
    # arguments marks it as a deliberate call, and refused here, where nothing does:
    # ``Navigate to the checkout page.`` has exactly this shape and is a sentence.
    action = head.strip("(),\"' ").lower()
    if action in BROWSE_ACTIONS:
        return ParsedCall(action, {}, rest.strip().strip("\"'"), True)
    return _EMPTY


def _as_args(raw: str) -> dict[str, Any] | None:
    """The tool arguments from a JSON or call-style emission, or ``None`` if neither."""
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        inner = obj.get("arguments") or obj.get("args") or obj
        if isinstance(inner, dict):
            return dict(inner)
    m = _CALL.match(raw)
    if not m:
        return None
    return dict(_scan_kwargs(m.group(2)))


def _scan_kwargs(body: str) -> dict[str, str]:
    """The keyword arguments in a call body, read left to right and escape-aware.

    Scanned rather than matched with one expression because the value may contain its own
    quote: ``text="He said \\"hi\\""`` is one argument, and a lazy regular expression ends
    it at the escaped quote, yielding ``He said \\``. Policies write escaped quotes often,
    because page labels and typed text contain them.

    The cursor moves past each value before the next name is looked for, so a ``name=``
    sequence inside a value is never read as a new argument. The first value wins when a
    name repeats, which matches the earlier reader.
    """
    found: dict[str, str] = {}
    at = 0
    while True:
        m = _ASSIGN.search(body, at)
        if m is None:
            return found
        value, at = _read_quoted(body, m.end(), m.group(2))
        found.setdefault(m.group(1), value)


def _read_quoted(body: str, start: int, quote: str) -> tuple[str, int]:
    """The literal that begins at ``start``, and the index just past its closing quote.

    An unterminated literal returns what there is and the end of the body, so a value
    whose closing quote is missing yields the text the policy did write rather than
    nothing at all.
    """
    out: list[str] = []
    at = start
    while at < len(body):
        char = body[at]
        if char == "\\" and at + 1 < len(body):
            out.append(_ESCAPES.get(body[at + 1], body[at + 1]))
            at += 2
            continue
        if char == quote:
            return "".join(out), at + 1
        out.append(char)
        at += 1
    return "".join(out), at


def _as_json(text: str) -> Any:
    """``text`` decoded as JSON, or ``None`` when it is not JSON."""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _units(raw: str) -> list[str]:
    """The reply split into the pieces that may each hold one call.

    A line is the usual piece, but formatted JSON is not one line. A serve asked for its
    tool call pretty-printed emits an object spread over six lines, and splitting on
    newlines destroys every one of them, so the step records as unreadable although the
    policy answered in the format it was told to use.

    So: a reply that is one JSON array is its elements, each element being one candidate.
    Otherwise every balanced ``{...}`` run that decodes as JSON is a piece, and the lines
    those pieces do not already cover are pieces too. Both are needed: a reasoning serve
    writes its thought as a small JSON object and the call on the line below it, and
    keeping only the blocks would drop the call and record the step as unanswered.

    A line inside a block is left out, so a formatted object is read once rather than as
    one call plus five fragments.
    """
    whole = _as_json(raw)
    if isinstance(whole, list):
        return [json.dumps(item) for item in whole]
    blocks = [block for block in _json_blocks(raw) if _as_json(block) is not None]
    covered = "\n".join(blocks)
    loose = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and line.strip() not in covered
    ]
    return blocks + loose


def _json_blocks(raw: str) -> list[str]:
    """Every balanced ``{...}`` run in ``raw``, ignoring braces inside quoted strings.

    Quote tracking is the point: a target of ``"{menu}"`` holds braces that close nothing,
    and counting them splits the object in the wrong place.
    """
    blocks: list[str] = []
    depth, start, quote, escaped = 0, 0, "", False
    for at, char in enumerate(raw):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "{":
            start = at if depth == 0 else start
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                blocks.append(raw[start : at + 1])
    return blocks


def _is_schema_call(line: str) -> bool:
    """Whether ``line`` is a call written in the advertised schema."""
    args = _as_args(line)
    return args is not None and bool(canonical_action(str(args.get("action") or "")))


def _payload(action: str, args: Mapping[str, Any]) -> str:
    """The argument that carries the action's content.

    ``select`` carries it in ``text``, like ``type``: ``target`` names the control and
    ``text`` names the option chosen inside it, which is what the recording holds.

    ``ask`` and ``select`` have no fallback. Borrowing ``target`` would turn a malformed
    ask into a question the policy never wrote, and the judge would then grade that
    invention; for a select it would report the control as the option chosen, which is
    the comparison this reader exists to stop making.

    ``type`` keeps its fallback. The same argument applies to it, but it has had the
    fallback for every pass scored so far, and removing it here would move numbers that
    have nothing to do with the option a select carries.
    """
    key = {"type": "text", "select": "text", "ask": "question"}.get(action, "target")
    fallback = "" if action in {"ask", "select"} else str(args.get("target") or "")
    return (str(args.get(key) or "") or fallback).strip()


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP}


def target_match(predicted: str, gold: str, *, tolerant: bool) -> bool:
    """Whether ``predicted`` names the same element as ``gold``.

    Strict is an exact match after lower-casing and collapsing whitespace. Tolerant
    requires **at least half** of the gold's content words, so a reworded target counts —
    "the current location button" against a label of "Around current location" shares
    2 of 3 — while a coincidental stopword overlap does not. Half is loose: "San Diego"
    matches a target of "San Francisco". Requiring all of them would reject nearly every
    real rewording.
    """
    pred, want = (predicted or "").strip(), (gold or "").strip()
    if not pred or not want:
        return False
    if not tolerant:
        return " ".join(pred.lower().split()) == " ".join(want.lower().split())
    gold_tokens = _tokens(want)
    if not gold_tokens:
        return False
    shared = len(gold_tokens & _tokens(pred))
    return shared * 2 >= len(gold_tokens)
