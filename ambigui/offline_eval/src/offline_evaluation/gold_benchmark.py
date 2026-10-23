# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Turn an AmbiGUI gold evaluation row into the steps the offline protocol scores.

The released gold evaluation file -- ``gold_eval_set.jsonl``, 298 rows, 3,115 steps --
records each step's action as a sentence and carries no structured ``gold_action``,
``dialogue_json`` or ``ask_side`` field. This module is the reader for that file. It
produces the :class:`~offline_evaluation.substrate.StepItem` list the runner and the
scorer read. It adds a second staged lead line for rows with no dialogue.

Four readings here are deliberate, and each is wrong in the obvious way.

**The arm comes from the label, because the file ships no arm.** ``ambiguity`` is
``Ambiguous`` on 143 rows and ``Clear`` on 155. Those are the ask and act arms.

**The dialogue is gated on role as well as on visibility.** A turn is posed when
``visible_from_step`` has been reached. Visibility alone also poses the recorded
``clarifying_answer``, at 893 of the 2,083 dialogue steps across 58 rows -- the recording
answering the very question the ask arm exists to make the model ask. Withholding the two
clarifying roles costs nothing: 0 of 2,083 steps are left without a posable turn.

**Frames are joined against a root that the caller must give.** ``image_path`` is
relative. ``client.py`` counts an unopenable image as ``unreadable`` instead of raising,
so a relative path becomes a quiet zero rather than a crash. A missing frame stops the
load here.

**The two item lists are never merged.** ``open_items`` has answers a simulated user can
give; ``items_from_decision_points`` does not, and on the 117 rows carrying both they
name different parameters. Only ``open_items`` is read.

Pure: no model, no network, no randomness.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from offline_evaluation.gold_actions import parse_gold_action
from offline_evaluation.substrate import (
    DEADLINE_NONE,
    DIALOGUE_LEAD,
    SPEAKERS,
    WITHHELD_ROLES,
    StepItem,
)

#: ``ambiguity`` -> which behaviour the benchmark says is correct. The released file has
#: no ``arm`` or ``ask_side`` key; this label is what stands in for one.
ARM_BY_LABEL: Mapping[str, str] = {"Ambiguous": "ask", "Clear": "act"}

#: ``window.source`` values, in the order of how much they can be trusted. ``recorded``
#: is 12 windows written by hand; ``proposed`` is 301 read off the recording by a model
#: and scored 8 of 12 against the recorded ones. The file's own guidance is to report
#: them separately and never to average them, so the source travels with the deadline.
WINDOW_RECORDED = "recorded"
WINDOW_PROPOSED = "proposed"

#: Opens the posed text on a row with no recorded dialogue -- 183 of the 298 released
#: rows. :data:`~offline_evaluation.substrate.DIALOGUE_LEAD` says the user and the
#: assistant have been talking, which is false on these, so they get their own lead
#: rather than a sentence the row does not support. The scorer's staging check accepts
#: either, because both are written by a staged read and nothing else produces them.
SINGLE_TURN_LEAD = "the user made one request. it is below."

#: The lead lines a staged read writes, and the only two. A step is staged when its
#: posed text opens with one of them; nothing but a staged read produces either. Every
#: reader of a posed instruction (the scorer's staging check) matches against this tuple,
#: so a third lead cannot be added in one place and missed in another.
LEADS = (DIALOGUE_LEAD, SINGLE_TURN_LEAD)


def arm_of(row: Mapping[str, Any]) -> str:
    """``ask`` or ``act`` for this row, or ``""`` when the label is not one of the two."""
    return ARM_BY_LABEL.get(str(row.get("ambiguity") or ""), "")


def gold_open_items(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """The row's curated open items: the ones a simulated user has answers for.

    ``items_from_decision_points`` is deliberately not included. It is read off the
    annotators' decision records and carries no answer, so an ask credited against it
    could never be replied to.
    """
    return tuple(dict(item) for item in row.get("open_items") or ())


def ask_deadline(items: Sequence[Mapping[str, Any]]) -> tuple[int | None, str]:
    """The last step at which asking still helps, and how that step was settled.

    The bound is the tightest one over the items whose value the recording actually
    fixed. An item with ``never_committed`` true has no bound to give: the run never
    settled the value, so its window runs to the last step and "asked in time" would be
    trivially true. 149 of the 313 items are in that state.

    Returns ``(None, "none")`` when the row has no bounded item at all.

    On a tie, ``min`` over ``(latest_step, source)`` picks the alphabetically earlier
    source, so ``proposed`` wins over ``recorded``. That is the conservative choice: it
    never claims a hand-recorded provenance the tightest bound does not uniquely have.
    Per-item provenance is kept intact on ``open_items`` and is what
    :mod:`offline_evaluation.ask_timing` reports, so nothing is lost by this tie-break.
    """
    bounded = []
    for item in items:
        window = item.get("window") or {}
        latest = window.get("latest_step")
        if window.get("never_committed") or not isinstance(latest, int):
            continue
        bounded.append((latest, str(window.get("source") or WINDOW_PROPOSED)))
    if not bounded:
        return None, DEADLINE_NONE
    step, source = min(bounded)
    return step, source


def staged_dialogue(row: Mapping[str, Any], step_index: int) -> str:
    """Every turn an agent standing at ``step_index`` could have heard, as a prompt.

    A turn is posed when ``visible_from_step`` is set and has been reached, and its role
    is neither ``clarifying_question`` nor ``clarifying_answer``. Rows with no dialogue --
    183 of 298 -- fall back to ``instruction`` under :data:`SINGLE_TURN_LEAD`, which is
    the whole conversation on those.

    The fallback fires only on a row that really has no dialogue. A row that has turns
    but none posable at this step raises instead: on the released file that never
    happens -- 0 of the 2,083 dialogue steps are left with no posable turn by the role
    gate -- so staging such a row as a one-request row would be a later drop changing
    what the benchmark means, quietly.

    :raises ValueError: the row has turns, but none can be posed at ``step_index``.
    """
    lines = []
    for turn in row.get("turns") or ():
        visible = turn.get("visible_from_step")
        text = str(turn.get("text") or "").strip()
        if not isinstance(visible, int) or visible > step_index or not text:
            continue
        if str(turn.get("role") or "") in WITHHELD_ROLES:
            continue
        speaker = SPEAKERS.get(str(turn.get("speaker") or ""), "user")
        lines.append(f"{speaker}: {text}")
    if not lines:
        if row.get("turns"):
            raise ValueError(
                f"row has turns but none posable at step {step_index}: "
                f"{row.get('sample_id')!r}"
            )
        return "\n".join(
            [SINGLE_TURN_LEAD, f"user: {str(row.get('instruction') or '').strip()}"]
        )
    return "\n".join([DIALOGUE_LEAD, *lines])


def render_action_history(steps: Sequence[Mapping[str, Any]], upto: int) -> str:
    """The recorded actions before this step, one numbered line each.

    The recorded ``action`` string is posed as written. It is already the sentence a
    person reading the trajectory would want, and rewriting it from the parsed fields
    would drop the detail the parser deliberately does not score, such as a key name.
    """
    return "\n".join(
        f"{position}. {str(step.get('action') or '').strip()}"
        for position, step in enumerate(steps[:upto], start=1)
    )


def resolve_frames(
    steps: Sequence[Mapping[str, Any]], index: int, cap: int, image_root: Path
) -> tuple[str, ...]:
    """The ``cap`` most recent screens up to and including the one at ``index``.

    Never a later screen: the image at step ``i`` is the screen **before** action ``i``
    ran, so the image at ``i + 1`` shows the effect and would answer the question an ask
    row exists to test.

    Raises :class:`FileNotFoundError` on a frame that is not on disk, and on a step that
    records no frame at all. A silent skip here becomes a blind rollout that still
    reports a score: skipping step ``i``'s own image shows the model step ``i - 1``'s
    screen as though it were the current one. Every one of the 3,115 steps of the rebuilt file
    records an image, so this raises on a later drop rather than on anything shipped.
    """
    if cap <= 0:
        return ()
    paths = []
    for position, step in enumerate(steps[: index + 1]):
        relative = str(step.get("image_path") or "").strip()
        if not relative:
            raise FileNotFoundError(f"step {position} records no image_path")
        frame = image_root / relative
        if not frame.is_file():
            raise FileNotFoundError(f"frame missing under the image root: {relative}")
        paths.append(str(frame))
    return tuple(paths[-cap:])


def expand_row(
    row: Mapping[str, Any], *, frame_cap: int, image_root: Path | None
) -> list[StepItem]:
    """Every scorable step of one gold row.

    ``len(steps)`` is the step count. The row's own ``n_steps`` disagrees with it on all
    298 rows, so nothing here reads that key.

    With ``image_root`` None the frames are left empty; :func:`expand` resolves them later,
    for the steps it keeps.
    """
    steps = list(row.get("steps") or ())
    if not steps:
        return []
    items = gold_open_items(row)
    deadline, deadline_source = ask_deadline(items)
    # ``gold_question``/``gold_parameter`` are single-valued on StepItem and carry the
    # first open item, while the deadline is the tightest bound over all of them. On a
    # multi-item row those describe different items. Nothing in the ask protocol reads
    # the two question fields -- the judge matches against the whole ``open_items``
    # tuple, which is carried intact -- so they are a label for a reader, not an input.
    first = items[0] if items else {}
    out = []
    for index, step in enumerate(steps):
        parsed = parse_gold_action(
            str(step.get("action") or ""),
            str(step.get("element_source") or ""),
        )
        out.append(
            StepItem(
                source=str(row.get("source") or ""),
                sample_id=str(row.get("sample_id") or ""),
                step_index=index,
                instruction=staged_dialogue(row, index),
                history=render_action_history(steps, index),
                frames=(
                    ()
                    if image_root is None
                    else resolve_frames(steps, index, frame_cap, image_root)
                ),
                arm=arm_of(row),
                gold_operation=parsed.operation,
                gold_target=parsed.target,
                gold_text=parsed.text,
                gold_question=str(first.get("question") or ""),
                gold_parameter=str(first.get("name") or ""),
                ask_deadline=deadline,
                deadline_source=deadline_source,
                open_items=items,
            )
        )
    return out


def expand(
    rows: Iterable[Mapping[str, Any]],
    *,
    frame_cap: int,
    image_root: Path,
    limit: int = 0,
) -> list[StepItem]:
    """Every scorable step of every gold row, in a fixed order.

    The order is fixed so two runs of two models line up item for item and can be
    compared without a join that could silently drop rows.

    ``limit`` (0 = all) keeps the first steps in that order. It is applied before any
    frame is looked up, so a small run needs only the frames of the steps it keeps.
    """
    rows = list(rows)
    steps_of = {
        (str(row.get("source") or ""), str(row.get("sample_id") or "")): list(
            row.get("steps") or ()
        )
        for row in rows
    }
    items = [
        item
        for row in rows
        for item in expand_row(row, frame_cap=frame_cap, image_root=None)
    ]
    items.sort(key=lambda i: (i.source, i.sample_id, i.step_index))
    if limit:
        items = items[:limit]
    return [
        replace(
            item,
            frames=resolve_frames(
                steps_of[(item.source, item.sample_id)],
                item.step_index,
                frame_cap,
                image_root,
            ),
        )
        for item in items
    ]
