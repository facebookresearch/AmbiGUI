# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Read the JSON object out of a model reply (pure, no network).

A reply can start with a ``<think>...</think>`` reasoning prefix, wrap the JSON in a
```` ```json ```` code fence, or add prose after it. ``parse_judge_json`` strips all of that
and returns the object, or ``None`` when the reply holds no valid JSON object. The answerer
(``scripts/answerer.py``) reads its replies with it.
"""

from __future__ import annotations

import json
import re


def _strip_reasoning_and_fences(raw: str) -> str:
    """Strip a ``<think>...</think>`` prefix and a ```` ```json ```` code fence, then
    return the substring most likely to be the final JSON object."""
    raw = raw.strip()

    # Strip <think>...</think> reasoning prefix (closed or truncated/unclosed).
    if "<think>" in raw:
        end = raw.find("</think>")
        if end >= 0:
            raw = raw[end + len("</think>") :].strip()
        else:
            after_think = raw.find("<think>") + len("<think>")
            first_brace = raw.find("{", after_think)
            if first_brace >= 0:
                raw = raw[first_brace:]

    # Prefer a fenced ```json { ... } ``` block.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        return fence.group(1)

    # Prefer the LAST {...} block (a reasoning trace may contain an example object first).
    matches = list(re.finditer(r"\{.*?\}(?=[^}]*$)", raw, re.DOTALL))
    if matches:
        return matches[-1].group(0)
    greedy = re.search(r"\{.*\}", raw, re.DOTALL)
    return greedy.group(0) if greedy else raw


def parse_judge_json(raw: str) -> dict | None:
    """Parse a judge (tag/match) JSON object; None on malformed input."""
    payload = _strip_reasoning_and_fences(raw)
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
