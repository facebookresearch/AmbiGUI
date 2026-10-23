# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Choosing the judge that decides whether a rollout completed its task.

One entry point, used by both the environment and the evaluation loop. There
used to be two, and a run asked for one judge on the command line while the
loop went on calling another -- every task came back a judge exception.

``lookup`` is the only judge. Any other kind is refused.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from typing import Any, cast

from evaluation.action_trace import render_actions
from evaluation.endpoint_auth import api_key_for

#: Longest edge a frame is sent at. Full-size screenshots are several megabytes
#: and the describer reads a dozen of them.
FRAME_WIDTH = 1000


class JudgeMisconfigured(ValueError):
    """The run named a judge that cannot score anything.

    Distinct from a judge that failed: an outage is transient and a rollout that
    cannot be scored is reasonably counted as unsuccessful, but a judge kind that
    does not exist, or one missing a serve it needs, is wrong for every rollout in
    the run. Callers that swallow judge failures must let this one through --
    otherwise a misconfigured run reports a complete, all-zero result.

    A ``ValueError`` so that callers written before it existed still catch it.
    """


def score_rollout_v1(
    kind: str,
    judge_url: str,
    judge_model: str,
    instruction: str,
    shots: list[str],
    actions: list[dict[str, Any]],
    final_url: str = "",
    describer_url: str | None = None,
    describer_model: str | None = None,
) -> tuple[bool, str]:
    """Score one rollout with the chosen judge. Returns ``(success, why)``.

    ``lookup``: a text judge reads the actions and asks a vision model specific
    questions about the screens when the verdict turns on something visual. Roughly
    27/60 on the ground-truth control, with next to no false alarms; two runs of the
    same sixty gave 28/0 and 26/1.
    """
    if kind != "lookup":
        raise JudgeMisconfigured(f"unknown judge kind {kind!r}; use 'lookup'")
    if not describer_url:
        raise JudgeMisconfigured(
            "the 'lookup' judge needs a describer endpoint; without one it "
            "cannot see the screen and degrades to the 4/60 arm"
        )
    return _score_with_lookup(
        judge_url,
        judge_model,
        describer_url,
        describer_model or judge_model,
        instruction,
        shots,
        actions,
        final_url,
    )


def encode_frame(path: str, width: int = FRAME_WIDTH) -> str | None:
    """One screenshot as base64 JPEG, or None if it cannot be read."""
    try:
        from PIL import Image

        # Bound to its own name: `Image.open` yields an ImageFile and `convert`
        # returns an Image, so reassigning the `with` target is a type error under
        # Pillow's current stubs even though it runs.
        with Image.open(path) as opened:
            im = opened.convert("RGB")
            if im.width > width:
                im = im.resize((width, max(1, int(im.height * width / im.width))))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001 - a missing frame is not a crash
        return None


def _score_with_lookup(
    judge_url: str,
    judge_model: str,
    describer_url: str,
    describer_model: str,
    instruction: str,
    shots: list[str],
    actions: list[dict[str, Any]],
    final_url: str,
) -> tuple[bool, str]:
    """The current judge: read the actions, ask about the screen, then decide."""
    from openai import OpenAI

    from evaluation.lookup_judge import judge as lookup_judge

    judge_client = OpenAI(base_url=judge_url, api_key=api_key_for(judge_url))
    describer_client = OpenAI(base_url=describer_url, api_key=api_key_for(describer_url))

    def ask(prompt: str) -> str:
        return (
            judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1200,
            )
            .choices[0]
            .message.content
            or ""
        )

    def look(prompt: str, frames: Sequence[str]) -> str:
        # A multimodal message: one text part and one per readable frame. Typed
        # loosely and cast at the call, because the client's own union of
        # message shapes does not describe an image list cleanly.
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in frames:
            encoded = encode_frame(path)
            if encoded:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    }
                )
        return (
            describer_client.chat.completions.create(
                model=describer_model,
                messages=cast(Any, [{"role": "user", "content": content}]),
                temperature=0.0,
                max_tokens=600,
            )
            .choices[0]
            .message.content
            or ""
        )

    rendered = render_actions(actions)
    if final_url:
        rendered += f"\n(final address: {final_url})"
    return lookup_judge(instruction, rendered, shots, ask, look)



def score_rollout(
    kind: str,
    judge_url: str,
    judge_model: str,
    instruction: str,
    shots: list[str],
    actions: list[dict[str, Any]],
    final_url: str = "",
    describer_url: str | None = None,
    describer_model: str | None = None,
    *,
    judge_version: str = "judge_v1",
    curated_row: dict[str, Any] | None = None,
    request_fn: Any = None,
    encoded_frames: dict[str, str] | None = None,
    original_task_time_evidence: dict[str, Any] | None = None,
) -> tuple[bool | None, str]:
    """Legacy API stays v1, explicitly tagged; v2 NEVER falls back to that route.

    v2 needs the exact VERSION, validated curated intent, a journaling callback and
    the encoded frames (see evaluation/judge_v2_adapter.py). Existing v1
    callers keep their positional signature and original result semantics.
    """
    if judge_version == "judge_v1":
        return score_rollout_v1(kind, judge_url, judge_model, instruction, shots,
                                actions, final_url, describer_url, describer_model)
    from evaluation.score_rollout_v2 import VERSION, score_rollout_v2
    if judge_version != VERSION:
        raise JudgeMisconfigured(f"unknown judge version {judge_version!r}; no v1 fallback")
    if request_fn is None or encoded_frames is None:
        raise JudgeMisconfigured("v2 requires the immutable evidence worker's journal callback and validated encoded frames")
    verdict = score_rollout_v2(kind, judge_url, judge_model, instruction, shots,
        actions, final_url, describer_url, describer_model, curated_row=curated_row,
        request_fn=request_fn, encoded_frames=encoded_frames,
        original_task_time_evidence=original_task_time_evidence)
    return verdict.success, verdict.why
