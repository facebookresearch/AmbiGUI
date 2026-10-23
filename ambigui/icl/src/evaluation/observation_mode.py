# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""What the policy is shown each turn, as a named, recorded choice.

Every released run uses one observation mode, ``som_state``: a screenshot with each
element's handle (its BrowserGym ``bid``) drawn on it ("set of marks"), and, as text, only
what changed on the page since the last turn. It is the only mode, and is recorded with
the run.

**The judge's frames are never marked.** Only the policy's copy of a frame is marked and
scaled down. The frames written to disk -- the ones the task-success judge reads and a
person reviews -- stay as the browser rendered them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True)
class ObservationMode:
    """One way of showing the policy a page.

    Attributes:
        name: how the mode is named in the records.
        visible_only: prune the tree to elements actually on screen.
        bbox: put each element's coordinates in the tree text.
    """

    name: str
    visible_only: bool
    bbox: bool


#: The mode every run uses: marked frame, and only what changed on the page since the
#: last turn.
SOM_STATE: Final[ObservationMode] = ObservationMode("som_state", True, False)

#: Longest edge of the frame the policy is shown.
#:
#: At the browser's own 1280x720 a frame costs 880 tokens of the policy's context every
#: turn; at 768 it costs 336. The frames written to disk are not resized -- the
#: task-success judge and anyone reviewing a run read those, and they should see what the
#: browser drew.
FRAME_MAX_DIM: Final[int] = 768

def downscale(image: Any, max_dim: int) -> Any:
    """``image`` with its longest edge brought down to ``max_dim``.

    Returned unchanged when it already fits, so a small frame is not resampled for
    nothing. Only the policy's copy is reduced; the frame on disk keeps the
    resolution the browser rendered.
    """
    if image is None or max(image.size) <= max_dim:
        return image
    shrunk = image.copy()
    shrunk.thumbnail((max_dim, max_dim))
    return shrunk


def render_tree(axtree: Any, properties: dict[str, Any], mode: ObservationMode) -> str:
    """The accessibility tree as ``mode`` shows it to the policy.

    ``properties`` is BrowserGym's ``extra_element_properties``: per bid, whether
    the element is on screen, whether it is clickable, and where it sits. It is
    required for pruning.
    """
    from browsergym.utils.obs import flatten_axtree_to_str  # noqa: PLC0415

    rendered: str = flatten_axtree_to_str(
        axtree,
        extra_properties=properties or None,
        filter_visible_only=mode.visible_only,
        with_bounding_box_coords=mode.bbox,
        with_clickable=mode.bbox,
    )
    return rendered


def render_frame(screenshot: Any, properties: dict[str, Any]) -> Any:
    """The frame with each element's bid drawn on it, or None when there is no frame.

    Marking is best-effort: an element whose geometry the browser could not report
    is left unmarked rather than failing the turn, and a frame that cannot be
    marked at all is sent unmarked. A missing annotation costs the policy a hint;
    a raised exception costs the rollout.
    """
    if screenshot is None:
        return None
    if not properties:
        return screenshot
    from browsergym.utils.obs import overlay_som  # noqa: PLC0415

    try:
        return overlay_som(screenshot, properties)
    except Exception:  # noqa: BLE001 - an unmarked frame beats a lost turn
        return screenshot
