# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Swap the ask paragraph of the policy system prompt for a recorded variant.

ASK_PROMPT_VARIANT=pN replaces the repo's own ask paragraph (data/prompts/ask_p0.txt,
verbatim) in training.cascp_common.CASCP_SYSTEM with data/prompts/ask_pN.txt. Called before
the scorer imports it, so the scorer's `from training.cascp_common import CASCP_SYSTEM`
binds the variant. Unset: nothing changes.
"""

import hashlib
import os
import pathlib
import sys

_DIR = pathlib.Path(__file__).resolve().parent / "data" / "prompts"


def apply() -> str | None:
    variant = os.environ.get("ASK_PROMPT_VARIANT") or ""
    if not variant:
        return None
    import training.cascp_common as cc  # noqa: PLC0415

    base = (_DIR / "ask_p0.txt").read_text().strip()
    new = (_DIR / f"ask_{variant}.txt").read_text().strip()
    if cc.CASCP_SYSTEM.count(base) != 1:
        raise SystemExit("prompt_variant: the baseline ask paragraph is not in CASCP_SYSTEM exactly once")
    cc.CASCP_SYSTEM = cc.CASCP_SYSTEM.replace(base, new)
    digest = hashlib.sha256(cc.CASCP_SYSTEM.encode()).hexdigest()[:12]
    # stderr: preflight_roles.py writes its JSON verdict to stdout.
    print(f"[prompt] ask variant {variant}: system prompt sha256 {digest}", file=sys.stderr, flush=True)
    return variant
