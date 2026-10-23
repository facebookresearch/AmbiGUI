# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Canonical schema for the five underspecification axes.

How each axis is named and which labels it allows. Labels written in older spellings are
mapped onto one canonical form by ``canonicalize``, and an unknown label raises rather than
passing through.

Spellings reconciled here:
  * ``types`` -- the long form ``"Cross-modal conflict"`` and the short form
    ``"Cross-modal"`` both occur; the short one is canonical and the long one is an alias.
  * ``feasibility`` -- the vocabulary has three classes (Feasible / Capability boundary /
    Environment boundary), even where a file only ever uses ``"Feasible"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Axis:
    """One taxonomy axis: where it lives, its shape, and its allowed labels.

    ``kind`` is one of ``"single"`` (one categorical label), ``"ordinal"`` (one
    label whose position in ``vocab`` is its rank), or ``"multilabel"`` (a set of
    labels). For ``"ordinal"`` axes the order of ``vocab`` IS the ranking
    (index 0 is lowest); for the other kinds order is immaterial.
    """

    gold_field: str
    panel_field: str
    kind: str
    vocab: tuple[str, ...]
    aliases: Mapping[str, str]


AXES: dict[str, Axis] = {
    "ambiguity": Axis(
        "ambiguity_v40", "ambiguity", "single", ("Clear", "Ambiguous"), {}
    ),
    "severity": Axis(
        "severity_v40", "severity", "ordinal", ("minor", "major", "critical"), {}
    ),
    "risk": Axis(
        "risk_v40",
        "risk",
        "multilabel",
        ("Reversible", "Irreversible", "Cascading"),
        {},
    ),
    "feasibility": Axis(
        "feasibility_v40",
        "feasibility",
        "single",
        ("Feasible", "Capability boundary", "Environment boundary"),
        {},
    ),
    "types": Axis(
        "types_v40",
        "types",
        "multilabel",
        ("Referential", "Intent", "Parametric", "Cross-modal"),
        {"Cross-modal conflict": "Cross-modal"},
    ),
}


def canonicalize(axis: str, value: str) -> str:
    """Return the canonical label for ``value`` on ``axis``.

    Applies known aliases (e.g. ``"Cross-modal conflict" -> "Cross-modal"``).
    Raises ``ValueError`` -- naming the offending axis and value -- on an unknown
    axis or an out-of-vocab label, so schema drift surfaces loudly instead of
    silently mismatching.
    """
    return _canonicalize(axis, value, AXES, "axis")


def _canonicalize(axis: str, value: str, axes: dict[str, Axis], noun: str) -> str:
    """Validate ``value`` against one axis table. Shared by both public wrappers.

    ``noun`` names the table in the error text. The two tables disagree about
    ``risk``, so a message that says only "unknown axis" would not tell a reader
    which vocabulary rejected their label.
    """
    if axis not in axes:
        raise ValueError(f"unknown {noun} {axis!r} (known: {sorted(axes)})")
    spec = axes[axis]
    resolved = spec.aliases.get(value, value)
    if resolved not in spec.vocab:
        raise ValueError(
            f"out-of-vocab label {value!r} for {noun} {axis!r} "
            f"(vocab: {sorted(spec.vocab)})"
        )
    return resolved
