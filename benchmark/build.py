#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rebuild the AmbiGUI gold files from pointers + our annotations + your copy of the sources.

The public benchmark ships no third-party content. Every value that came from a source
dataset (instruction text, dialogue turns, action strings, element names, coordinates,
site names) was replaced in ``benchmark/annotations/`` by a *template* that points at the
source record. This script reads your local copy of each source, recomputes the source
values ("atoms") named by ``benchmark/manifest/pointers.jsonl``, checks each against the
SHA-256 recorded there (so a different source version is caught, not silently used), and
fills the templates back in.

    python benchmark/build.py \
        --mind2web-dir  <snapshot of the Mind2Web Hugging Face dataset osunlp/Multimodal-Mind2Web: holds data/test_*.parquet> \
        --metagui-dir   <META-GUI test.tar extracted: holds test/dialog_*/> \
        --osworld-dir   <clone of xlang-ai/OSWorld at the pinned commit> \
        --out           <output dir>

Writes ``gold_eval_set.jsonl``, ``gold_benchmark_with_targets.jsonl``,
``open_items.windows.jsonl`` and ``ambigui_online_benchmark.jsonl``. A source you do not pass is skipped: its rows are left
out and reported (``--allow-missing-source``), otherwise the run stops. Every row whose source
differs from the release (for example an OSWorld task config with another hash) is checked and
listed, not only the first one.

Each benchmark row has two lists of open items, as in the gold file: ``open_items`` (the
items the simulated user answers from and the ask judge grades against) and
``items_from_decision_points`` (read off the annotators' decision records). Use
``open_items`` only; never merge the two lists.

Only labels from approved producers ship (``manifest/label_provenance.yaml``). Every label
that is not approved is listed in ``manifest/removed_labels.json``: the annotation files
already carry ``null`` for it, and this script applies the same list again to every row it
writes, so a removed label can never come back by mistake. Fields that only record a model
panel's output are not written at all.

Template format (the only thing a reader of ``annotations/`` needs to know):

* a plain JSON value is ours and is copied as is;
* ``{"$src": "<atom>"}`` is a number taken from the source;
* ``{"$tpl": "<text>"}`` is a string in which every ``{{atom|transform...[a:b]}}`` is
  replaced by that atom, passed through the transforms, then sliced.  Transforms:
  ``nbsp`` (U+00A0 -> space), ``clean`` (collapse whitespace), ``esc`` (``"`` -> ``\\"``),
  ``lower``, ``upper``.  ``{{$pool_root}}`` and ``{{$package_root}}`` are build-time
  variables (where *you* keep the screenshots), not source atoms.

Screenshots are never shipped or rebuilt here; obtain them from each source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest"
ANNOTATIONS = HERE / "annotations"

OUTPUT_FILES = (
    "gold_eval_set.jsonl",
    "gold_benchmark_with_targets.jsonl",
    "open_items.windows.jsonl",
    "ambigui_online_benchmark.jsonl",
)
REMOVED_LABELS = "removed_labels.json"

#: The ``source`` value of a row. Each names the public dataset ``build.py`` reads:
#: mind2web -> Hugging Face osunlp/Multimodal-Mind2Web (test splits), meta-gui -> Hugging
#: Face X-LANCE/META-GUI (test.tar), osworld -> GitHub xlang-ai/OSWorld (task files).
SOURCE_M2W = "mind2web"
SOURCE_MG = "meta-gui"
SOURCE_OSW = "osworld"

# --------------------------------------------------------------------------- templates

_SLOT = re.compile(r"\{\{([^{}\[\]|]+)((?:\|[a-z]+)*)(?:\[(\d+):(\d+)\])?\}\}")


def transform(value: str, names: Sequence[str]) -> str:
    """Apply the named transforms, in order, to a source string."""
    for name in names:
        if name == "nbsp":
            value = value.replace("\u00a0", " ")
        elif name == "clean":
            value = " ".join(value.split())
        elif name == "esc":
            value = value.replace('"', '\\"')
        elif name == "lower":
            value = value.lower()
        elif name == "upper":
            value = value.upper()
        else:
            raise ValueError(f"unknown transform {name!r}")
    return value


def render(template: str, atoms: Mapping[str, Any], variables: Mapping[str, str]) -> str:
    """Fill every ``{{...}}`` slot in ``template``."""

    def sub(m: re.Match[str]) -> str:
        ref, tr, a, b = m.group(1), m.group(2), m.group(3), m.group(4)
        if ref.startswith("$"):
            return variables[ref]
        value = transform(str(atoms[ref]), [t for t in tr.split("|") if t])
        return value[int(a) : int(b)] if a is not None else value

    return _SLOT.sub(sub, template)


def resolve(node: Any, atoms: Mapping[str, Any], variables: Mapping[str, str]) -> Any:
    """Rebuild one annotation value (recursively) from its templates."""
    if isinstance(node, dict):
        if set(node) == {"$tpl"}:
            return render(node["$tpl"], atoms, variables)
        if set(node) == {"$src"}:
            return atoms[node["$src"]]
        return {k: resolve(v, atoms, variables) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve(v, atoms, variables) for v in node]
    return node


def referenced_atoms(node: Any, out: set[str] | None = None) -> set[str]:
    """Every source atom a value refers to."""
    out = set() if out is None else out
    if isinstance(node, dict):
        if set(node) == {"$tpl"}:
            out.update(m.group(1) for m in _SLOT.finditer(node["$tpl"]) if not m.group(1).startswith("$"))
        elif set(node) == {"$src"}:
            out.add(node["$src"])
        else:
            for v in node.values():
                referenced_atoms(v, out)
    elif isinstance(node, list):
        for v in node:
            referenced_atoms(v, out)
    return out


def atom_hash(value: Any) -> str:
    """First 128 bits (32 hex) of the SHA-256 of an atom's text (strings as is, numbers as JSON)."""
    text = value if isinstance(value, str) else json.dumps(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


# ------------------------------------------------------------------ removed labels
# ``manifest/removed_labels.json`` lists every label that may not ship. A rule names a
# field path (``a.b``; ``x[]`` walks every element of list ``x``), the rows it applies to,
# and an action:
#
# * ``null``           set the field to ``null`` (the key stays, so a reader can see it
#                      was removed);
# * ``drop_key``       remove the key (fields that only record a model panel's output);
# * ``set``            replace the value with ``value`` (an edit to our own text);
# * ``drop_elements``  remove the list elements that match ``element_where`` or
#                      ``drop_values``; with ``empty_to_null`` a list that loses every
#                      element becomes ``null`` (removed), not ``[]`` (nothing there).
#                      ``element_where: {}`` matches every element.
#
# Rows: ``"all"``, ``{"source": [...]}``, ``{"sample_id": [...]}`` or
# ``{"field_in": {path: [values]}}``. Rows are selected on the row as it was before any
# rule ran. ``element_where`` on a ``null`` rule limits it to the elements of the last
# ``[]`` list whose fields equal the given values.


def load_removals(manifest: Path = MANIFEST) -> dict[str, Any]:
    return json.loads((manifest / REMOVED_LABELS).read_text(encoding="utf-8"))


def get_path(node: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def row_selected(row: Mapping[str, Any], rows: Any) -> bool:
    if rows == "all":
        return True
    if "source" in rows:
        return row.get("source") in rows["source"]
    if "sample_id" in rows:
        return row.get("sample_id") in rows["sample_id"]
    if "field_in" in rows:
        return all(get_path(row, p) in vals for p, vals in rows["field_in"].items())
    raise ValueError(f"unknown row selector {rows!r}")


def _element_matches(elem: Any, rule: Mapping[str, Any]) -> bool:
    where = rule.get("element_where")
    if where is not None:
        return isinstance(elem, dict) and all(get_path(elem, k) == v for k, v in where.items())
    return elem in rule.get("drop_values", [])


def _apply(node: Any, parts: Sequence[str], rule: Mapping[str, Any], count: list[int]) -> None:
    if not isinstance(node, dict):
        return
    head, rest = parts[0], parts[1:]
    if head.endswith("[]"):
        key = head[:-2]
        items = node.get(key)
        if not isinstance(items, list):
            return
        if not rest:
            if rule["action"] != "drop_elements":
                raise ValueError(f"{rule['path']}: a path ending in [] needs drop_elements")
            kept = [e for e in items if not _element_matches(e, rule)]
            count[0] += len(items) - len(kept)
            node[key] = None if (items and not kept and rule.get("empty_to_null")) else kept
            return
        last_list = not any(p.endswith("[]") for p in rest)
        for elem in items:
            if last_list and "element_where" in rule and not _element_matches(elem, rule):
                continue
            _apply(elem, rest, rule, count)
        return
    if rest:
        _apply(node.get(head), rest, rule, count)
        return
    if head not in node:
        return
    if rule["action"] == "null":
        if node[head] is not None:
            count[0] += 1
        node[head] = None
    elif rule["action"] == "drop_key":
        count[0] += 1
        del node[head]
    elif rule["action"] == "set":
        if node[head] != rule["value"]:
            count[0] += 1
        node[head] = rule["value"]
    else:
        raise ValueError(f"{rule['path']}: drop_elements needs a path ending in []")


def apply_removals(name: str, row: Mapping[str, Any], spec: Mapping[str, Any], counts: dict[str, int] | None = None) -> dict[str, Any]:
    """``row`` with every rule for output file ``name`` applied (a deep copy)."""
    out = json.loads(json.dumps(row))
    for rule in spec.get("files", {}).get(name, {}).get("rules", []):
        if not row_selected(row, rule["rows"]):
            continue
        n = [0]
        _apply(out, rule["path"].split("."), rule, n)
        if counts is not None:
            counts[rule["id"]] = counts.get(rule["id"], 0) + n[0]
    return out


# --------------------------------------------------------------- shared source helpers
# Ports of the deterministic helpers the internal build used to read element names.


_BOUNDS = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def _clean(value: Any) -> str:
    return " ".join(("" if value is None else str(value)).split())


def view_hierarchy_label(root: ET.Element, point: Sequence[int]) -> str:
    """Name of the Android view at ``point``: smallest containing node's text or
    content-desc, else a labelled ancestor's, else the node's resource-id tail."""
    x, y = point[0], point[1]
    hits: list[tuple[int, int, ET.Element, list[ET.Element]]] = []
    stack: list[tuple[ET.Element, list[ET.Element]]] = [(root, [])]
    while stack:
        node, ancestors = stack.pop()
        for child in node:
            if child.tag != "node":
                continue
            found = _BOUNDS.match(child.get("bounds") or "")
            if found:
                x1, y1, x2, y2 = (int(v) for v in found.groups())
                if x1 <= x <= x2 and y1 <= y <= y2:
                    hits.append(((x2 - x1) * (y2 - y1), -len(ancestors), child, ancestors + [node]))
            stack.append((child, ancestors + [node]))
    if not hits:
        return ""
    _, _, node, ancestors = min(hits, key=lambda h: (h[0], h[1]))
    for candidate in [node, *reversed(ancestors)]:
        for key in ("text", "content-desc"):
            text = _clean(candidate.get(key))
            if text:
                return text
    return _clean(node.get("resource-id")).rsplit("/", 1)[-1]


_DOM_KEYS = ("aria_label", "aria-label", "alt", "placeholder", "title", "label", "value", "name")


def dom_label(candidates: Iterable[Any]) -> str:
    """First labelled attribute of the FIRST positive Mind2Web DOM candidate."""
    for candidate in candidates:
        try:
            cand = json.loads(candidate) if isinstance(candidate, str) else candidate
            attrs = cand.get("attributes")
            attrs = json.loads(attrs) if isinstance(attrs, str) else attrs
        except (ValueError, AttributeError):
            return ""
        if not isinstance(attrs, dict):
            return ""
        for key in _DOM_KEYS:
            text = _clean(attrs.get(key))
            if text:
                return text
        return ""
    return ""


def _bbox_centre(candidates: Sequence[Any]) -> tuple[int, int] | None:
    if not candidates:
        return None
    cand = json.loads(candidates[0]) if isinstance(candidates[0], str) else candidates[0]
    attrs = cand.get("attributes")
    attrs = json.loads(attrs) if isinstance(attrs, str) else attrs
    rect = (attrs or {}).get("bounding_box_rect")
    if not rect:
        return None
    x, y, w, h = (float(v) for v in rect.split(","))
    return round(x + w / 2), round(y + h / 2)


# ---------------------------------------------------------------------- source loaders


class SourceDiffers(ValueError):
    """The source record exists but is not the version the release was made from."""


def load_mind2web(root: Path, annotation_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """Mind2Web test-split steps (Hugging Face osunlp/Multimodal-Mind2Web) for the given tasks: ``{annotation_id: {action_uid: row}}``."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    cols = [
        "action_uid", "operation", "pos_candidates", "website", "domain", "subdomain",
        "annotation_id", "confirmed_task", "target_action_index", "target_action_reprs",
    ]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    shards = sorted((root / "data").glob("test_*.parquet"))
    if not shards:
        raise SystemExit(f"no data/test_*.parquet under {root}")
    wanted = pa.array(sorted(annotation_ids))
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=cols)
            table = table.filter(pc.is_in(table["annotation_id"], value_set=wanted))
            for row in table.to_pylist():
                out.setdefault(row["annotation_id"], {})[row["action_uid"]] = row
    return out


def mind2web_atoms(pointer: Mapping[str, Any], steps_by_uid: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    atoms: dict[str, Any] = {}
    first = next(iter(steps_by_uid.values()))
    for key, field in (("task", "confirmed_task"), ("website", "website"), ("domain", "domain"), ("subdomain", "subdomain")):
        atoms[key] = first[field]
    for step in pointer["steps"]:
        k, src = step["step"], steps_by_uid[step["action_uid"]]
        if str(src["target_action_index"]) != str(step["target_action_index"]):
            raise ValueError(f"{pointer['sample_id']} step {k}: target_action_index moved")
        op = json.loads(src["operation"])
        atoms[f"s{k}.tar"] = src["target_action_reprs"]
        atoms[f"s{k}.operation"] = src["operation"]
        atoms[f"s{k}.op"] = str(op.get("op") or op.get("original_op") or "")
        atoms[f"s{k}.value"] = str(op.get("value", "") or "")
        atoms[f"s{k}.dom"] = dom_label(src["pos_candidates"] or [])
        centre = _bbox_centre(src["pos_candidates"] or [])
        if centre is not None:
            atoms[f"s{k}.cx"], atoms[f"s{k}.cy"] = centre
    return atoms


def metagui_index(root: Path) -> dict[str, Path]:
    """``{dialog_id: dialog dir}`` over an extracted META-GUI tree."""
    out: dict[str, Path] = {}
    for f in sorted(root.glob("**/dialog.json")):
        d = f.parent
        idf = d / "dialog_id.txt"
        out[idf.read_text().strip() if idf.is_file() else d.name] = d
    return out


def metagui_atoms(pointer: Mapping[str, Any], dialog_dir: Path) -> dict[str, Any]:
    atoms: dict[str, Any] = {}
    cat = dialog_dir / "category.txt"
    atoms["category"] = cat.read_text().strip() if cat.is_file() else "misc"
    dialog = json.loads((dialog_dir / "dialog.json").read_text())
    for i, turn in enumerate(dialog):
        atoms[f"d{i}.text"] = str(turn.get("text", ""))
        atoms[f"d{i}.speaker"] = "user" if turn.get("isUser") else "system"
    actions: dict[str, dict[str, str]] = {}
    for step in pointer["steps"]:
        k, turn_dir, key = step["step"], step["turn_dir"], step["action_key"]
        if turn_dir not in actions:
            actions[turn_dir] = json.loads((dialog_dir / turn_dir / "actions.json").read_text())
        act = actions[turn_dir][key]
        atoms[f"s{k}.act"] = act
        parts = act.split("|")
        params = [p for p in parts[2:] if p and p != "NULL"]
        xy = params[0].split(",") if parts[1:2] == ["click"] and params else []
        if len(xy) == 2 and all(v.strip().lstrip("-").isdigit() for v in xy):
            atoms[f"s{k}.x"], atoms[f"s{k}.y"] = int(xy[0]), int(xy[1])
            xml = dialog_dir / turn_dir / f"{key}.xml"
            try:
                atoms[f"s{k}.xml"] = view_hierarchy_label(ET.parse(xml).getroot(), (int(xy[0]), int(xy[1])))
            except (OSError, ET.ParseError):
                atoms[f"s{k}.xml"] = ""
    return atoms


def osworld_atoms(pointer: Mapping[str, Any], osworld_root: Path) -> dict[str, Any]:
    """Atoms of one OSWorld task, read from ``evaluation_examples/examples/<domain>/<id>.json``
    at the pinned commit: the instruction, and a check that the task config is the one the
    release was made against.

    OSWorld rows ship no recorded trajectory: no action steps, no step count and no click
    coordinates. They use OSWorld's own environment and grounding.
    """
    native = pointer["native"]
    task = json.loads((osworld_root / native["osworld_path"]).read_text())
    if task.get("id") != native["osworld_task_id"]:
        raise SourceDiffers(f"OSWorld task id {task.get('id')!r} != {native['osworld_task_id']!r} ({native['osworld_path']})")
    got = atom_hash(json.dumps(task.get("config"), sort_keys=True, ensure_ascii=True))
    if got != native["osworld_config_sha256"]:
        raise SourceDiffers(
            f"OSWorld task {native['osworld_task_id']}: config hash {got} != {native['osworld_config_sha256']}"
            f" ({native['osworld_path']})"
        )
    return {"domain": native["domain"], "instruction": task["instruction"]}


# ------------------------------------------------------------------------------ build


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_annotation(name: str, root: Path = ANNOTATIONS) -> list[dict[str, Any]]:
    """An annotation file, re-joined from its ``.part-NN.jsonl`` shards in order."""
    stem = name[: -len(".jsonl")]
    parts = sorted(root.glob(f"{stem}.ours.part-*.jsonl"))
    if not parts:
        raise SystemExit(f"no annotation shards for {name} under {root}")
    rows: list[dict[str, Any]] = []
    for p in parts:
        rows.extend(read_jsonl(p))
    return rows


def compute_atoms(
    pointers: Sequence[Mapping[str, Any]],
    mind2web_dir: Path | None,
    metagui_dir: Path | None,
    osworld_dir: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Atoms for every row whose source is available, and why the others are missing."""
    atoms: dict[str, dict[str, Any]] = {}
    missing: dict[str, str] = {}
    m2w = {}
    if mind2web_dir is not None:
        ids = {p["native"]["annotation_id"] for p in pointers if p["source"] == SOURCE_M2W}
        m2w = load_mind2web(mind2web_dir, ids)
    mg = metagui_index(metagui_dir) if metagui_dir is not None else {}
    for p in pointers:
        sid, src = p["sample_id"], p["source"]
        try:
            if src == SOURCE_M2W:
                if mind2web_dir is None:
                    missing[sid] = "no --mind2web-dir"
                    continue
                atoms[sid] = mind2web_atoms(p, m2w[p["native"]["annotation_id"]])
            elif src == SOURCE_MG:
                if metagui_dir is None:
                    missing[sid] = "no --metagui-dir"
                    continue
                atoms[sid] = metagui_atoms(p, mg[p["native"]["dialog_id"]])
            elif src == SOURCE_OSW:
                if osworld_dir is None:
                    missing[sid] = "no --osworld-dir"
                    continue
                atoms[sid] = osworld_atoms(p, osworld_dir)
            else:
                missing[sid] = f"unknown source {src!r}"
        except SourceDiffers as exc:
            missing[sid] = f"source differs from the release: {exc}"
        except (KeyError, IndexError, OSError, ValueError) as exc:
            missing[sid] = f"source record not found or unreadable: {exc!r}"
    return atoms, missing


def unhashed_atoms(
    pointers: Sequence[Mapping[str, Any]], annotations: Path = ANNOTATIONS
) -> dict[str, set[str]]:
    """``{sample_id: atoms}`` for every template atom that has no recorded hash.

    ``verify`` checks only the atoms listed in ``atom_sha256``; this makes sure every atom a
    template refers to is listed there, so no source value is filled in unchecked.
    """
    hashed = {p["sample_id"]: set(p["atom_sha256"]) for p in pointers}
    out: dict[str, set[str]] = {}
    for name in OUTPUT_FILES:
        for row in read_annotation(name, annotations):
            sid = row["sample_id"]
            gap = referenced_atoms(row) - hashed.get(sid, set())
            if gap:
                out.setdefault(sid, set()).update(gap)
    return out


def verify(pointer: Mapping[str, Any], atoms: Mapping[str, Any]) -> list[str]:
    """Atoms whose hash differs from the one recorded at release time."""
    bad = []
    for ref, digest in pointer["atom_sha256"].items():
        if ref not in atoms:
            bad.append(f"{ref}: missing")
        elif atom_hash(atoms[ref]) != digest:
            bad.append(f"{ref}: hash differs")
    return bad


def build(
    mind2web_dir: Path | None,
    metagui_dir: Path | None,
    osworld_dir: Path | None,
    variables: Mapping[str, str] | None = None,
    allow_missing: bool = False,
    manifest: Path = MANIFEST,
    annotations: Path = ANNOTATIONS,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], dict[str, int]]:
    """Rebuilt rows per output file, ``{sample_id: reason}`` for rows left out, and how
    many values each rule of ``manifest/removed_labels.json`` removed while building.

    The annotation files already carry the removals, so on a clean release every count
    is 0; a non-zero count means an annotation file held a label that may not ship.
    """
    variables = {"$pool_root": "", "$package_root": "", **(variables or {})}
    pointers = read_jsonl(manifest / "pointers.jsonl")
    by_id = {p["sample_id"]: p for p in pointers}
    unhashed = unhashed_atoms(pointers, annotations)
    if unhashed:
        raise SystemExit(
            f"{len(unhashed)} row(s) use a source value with no recorded hash in pointers.jsonl:\n"
            + "\n".join(f"  {sid}: {', '.join(sorted(refs))}" for sid, refs in sorted(unhashed.items()))
        )
    atoms, missing = compute_atoms(pointers, mind2web_dir, metagui_dir, osworld_dir)
    for sid, row_atoms in atoms.items():
        bad = verify(by_id[sid], row_atoms)
        if bad:
            missing[sid] = "source differs from the release: " + "; ".join(bad[:5])
    for sid in missing:
        atoms.pop(sid, None)
    if missing and not allow_missing:
        raise SystemExit(
            f"{len(missing)} row(s) cannot be rebuilt:\n"
            + "\n".join(f"  {k} ({by_id[k]['source'] if k in by_id else '?'}): {v}" for k, v in sorted(missing.items()))
            + "\nCheck that each source is at the version pinned in benchmark/manifest/sources.yaml, or pass"
            " --allow-missing-source to write the other rows and list these."
        )
    spec = load_removals(manifest)
    removed: dict[str, int] = {}
    out: dict[str, list[dict[str, Any]]] = {}
    for name in OUTPUT_FILES:
        rows = []
        for row in read_annotation(name, annotations):
            sid = row["sample_id"]
            if sid in missing:
                continue
            rows.append(apply_removals(name, resolve(row, atoms.get(sid, {}), variables), spec, removed))
        out[name] = rows
    return out, missing, removed


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mind2web-dir", type=Path,
        help="snapshot of the Mind2Web Hugging Face dataset: the folder that holds data/test_*.parquet",
    )
    ap.add_argument("--metagui-dir", type=Path, help="META-GUI test.tar extracted: the folder that holds test/dialog_*/")
    ap.add_argument("--osworld-dir", type=Path, help="checkout of xlang-ai/OSWorld at the pinned commit")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool-root", default="", help="prefix for steps[].image_path in gold_benchmark_with_targets")
    ap.add_argument("--package-root", default="", help="prefix for gold.steps_json[].img")
    ap.add_argument(
        "--allow-missing-source", action="store_true",
        help="write the rows that can be rebuilt and list the others (exit code 2) instead of stopping",
    )
    a = ap.parse_args(argv)
    out, missing, removed = build(
        a.mind2web_dir, a.metagui_dir, a.osworld_dir,
        {"$pool_root": a.pool_root, "$package_root": a.package_root},
        allow_missing=a.allow_missing_source,
    )
    a.out.mkdir(parents=True, exist_ok=True)
    for name, rows in out.items():
        with (a.out / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(f"wrote {len(rows)} rows -> {a.out / name}")
    for sid, why in missing.items():
        print(f"skipped {sid}: {why}", file=sys.stderr)
    leaked = {k: v for k, v in removed.items() if v}
    if leaked:
        print(f"ERROR: annotation files held removed labels (written as null/dropped): {leaked}", file=sys.stderr)
        return 1
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
