# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Small helpers of the demonstration builder: hashing, JSON files, task fields."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_key(text: str) -> str:
    return hashlib.sha256(" ".join(str(text).casefold().split()).encode()).hexdigest()


def task_id(row: dict) -> str:
    key = str(row.get("task_id") or row.get("sample_id") or "")
    if not key:
        raise ValueError("task has no ID")
    return key


def task_family(key: str) -> str:
    # High-stakes/repeated variants of the same source are one leakage group.
    return str(key).split("__", 1)[0]


def task_text(row: dict) -> str:
    return str(row.get("canonical_instruction") or row.get("instruction") or row.get("task") or "")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)
