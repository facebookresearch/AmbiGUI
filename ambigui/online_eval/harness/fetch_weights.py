#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""fetch_weights.py <hf repo id> <revision> <dest_dir> -- download one checkpoint from Hugging Face.

The revision must be a commit hash, so the weights are the same bytes on every
machine. A gated repository needs HF_TOKEN in the environment.
"""
import os
import re
import sys
import time

if len(sys.argv) != 4 or sys.argv[1] in ("-h", "--help"):
    print("usage: python fetch_weights.py <hf repo id> <40-character revision> <dest_dir>")
    raise SystemExit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)

from huggingface_hub import snapshot_download  # noqa: E402

repo, revision, dest = sys.argv[1], sys.argv[2], sys.argv[3]
if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
    raise SystemExit(f"[weights] revision must be a 40-character commit hash, got {revision!r}")
# Checkpoint files only, the same suffixes the paper runs fetched: everything vLLM loads
# is one of these. Subdirectories such as examples, figures or original/ are skipped.
_KEEP = (".safetensors", ".json", ".jinja", ".model", ".txt", ".py", ".tiktoken")
os.makedirs(dest, exist_ok=True)
t0 = time.time()
path = snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=dest,
    allow_patterns=[f"*{suffix}" for suffix in _KEEP],
    ignore_patterns=["*/*"],
    token=os.environ.get("HF_TOKEN") or None,
)
total = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
print(f"[weights] {repo}@{revision}: {total / 1e9:.1f} GB in {time.time() - t0:.0f}s -> {dest}", flush=True)
