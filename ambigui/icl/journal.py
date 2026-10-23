# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Lossless journal of the judge calls made while the demonstrations are prepared."""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .common import canonical_json, digest


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


class Journal:
    def __init__(self, directory: Path, *, context: dict, grader_model: str) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "img").mkdir(exist_ok=True)
        self.context = context
        self.grader_model = grader_model
        self.calls: list[dict] = []
        self.lock = threading.Lock()
        self.next_call = 0
        if (directory / "calls.jsonl").exists():
            for line in (directory / "calls.jsonl").read_text().splitlines():
                try:
                    old = json.loads(line)
                    self.next_call = max(self.next_call, int(old.get("call_id", -1)) + 1)
                except (ValueError, TypeError):
                    # A killed write's partial line is retained in the file, not
                    # promoted into a completed request/response event.
                    continue
        self.installed = False

    def strip_images(self, value: Any) -> Any:
        value = plain(value)
        if isinstance(value, list):
            return [self.strip_images(v) for v in value]
        if not isinstance(value, dict):
            return value
        if value.get("type") == "image_url":
            iu = value["image_url"]
            url = iu.get("url") if isinstance(iu, dict) else iu
            if not str(url).startswith("data:"):
                return value
            head, encoded = url.split(",", 1)
            raw = base64.b64decode(encoded, validate=True)
            sha = hashlib.sha256(raw).hexdigest()
            suffix = ".png" if "png" in head else ".jpg"
            filename = f"img/{sha}{suffix}"
            path = self.directory / filename
            if not path.exists():
                path.write_bytes(raw)
            return {"type": "image_file", "file": filename, "sha256": sha,
                    "bytes": len(raw), "mime": head.split(";", 1)[0][5:]}
        return {k: self.strip_images(v) for k, v in value.items()}

    def append(self, event: dict) -> None:
        with self.lock:
            with (self.directory / "calls.jsonl").open("a") as f:
                f.write(canonical_json(event) + "\n")

    def install(self) -> None:
        """Intercept newly constructed OpenAI clients. This changes recording only."""
        if self.installed:
            raise RuntimeError("journal already installed")
        import openai
        self.real_openai = openai.OpenAI
        journal = self

        def factory(*args, **kwargs):
            # Hosted endpoints: the key comes from the environment (evaluation.endpoint_auth).
            # The caller passes the placeholder "EMPTY", which a local vLLM accepts.
            from evaluation.endpoint_auth import NO_KEY, api_key_for
            if kwargs.get("api_key") in (None, NO_KEY):
                kwargs["api_key"] = api_key_for(kwargs.get("base_url"))
            real = journal.real_openai(*args, **kwargs)
            return JournalClient(real, journal, str(kwargs.get("base_url", "")))

        openai.OpenAI = factory
        self.installed = True

    def restore(self) -> None:
        if self.installed:
            import openai
            openai.OpenAI = self.real_openai
            self.installed = False


class JournalClient:
    def __init__(self, client: Any, journal: Journal, endpoint: str) -> None:
        self.client, self.journal, self.endpoint = client, journal, endpoint
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def __getattr__(self, key: str) -> Any:
        return getattr(self.client, key)

    def create(self, **kwargs: Any) -> Any:
        journal = self.journal
        model = kwargs.get("model")
        if model != journal.grader_model:
            raise ValueError(f"unfrozen model identity requested: {model!r}")
        with journal.lock:
            call_id = journal.next_call
            journal.next_call += 1
        request = journal.strip_images({k: v for k, v in kwargs.items()
                                       if k not in ("extra_headers", "api_key")})
        event = {**journal.context, "call_id": call_id, "role": "grader", "endpoint": self.endpoint,
                 "caller": [{"file": Path(s.filename).name, "function": s.name}
                            for s in traceback.extract_stack(limit=6)[:-1]],
                 "request": request, "request_sha256": digest(request)}
        journal.append({**event, "status": "requested", "time_unix": time.time()})
        start = time.monotonic()
        try:
            response = self.client.chat.completions.create(**kwargs)
            dumped = plain(response)
            if dumped.get("model") != model:
                raise ValueError(f"response model {dumped.get('model')!r} differs from frozen request {model!r}")
            event.update({"status": "completed", "response": dumped,
                          "wall_s": time.monotonic() - start})
        except Exception as exc:
            event.update({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                          "wall_s": time.monotonic() - start})
            journal.append(event)
            journal.calls.append(event)
            raise
        journal.append(event)
        journal.calls.append(event)
        return response
