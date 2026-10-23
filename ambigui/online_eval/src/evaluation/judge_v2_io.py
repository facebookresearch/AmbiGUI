# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Frame and message IO for judge v2, matching the v1 lookup judge exactly.
No credential lookup, retries, model discovery or endpoint substitution.
"""
from __future__ import annotations
import base64
import hashlib
import io
import os
from pathlib import Path
from typing import Any
import urllib.request
import urllib.error
from evaluation.score_rollout_v2 import canonical

NATIVE_CONTRACT = {"profile": "21e7f03_lookup_trace", "judge_lookup_max_tokens": 1200,
                   "judge_trace_max_tokens": 600, "describer_max_tokens": 600,
                   "max_questions": 3, "frame_selector": "21e7f03_native_select_frames",
                   "frame_encoder": "jpeg_width1000_quality80", "history": "full_untrimmed"}


class ManifestError(ValueError):
    pass


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_endpoint(endpoint: Any, label: str) -> None:
    if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")) or "@" in endpoint or "?" in endpoint or "#" in endpoint:
        raise ManifestError(f"invalid explicit {label} endpoint")


def validate_roles(config: dict) -> None:
    if not isinstance(config, dict) or config.get("kind") not in {"lookup", "trace", "screenshot"}:
        raise ManifestError("original judge kind must be explicitly pinned")
    for role in ("judge", "describer") if config["kind"] == "lookup" else ("judge",):
        value = config.get(role)
        if not isinstance(value, dict) or not isinstance(value.get("model"), str) or not value["model"]:
            raise ManifestError(f"missing original {role} model; never substitute")
        endpoint = value.get("endpoint")
        validate_endpoint(endpoint, "original " + role)
        if value.get("reasoning") not in {"off", "on"}:
            raise ManifestError(f"missing original {role} reasoning setting")
        extra = value.get("request_extra")
        if not isinstance(extra, dict) or any(k not in {"chat_template_kwargs", "enable_thinking", "reasoning_effort", "seed"} for k in extra):
            raise ManifestError("only pinned reasoning/template/seed extras allowed; no model/message/limit override")
        thinking = extra.get("chat_template_kwargs", {})
        if not isinstance(thinking, dict) or any(k != "enable_thinking" for k in thinking):
            raise ManifestError("unsupported chat-template override")
        flags = [x for x in [thinking.get("enable_thinking"), extra.get("enable_thinking")] if x is not None]
        if any(type(x) is not bool or x != (value["reasoning"] == "on") for x in flags):
            raise ManifestError("reasoning configuration contradicts request extras")
        if "qwen3-vl-235b" in value["model"].lower() and (value["reasoning"] != "off" or not flags or any(flags)):
            raise ManifestError("Qwen3-VL235B default group requires reasoning off explicitly on wire")


def native_encode(raw: bytes) -> str:
    """The v1 encode_frame transform, except that missing or unreadable frames raise."""
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as opened:
        im = opened.convert("RGB")
        if im.width > 1000:
            im = im.resize((1000, max(1, int(im.height * 1000 / im.width))))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def write_json_exclusive(path: Path, obj: Any) -> None:
    raw = canonical(obj) + b"\n"
    # fsync before rename prevents journal responses being acknowledged before durability.
    tmp = path.with_name(path.name + ".part")
    with tmp.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)  # atomic, no overwrite even under duplicate workers
    tmp.unlink()
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("judge endpoint redirected; refuse silent endpoint substitution")


class HTTPJudge:
    def __init__(self, timeout: float = 120):
        self.timeout = timeout

    def __call__(self, endpoint: str, body: dict) -> dict:
        # The API key comes from evaluation/endpoint_auth.py, chosen exactly as for the
        # v1 judge (JUDGE_API_KEY / DESCRIBER_API_KEY / AMBIGUI_API_KEY). No header
        # when no key is set. The request body is unchanged and the key is never
        # journaled. Timeout 120 s, no retry (the tested v2 setting).
        from evaluation.endpoint_auth import auth_headers
        url = endpoint.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(url, data=canonical(body), headers={"Content-Type": "application/json", **auth_headers(endpoint)}, method="POST")
        try:
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=self.timeout) as response:
                return {"raw_base64": base64.b64encode(response.read()).decode(), "http_status": response.status}
        except urllib.error.HTTPError as response:
            # Error bodies are raw responses too: journal them before rejecting the verdict.
            return {"raw_base64": base64.b64encode(response.read()).decode(), "http_status": response.code}
