# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Explicit future-caller adapter for the source-verified 2.0.3 judge.

No discovery, credentials, retries, endpoint relocation, or v1 fallback. The caller
must certify curated-only intent, pin both original roles, and durably journal
full request/response bytes. Existing positional training APIs remain v1.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evaluation.score_rollout_v2 import (
    VERSION, DEFAULT_OUR_MODEL, canonical, curated_intent, digest,
    score_rollout_v2,
)

CURATED_CONTRACT = "curated_open_items_only"


class SourceError(ValueError):
    """An absent, malformed or mismatched source is unknown, not policy failure."""


def exact_task_id(row: dict) -> str:
    if not isinstance(row, dict):
        raise SourceError("curated_row_not_object")
    values = [row[k] for k in ("task_id", "sample_id") if k in row]
    if not values or any(not isinstance(v, str) or not v for v in values):
        raise SourceError("missing_exact_task_id")
    if any(v != values[0] for v in values):
        raise SourceError("conflicting_exact_task_ids")
    return values[0]


def load_original_task_rows(path: Path) -> list[dict]:
    """Explicit future CLI uses raw release rows, not a legacy normalising builder."""
    try:
        rows = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise SourceError(f"invalid_original_task_rows: {exc}") from exc
    for row in rows:
        exact_task_id(row)
        if not isinstance(row.get("instruction"), str) or not row["instruction"]:
            raise SourceError("missing_original_instruction_in_release_row")
    return rows


def validated_curated_row(row: dict | None, task_id: str, instruction: str,
                          contract: str) -> dict:
    if contract != CURATED_CONTRACT:
        raise SourceError("uncertified_curated_only_source: legacy combined full84 is not accepted")
    if row is None:
        raise SourceError("missing_curated_row")
    if exact_task_id(row) != task_id:
        raise SourceError("curated_task_id_mismatch: no fuzzy or repeat-suffix join")
    if not isinstance(instruction, str) or not instruction:
        raise SourceError("missing_original_instruction")
    if row.get("instruction") != instruction:
        raise SourceError("curated_original_instruction_mismatch: no greeting substitution")
    curated_intent(row)  # all original open_items, no origin filter or derived merge
    return row


def load_curated_sidecar(path: Path, contract: str) -> dict[str, dict]:
    if contract != CURATED_CONTRACT:
        raise SourceError("uncertified_curated_only_sidecar")
    try:
        raw = path.read_bytes()
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            obj = json.loads(raw)
            rows = obj if isinstance(obj, list) else [obj]
    except (OSError, ValueError) as exc:
        raise SourceError(f"invalid_curated_sidecar: {exc}") from exc
    index = {}
    for row in rows:
        key = exact_task_id(row)
        if key in index:
            raise SourceError(f"duplicate_curated_task_id: {key}")
        index[key] = row
    return index


def validate_config(config: dict, *, kind: str, judge_url: str, judge_model: str,
                    describer_url: str | None, describer_model: str | None) -> dict:
    from evaluation.judge_v2_io import NATIVE_CONTRACT, validate_roles
    if not isinstance(config, dict) or config.get("version") != VERSION:
        raise SourceError("judge_config_version_mismatch")
    if config.get("intent_contract") != CURATED_CONTRACT:
        raise SourceError("judge_config_requires_curated_only_contract")
    if config.get("native_contract") != NATIVE_CONTRACT:
        raise SourceError("judge_config_native_contract_mismatch")
    roles = config.get("original_judge")
    validate_roles(roles)
    if kind != "lookup" or roles["kind"] != kind:
        raise SourceError("release_203_requires_original_lookup_route")
    expected = {"judge": (judge_url, judge_model),
                "describer": (describer_url, describer_model)}
    for role, (endpoint, model) in expected.items():
        value = roles[role]
        if value["endpoint"] != endpoint or value["model"] != model:
            raise SourceError(f"{role}_endpoint_or_model_mismatch: no substitution")
        # The served name is the user's choice (vLLM --served-model-name). It must be
        # the same model, Qwen3-VL-235B-A22B-Instruct, in any letter case, and it must
        # equal --judge-model / --describer-model exactly (checked above).
        if not isinstance(model, str) or model.casefold() != DEFAULT_OUR_MODEL.casefold() or value["reasoning"] != "off":
            raise SourceError(f"{role}_requires_original_Q235_reasoning_off")
        if not isinstance(value.get("backend_identity"), str) or not value["backend_identity"].strip():
            raise SourceError(f"{role}_missing_explicit_backend_or_tier_identity")
        if value["request_extra"].get("chat_template_kwargs", {}).get("enable_thinking") is not False:
            raise SourceError(f"{role}_requires_explicit_reasoning_false_on_wire")
    if "endpoint_relocations" in config or "judge_override" in config:
        raise SourceError("future_caller_rejects_endpoint_or_backend_overrides")
    return roles


def validate_original_time(value: dict | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SourceError("invalid_original_task_time_evidence")
    from datetime import date, datetime
    try:
        if value.get("timestamp") is not None:
            datetime.fromisoformat(value["timestamp"].replace("Z", "+00:00"))
        if value.get("date") is not None:
            date.fromisoformat(value["date"])
        if value.get("timezone") is not None and not isinstance(value["timezone"], str):
            raise ValueError("timezone is not text")
    except (ValueError, TypeError, AttributeError) as exc:
        raise SourceError("invalid_original_task_date: no mtime/current-time inference") from exc
    return value


def validated_native_frames(shots: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    from evaluation.judge_v2_io import native_encode, sha
    if not shots:
        raise SourceError("missing_encoded_frame: no original screenshots")
    encoded, hashes = {}, {}
    for path in shots:
        if not isinstance(path, str) or not path:
            raise SourceError("invalid_original_frame_path")
        try:
            raw = Path(path).read_bytes()
            url = native_encode(raw)  # original Pillow RGB/width1000/JPEG-q80 path
            if not url.startswith("data:image/jpeg;base64,"):
                raise ValueError("not native JPEG wire")
            jpeg = base64.b64decode(url.split(",", 1)[1], validate=True)
            if not jpeg.startswith(b"\xff\xd8\xff"):
                raise ValueError("invalid encoded JPEG")
        except Exception as exc:
            raise SourceError(f"invalid_or_missing_encoded_frame: {path}: {type(exc).__name__}: {exc}") from exc
        encoded[path], hashes[path] = url, sha(raw)
    return encoded, hashes


class RequestJournal:
    """Callback completes BEFORE transport; full raw response completes BEFORE parse."""
    def __init__(self, roles: dict, callback: Callable[[dict], None], transport: Callable):
        if not callable(callback) or not callable(transport):
            raise SourceError("missing_full_raw_journal_callback_or_transport")
        self.roles, self.callback, self.transport = roles, callback, transport
        self.index = 0

    def request(self, role: str, body: dict) -> str:
        config = self.roles[role]
        if body["model"] != config["model"]:
            raise SourceError("request_model_substitution")
        if body.get("temperature") != 0.0 or body.get("max_tokens") != (600 if role == "describer" else 1200):
            raise SourceError("native_request_budget_or_temperature_mismatch")
        payload = {**body, **config["request_extra"]}
        index = self.index
        self.index += 1
        binding = {"version": VERSION, "index": index, "role": role,
                   "endpoint": config["endpoint"], "backend_identity": config["backend_identity"],
                   "request_sha256": digest(payload)}
        self.callback({**binding, "event": "request", "body": payload,
                       "raw_base64": base64.b64encode(canonical(payload)).decode()})
        try:
            response = self.transport(config["endpoint"], payload)
        except Exception as exc:
            self.callback({**binding, "event": "transport_error", "source_error": f"{type(exc).__name__}: {exc}"})
            raise
        if not isinstance(response, dict) or not isinstance(response.get("raw_base64"), str):
            raise SourceError("transport_must_return_full_raw_response_bytes")
        raw = base64.b64decode(response["raw_base64"], validate=True)
        self.callback({**binding, "event": "response", "raw_base64": response["raw_base64"],
                       "response_sha256": __import__("hashlib").sha256(raw).hexdigest(),
                       "http_status": response.get("http_status")})
        status = response.get("http_status")
        if status is not None and (type(status) is not int or not 200 <= status < 300):
            raise SourceError(f"judge_http_response_error: {status}")
        obj = json.loads(raw)
        if obj.get("model") != config["model"]:
            raise SourceError("response_model_missing_or_mismatch")
        choice = obj["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise SourceError("judge_response_truncated_or_interrupted")
        message = choice["message"]
        if message.get("reasoning_content"):
            raise SourceError("reasoning_false_role_emitted_reasoning")
        text = message.get("content")
        if not isinstance(text, str) or not text:
            raise SourceError("judge_response_missing_final_text")
        return text


@dataclass(frozen=True)
class SourceVerdict:
    success: bool | None
    why: str
    source_error: str | None = None
    version: str = VERSION
    checks: dict | None = None
    journal_path: str | None = None


def score_source(*, task_id: str, instruction: str, shots: list[str], actions: list[dict],
                 curated_row: dict | None, config: dict, kind: str, judge_url: str,
                 judge_model: str, describer_url: str | None, describer_model: str | None,
                 journal_callback: Callable[[dict], None], transport: Callable,
                 final_url: str = "", original_task_time_evidence: dict | None = None,
                 population_metadata: dict | None = None) -> SourceVerdict:
    """Source errors and unresolved model verdicts retain None, never bool(None)."""
    try:
        roles = validate_config(config, kind=kind, judge_url=judge_url, judge_model=judge_model,
                                describer_url=describer_url, describer_model=describer_model)
        row = validated_curated_row(curated_row, task_id, instruction, config["intent_contract"])
        time = validate_original_time(original_task_time_evidence)
        if not isinstance(actions, list) or any(not isinstance(a, dict) for a in actions):
            raise SourceError("invalid_original_actions")
        if not isinstance(final_url, str):
            raise SourceError("missing_explicit_original_final_url")
        if not callable(journal_callback):
            raise SourceError("missing_full_raw_journal_callback")
        encoded, hashes = validated_native_frames(shots)
        journal_callback({"event": "input", "version": VERSION, "task_id": task_id,
            "instruction": instruction, "actions": actions, "shots": shots,
            "curated_row": row, "curated_semantic_intent": curated_intent(row),
            "config": config, "final_url": final_url, "frame_sha256": hashes,
            "encoded_frame_sha256": {p: digest(url) for p, url in encoded.items()},
            "original_task_time_evidence": time, "population_metadata": population_metadata})
        journal = RequestJournal(roles, journal_callback, transport)
        result = score_rollout_v2(kind, judge_url, judge_model, instruction, shots, actions,
            final_url, describer_url, describer_model, curated_row=row,
            request_fn=journal.request, encoded_frames=encoded,
            original_task_time_evidence=time)
        # A mutation invalidates the result but does not cause a retry or new call.
        from evaluation.judge_v2_io import sha
        if any(sha(Path(p).read_bytes()) != value for p, value in hashes.items()):
            raise SourceError("original_frame_changed_during_judging")
        verdict = SourceVerdict(result.success, result.why, checks=result.checks)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        verdict = SourceVerdict(None, error, source_error=error)
    if callable(journal_callback):
        # If this durability callback fails, the caller must not report a measured verdict.
        journal_callback({"event": "verdict", "version": VERSION, "task_id": task_id,
            "success": verdict.success, "why": verdict.why, "source_error": verdict.source_error,
            "checks": verdict.checks})
    return verdict


class DirectoryJournal:
    def __init__(self, root: Path):
        import uuid
        root.mkdir(parents=True, exist_ok=True)
        self.directory = root / uuid.uuid4().hex
        self.directory.mkdir()
        self.count = 0

    def __call__(self, event: dict) -> None:
        from evaluation.judge_v2_io import write_json_exclusive
        path = self.directory / f"event_{self.count:03d}.json"
        write_json_exclusive(path, event)
        self.count += 1


def add_activation_arguments(parser: Any) -> None:
    parser.add_argument("--judge-version", choices=["judge_v1", VERSION], default="judge_v1",
                        help="Explicit activation; positional training API stays judge_v1")
    parser.add_argument("--curated-sidecar", type=Path, help="Certified curated-only release rows, exact IDs/instructions")
    parser.add_argument("--judge-config", type=Path, help="2.0.3 version/curated contract/native contract/original role pins")
    parser.add_argument("--judge-journal-root", type=Path, help="New output root for full immutable raw judge journals")


def activation_from_args(args: Any) -> tuple[dict | None, dict[str, dict] | None]:
    if args.judge_version == "judge_v1":
        return None, None
    if not args.curated_sidecar or not args.judge_config or not args.judge_journal_root:
        raise SourceError("explicit_203_requires_curated_sidecar_config_and_journal_root")
    try:
        config = json.loads(args.judge_config.read_bytes())
    except (OSError, ValueError) as exc:
        raise SourceError(f"invalid_judge_config: {exc}") from exc
    validate_config(config, kind=args.judge_kind, judge_url=args.judge_url,
        judge_model=args.judge_model, describer_url=args.describer_url,
        describer_model=args.describer_model)
    return config, load_curated_sidecar(args.curated_sidecar, config["intent_contract"])


def score_future_task(*, row: dict, shots: list[str], actions: list[dict], config: dict,
                      curated_row: dict | None, journal_root: Path,
                      kind: str, judge_url: str, judge_model: str,
                      describer_url: str | None, describer_model: str | None,
                      original_task_time_evidence: dict | None = None,
                      population_metadata: dict | None = None,
                      final_url: str = "", transport: Callable | None = None) -> SourceVerdict:
    from evaluation.judge_v2_io import HTTPJudge
    from dataclasses import replace
    journal = DirectoryJournal(journal_root)
    result = score_source(task_id=exact_task_id(row), instruction=row["instruction"],
        shots=shots, actions=actions, curated_row=curated_row, config=config,
        kind=kind, judge_url=judge_url, judge_model=judge_model,
        describer_url=describer_url, describer_model=describer_model,
        journal_callback=journal,
        transport=transport if transport is not None else HTTPJudge(), final_url=final_url,
        original_task_time_evidence=original_task_time_evidence,
        population_metadata=population_metadata)
    return replace(result, journal_path=str(journal.directory))
