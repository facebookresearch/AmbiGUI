# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Select and build the four fixed in-context demonstrations from training-only harvest rollouts.

Inputs (none ship with this repository; see docs/icl.md):

* ``train``: the training task file (rows with ``split == "train"``). A demonstration may only
  come from a task in it, and only when the task's ``source`` is an approved source
  (``SOURCE_RULE``: mind2web, meta-gui, osworld). Every other source is rejected.
* ``eval``: the task file that will be evaluated, and ``exclusion``: any further benchmark
  files. A training task whose family (ID before ``__``) or normalised instruction matches any
  of these rows is rejected, so no demonstration overlaps an evaluated task.
* harvest chunk archives (``<run>/harvest/<chunk>.tgz``) of earlier rollouts on training tasks,
  named in the config's ``demo_sources``.
"""
from __future__ import annotations

import ast
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

from .common import (digest, file_sha, read_jsonl, task_family, task_id,
                     task_text, text_key, write_json)
from .judging import grade_message

# Approved sources of training tasks. The older spellings in some task rows map to the same
# source. A training task with any other source (or none) is rejected as "source_not_approved".
APPROVED_SOURCES = {"mind2web": "mind2web", "multimodal-mind2web": "mind2web",
                    "meta-gui": "meta-gui", "osworld": "osworld", "osworld-human": "osworld"}
SOURCE_RULE = {"version": "approved-training-sources-v1",
               "approved": sorted(set(APPROVED_SOURCES.values())),
               "accepted_spellings": dict(sorted(APPROVED_SOURCES.items()))}
MANIFEST_VERSION = "isolated-demo-v2"  # v2 = v1 selection plus the approved-source rule

BAD_ENDS = {"harness_error", "generation_error", "truncated", "env_unreachable",
            "env_blocked", "unfinished"}
PAGE_VERBS = {"fill", "click", "dblclick", "hover", "select_option", "goto", "scroll", "press"}


def approved_source(row: dict) -> str | None:
    """The approved source name of a training task row, or None if its source is not approved."""
    return APPROVED_SOURCES.get(str(row.get("source") or "").strip().casefold())


def _message_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        return "\n".join(p.get("text", "") for p in (content or []) if p.get("type") == "text")
    return ""


def _code(raw: str) -> tuple[str, dict, str] | None:
    # Extract only a complete policy emission (the last XML function call, else the first JSON
    # object with a "name").
    visible = raw.rsplit("</think>", 1)[-1]
    name, arguments = None, {}
    functions = list(re.finditer(r"<function=([A-Za-z_][\w]*)>", visible, re.I))
    if functions:
        last = functions[-1]
        name = last.group(1)
        arguments = {k: v.strip() for k, v in re.findall(
            r"<parameter=([A-Za-z_][\w]*)>(.*?)</parameter>", visible[last.end():], re.S | re.I)}
    else:
        decoder = json.JSONDecoder()
        position = 0
        while True:
            start = visible.find("{", position)
            if start < 0:
                break
            try:
                obj, length = decoder.raw_decode(visible[start:])
            except json.JSONDecodeError:
                position = start + 1
                continue
            if isinstance(obj, dict) and "name" in obj:
                name = str(obj["name"])
                arguments = obj.get("arguments") or obj.get("parameters") or {}
                break
            position = start + length
    if not name or name.lower() not in {"browse", "web", "browser"} or not isinstance(arguments, dict):
        return None
    try:
        code = str(arguments.get("action") or "").strip()
        if not re.match(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", code):
            # Named-argument schema encoding belongs to the repo's action contract.
            from evaluation.browse_tool import action_code
            code = action_code(name, arguments)
        call = ast.parse(code, mode="eval").body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            return None
    except Exception:
        return None
    return call.func.id, {"name": "browse", "arguments": {"action": code}}, code


def _question(code: str) -> str:
    call = ast.parse(code, mode="eval").body
    args = list(call.args) + [kw.value for kw in call.keywords]
    for arg in args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return ""


def _observation_excerpt(text: str, code: str, limit: int = 6000) -> tuple[str, str]:
    if len(text) <= limit:
        return text, "none"
    # Keep exact text, not a generated caption; a bid window is added only when
    # truncating the prefix would remove the element the actual action addresses.
    prefix = text[:limit - 1200]
    ids = re.findall(r"['\"]([A-Za-z0-9]+)['\"]", code)
    extra = []
    for bid in ids[:2]:
        match = re.search(rf"\[{re.escape(bid)}\]", text)
        if match and match.start() >= len(prefix):
            extra.append(text[max(0, match.start() - 250):match.start() + 700])
    excerpt = prefix + ("\n[recorded observation excerpt continues around the addressed bid]\n" +
                        "\n".join(extra) if extra else "")
    return excerpt, "fixed prefix + exact bid-local excerpt, at most ~6000 characters"


def candidate(rec: dict, row: dict, provenance: dict, sidecar: str = "") -> dict | None:
    if (rec.get("error") or rec.get("judge_error") or rec.get("end_reason") in BAD_ENDS
            or any(t.get("finish_reason") == "length" for t in rec.get("turns", []))):
        return None
    instruction = task_text(row)
    if not instruction or len(instruction) > 3000:
        return None
    if rec.get("instruction") != instruction:
        # Do not silently bless old first-utterance prompts as full-task demos.
        return None
    label = row.get("ambiguity")
    if rec.get("ambiguity") != label:
        return None
    params = {p.get("name") for p in row.get("open_items", [])}
    eligible = []
    for i, turn in enumerate(rec.get("turns") or []):
        parsed = _code(turn.get("content") or "")
        if not parsed:
            continue
        verb, canonical_call, code = parsed
        if label == "Clear":
            if rec.get("success") is not True or rec.get("asks"):
                return None
            if verb not in PAGE_VERBS:
                continue
            matching = [a for a in rec.get("actions", [])
                        if (a.get("args") or {}).get("action") == code]
            # Harvest records carry no per-action turn index, so reject a code with any
            # failed dispatch rather than match an earlier failed emission to a later
            # successful repetition.
            if not matching or any(a.get("result") != "ok" for a in matching):
                continue
            dispatch = matching[0]
            kind, question = "successful_clear_act", None
        elif label == "Ambiguous":
            if verb != "send_msg_to_user":
                continue
            question = _question(code)
            credited = set((rec.get("ask_targets") or {}).get(question) or []).intersection(params)
            if not credited or question not in (rec.get("asks") or []):
                continue
            # First actually credited question only; no failed action prefix is retained.
            kind = "isolated_credited_question"
            dispatch = next((a for a in rec.get("actions", []) if a.get("tool") == "ASK"
                             and (a.get("args") or {}).get("question") == question), None)
            if dispatch is None:
                continue
        else:
            return None
        obs = _message_text(turn.get("messages") or [])
        conversion = "latest actual user observation; earlier conversation omitted"
        instruction_only = obs.strip().startswith("Instruction:") and "status:" not in obs
        if i == 0 and instruction_only and sidecar:
            obs = sidecar
            conversion = "recorded initial accessibility observation replaces omitted initial demo screenshot"
        elif i > 0 and instruction_only:
            # An initial sidecar cannot describe a later page. Without the
            # actual turn's observation, do not fabricate an isolated demo.
            continue
        if kind == "successful_clear_act" and not ("status:" in obs or "[" in obs):
            # A bid with no screenshot or text addressing it is an unusable demo.
            if verb != "goto":
                continue
        if obs.strip() == f"Instruction: {instruction}".strip():
            obs = "(training screenshot omitted; textual demonstration)"
        obs, truncation = _observation_excerpt(obs, code)
        # Prefer a concrete page action over scroll/goto, without looking at eval outputs.
        priority = 0 if verb in {"fill", "click", "select_option"} else 1
        result = {"task_id": task_id(row), "family": task_family(task_id(row)),
                  "split": "train", "source": row.get("source"), "label": label,
                  "instruction": instruction, "instruction_sha256": text_key(instruction),
                  "kind": kind, "question": question, "observation_text": obs,
                  "assistant": "<tool_call>" + json.dumps(canonical_call, ensure_ascii=False) + "</tool_call>",
                  "raw_policy_output": turn.get("content") or "", "source_turn": i,
                  "source_turn_number": turn.get("turn"), "source_rollout_id": rec.get("rollout_id"),
                  "source_policy_model": rec.get("policy_model"),
                  "source_rollout_success": rec.get("success"), "source_end_reason": rec.get("end_reason"),
                  "dispatch_evidence": dispatch, "provenance": provenance,
                  "source_record_sha256": digest(rec), "source_messages_sha256": digest(turn.get("messages")),
                  "modality": "text-only training demonstration; query retains live screenshot",
                  "input_reformatting": conversion, "observation_truncation": truncation,
                  "output_reformatting": "strip policy reasoning and serialise only its parsed actual action in the current browse schema; no invented action/question",
                  "preceding_turns_demonstrated": 0, "priority": priority,
                  "source_record": rec}
        eligible.append(result)
        if kind == "isolated_credited_question":
            break
    if not eligible:
        return None
    return sorted(eligible, key=lambda x: (x["priority"], x["source_turn"]))[0]


SECRET_TERMS = re.compile(r"\b(?:passwords?|passwd|passcodes?|api[-_ ]?keys?|(?:access|bearer|auth(?:entication)?|login|sign[-_ ]?in)[-_ ]?(?:tokens?|keys?|secrets?)|(?:one[-_ ]?time|two[-_ ]?factor|2fa|mfa|verification|security)[-_ ]?(?:codes?|tokens?|passwords?|passcodes?)|login[-_ ]?secrets?)\b", re.I)


def qualification_reason(cand: dict, cfg: dict) -> str | None:
    """Predeclared data-only predicates; never inspect current eval outputs."""
    if cand["kind"] == "successful_clear_act" and cfg.get("require_completed_clear_demos"):
        if cand["source_end_reason"] not in {"finished", "commit_withheld"}:
            return "clear_not_self_ended_completed_success"
        rec = cand.get("source_record") or {}
        if rec.get("cut_off") or rec.get("cutoff"):
            return "clear_cut_off_source"
    if cand["kind"] == "isolated_credited_question" and cfg.get("exclude_secret_request_demos"):
        if SECRET_TERMS.search(cand.get("question") or ""):
            return "positive_demo_contains_login_secret_term"
    return None


def choose(candidates: list[dict], seed: int) -> list[dict]:
    selected, families, texts, responses = [], set(), set(), set()
    # Fixed balanced order avoids an all-ask suffix. Demonstration selection never
    # consults evaluated-task emissions, labels beyond exclusion checks, or query scores.
    for kind in ("successful_clear_act", "isolated_credited_question",
                 "successful_clear_act", "isolated_credited_question"):
        pool = sorted((c for c in candidates if c["kind"] == kind),
                      key=lambda c: digest([seed, c["task_id"], c["source_rollout_id"], c["source_turn"]]))
        pick = next((c for c in pool if c["family"] not in families
                     and c["instruction_sha256"] not in texts
                     and text_key(c["assistant"]) not in responses), None)
        if pick is None:
            raise ValueError(f"not enough unique training-only {kind} candidates")
        selected.append(pick)
        families.add(pick["family"]); texts.add(pick["instruction_sha256"])
        responses.add(text_key(pick["assistant"]))
    return selected


def exclusion_entries(rows: list[dict]) -> list[dict]:
    """ID, family and normalised-instruction hash of each benchmark row (exclusion only)."""
    return [{"id": task_id(r), "family": task_family(task_id(r)), "text_sha256": text_key(task_text(r))}
            for r in rows]


def input_hashes(inputs: dict) -> dict:
    """sha256 of the train file and the evaluated task file, and the exclusion-index digest.

    Only the inputs that are given are hashed: at run time there is no training file
    (``inputs["train"]`` is None) and no exclusion list (key absent)."""
    out = {}
    if inputs.get("train") is not None:
        out["train_manifest_sha256"] = file_sha(inputs["train"])
    out["eval_tasks_sha256"] = file_sha(inputs["eval"])
    if "exclusion" in inputs:
        rows = [r for path in inputs.get("exclusion") or [] for r in read_jsonl(path)]
        out["exclusion_index_sha256"] = digest(exclusion_entries(rows))
        out["exclusion_files_sha256"] = [file_sha(p) for p in inputs.get("exclusion") or []]
    return out


def check_no_overlap(demos: list[dict], rows: list[dict]) -> None:
    """Refuse when a demonstration shares an ID, a task family or a normalised instruction with
    any of ``rows`` (the evaluated tasks, or further benchmark files)."""
    entries = exclusion_entries(rows)
    ids = {e["id"] for e in entries}
    families = {e["family"] for e in entries}
    texts = {e["text_sha256"] for e in entries}
    for demo in demos:
        if (demo["task_id"] in ids or demo["family"] in families
                or demo["instruction_sha256"] in texts or text_key(demo["instruction"]) in texts):
            raise ValueError(f"demonstration overlaps an evaluated or excluded task: {demo['task_id']}")


def build(cfg: dict, inputs: dict, harvest_root: Path, out: Path, judge_client, judge_model: str) -> dict:
    train = read_jsonl(inputs["train"])
    train_by_id = {task_id(r): r for r in train if r.get("split") == "train"}
    evaluated = read_jsonl(inputs["eval"])
    others = exclusion_entries([r for path in inputs.get("exclusion") or [] for r in read_jsonl(path)])
    blocked_families = {task_family(task_id(r)) for r in evaluated} | {e["family"] for e in others}
    blocked_texts = {text_key(task_text(r)) for r in evaluated} | {e["text_sha256"] for e in others}
    candidates, scanned, rejected = [], 0, Counter()
    exclusion_audit = []
    archive_provenance = []
    named_properties, named_actions = {}, set()
    if cfg.get("preserve_original_named_argument_demos"):
        from evaluation.tool_contract import tool_json_schema
        named_properties = tool_json_schema()["function"]["parameters"]["properties"]
        named_actions = set(named_properties["action"]["enum"])
    # This is a predeclared, fixed harvest-chunk pool, not a search through eval outputs.
    for source in cfg["demo_sources"]:
        archive = Path(harvest_root) / source["path"]
        if not archive.is_file():
            raise FileNotFoundError(f"harvest chunk not found: {archive}")
        provenance = {**source, "archive_sha256": file_sha(archive)}
        archive_provenance.append(provenance)
        # Stream in archive order. The initial observation precedes record.json in the
        # archive; seeking backwards for every sidecar would make gzip work quadratic.
        with tarfile.open(archive, mode="r|gz") as tar:
            observations = {}
            for member in tar:
                if not member.isfile():
                    continue
                if member.name.endswith("/step_00_initial.obs.json") and "/frames/" in member.name:
                    record_dir = member.name.split("/frames/", 1)[0]
                    observations[record_dir] = str(json.load(tar.extractfile(member)).get("text") or "")
                    continue
                if not member.name.endswith("/record.json"):
                    continue
                scanned += 1
                rec = json.load(tar.extractfile(member))
                row = train_by_id.get(str(rec.get("task_id")))
                if row is None:
                    rejected["not_exact_train_split_ID"] += 1; continue
                if approved_source(row) is None:
                    rejected["source_not_approved"] += 1; continue
                if task_family(task_id(row)) in blocked_families or text_key(task_text(row)) in blocked_texts:
                    rejected["eval_family_or_normalised_text_overlap"] += 1; continue
                record_dir = member.name.rsplit("/", 1)[0]
                sidecar = observations.get(record_dir, "")
                cand = candidate(rec, row, {**provenance, "record_member": member.name}, sidecar)
                if cand is not None:
                    cand["approved_source"] = approved_source(row)
                if cand is None:
                    rejected["quality_or_emitted_action_contract"] += 1
                else:
                    reason = qualification_reason(cand, cfg)
                    if reason is None and cfg.get("preserve_original_named_argument_demos"):
                        from .parsing import extract_call
                        name, arguments = extract_call(cand["raw_policy_output"])
                        if (name not in {"browse", "web", "browser"} or arguments.get("action") not in named_actions
                                or not set(arguments).issubset(named_properties)):
                            reason = "not_original_advertised_named_argument_emission"
                        else:
                            cand["assistant"] = "<tool_call>" + json.dumps({"name": "browse", "arguments": arguments}, ensure_ascii=False) + "</tool_call>"
                            cand["output_reformatting"] = "strip reasoning; serialise original policy-emitted named arguments (XML/JSON) without semantic changes; normalise carrier name to advertised browse"
                            cand["original_named_arguments"] = arguments
                    if reason is not None:
                        rejected[reason] += 1
                        exclusion_audit.append({"task_id": cand["task_id"], "source_rollout_id": cand["source_rollout_id"],
                            "kind": cand["kind"], "source_rollout_success": cand["source_rollout_success"],
                            "source_end_reason": cand["source_end_reason"], "question": cand["question"],
                            "raw_policy_output": cand["raw_policy_output"], "reason": reason,
                            "source_record_sha256": cand["source_record_sha256"], "provenance": cand["provenance"]})
                    else:
                        candidates.append(cand)
    chosen = choose(candidates, int(cfg["seed"]))
    from evaluation.missing_parameters import parameters_from_row
    for c in chosen:
        if c["question"]:
            grade = grade_message(judge_client, judge_model, c["instruction"], c["question"],
                                  parameters_from_row(train_by_id[c["task_id"]]))
            c["fresh_question_verification"] = grade
            if not grade["valid"] or grade["kind"] != "question" or not grade["targets"]:
                raise ValueError(f"fixed selected training demo failed fresh verification: {c['task_id']}: {grade}")
    messages = []
    for i, c in enumerate(chosen, 1):
        label = ("Clear: successful policy action" if c["label"] == "Clear"
                 else "Ambiguous: isolated credited question, not a certified action trajectory")
        messages.extend([
            {"role": "user", "content": f"Training demonstration {i}/4 ({label}; screenshot omitted).\nInstruction: {c['instruction']}\nRecorded observation:\n{c['observation_text']}"},
            {"role": "assistant", "content": c["assistant"]},
        ])
    manifest = {"version": MANIFEST_VERSION, "seed": int(cfg["seed"]), "shots": 4,
                "order": [c["kind"] for c in chosen], "messages": messages,
                "demo_messages_sha256": digest(messages),
                "selection": "predeclared harvest chunks; exact training IDs; approved sources only; quality gates; seed-hash ordering; no eval-output selection",
                "source_rule": SOURCE_RULE,
                "scanned_records": scanned, "eligible_candidates": len(candidates),
                "rejections": dict(rejected), "exclusion_audit": exclusion_audit,
                "qualification_predicates": {k: cfg.get(k, False) for k in ("require_completed_clear_demos", "exclude_secret_request_demos", "preserve_original_named_argument_demos")},
                "source_archives": archive_provenance,
                **input_hashes(inputs),
                "demonstrations": [{k: v for k, v in c.items() if k not in ("source_record", "priority")}
                                   for c in chosen]}
    manifest["fingerprint"] = digest(manifest)
    write_json(out / "demonstrations.json", manifest)
    write_json(out / "demo_source_records.json", [c["source_record"] for c in chosen])
    write_json(out / "demo_messages.json", messages)
    return manifest


def verify_frozen(manifest: dict, inputs: dict, cfg: dict, source_records: list[dict]) -> None:
    """Re-check frozen demonstrations against their inputs and source records.

    ``prepare`` calls it with the training file, the evaluated task file and the exclusion files.
    The ICL arm calls it again before it starts (``condition.verify_for_run``) with
    ``inputs = {"train": None, "eval": <the task file the run uses>}``: the task file must be the one
    the demonstrations were checked against, and no demonstration may overlap any of its tasks.
    """
    if manifest.get("fingerprint") != digest({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("demo provenance fingerprint drift")
    if manifest.get("shots") != 4 or len(manifest.get("messages", [])) != 8:
        raise ValueError("not the frozen four-shot protocol")
    if manifest["demo_messages_sha256"] != digest(manifest["messages"]):
        raise ValueError("demo message hash drift")
    current = input_hashes(inputs)
    for name, value in current.items():
        if name != "exclusion_files_sha256" and manifest.get(name) != value:
            raise ValueError(f"data drift: {name}")
    demos = manifest["demonstrations"]
    required = ("require_completed_clear_demos", "exclude_secret_request_demos", "preserve_original_named_argument_demos")
    configured = {key: cfg.get(key) for key in required}
    if not all(value is True for value in configured.values()):
        raise ValueError("this comparison requires the configured data-only qualification gates")
    if manifest.get("version") != MANIFEST_VERSION or manifest.get("qualification_predicates") != configured:
        raise ValueError("restored manifest qualification version/gates do not match the configured requirements")
    if manifest.get("source_rule") != SOURCE_RULE:
        raise ValueError("demonstrations were not selected with the approved-source rule")
    predicates = configured
    sources = {digest(record): record for record in source_records}
    train_rows = None
    if inputs.get("train") is not None:
        train_rows = {task_id(row): row for row in read_jsonl(inputs["train"]) if row.get("split") == "train"}
    check_no_overlap(demos, read_jsonl(inputs["eval"]))
    if inputs.get("exclusion"):
        check_no_overlap(demos, [r for path in inputs["exclusion"] for r in read_jsonl(path)])
    from .parsing import extract_call
    for index, demo in enumerate(demos):
        record = sources.get(demo["source_record_sha256"])
        if record is None or record.get("task_id") != demo["task_id"]:
            raise ValueError("frozen demo lacks immutable external training source-record evidence")
        if APPROVED_SOURCES.get(str(demo.get("source") or "").strip().casefold()) is None:
            raise ValueError(f"demonstration source is not approved: {demo['task_id']}")
        if train_rows is not None:
            row = train_rows.get(demo["task_id"])
            if row is None:
                raise ValueError("frozen demo task is not in the training file")
            if approved_source(row) is None or row.get("source") != demo.get("source"):
                raise ValueError(f"training task source is not approved: {demo['task_id']}")
            instruction, label = task_text(row), row.get("ambiguity")
        else:
            instruction, label = demo["instruction"], demo["label"]
        if record.get("instruction") != instruction or record.get("error") or record.get("judge_error"):
            raise ValueError("frozen demo source evidence is not an error-free full-task training record")
        if (demo["source_end_reason"] != record.get("end_reason")
                or demo["source_rollout_success"] != record.get("success")
                or demo["label"] != label):
            raise ValueError("frozen qualification claims differ from external source evidence")
        source_turn = int(demo["source_turn"])
        turns = record.get("turns") or []
        if source_turn >= len(turns) or turns[source_turn].get("content") != demo["raw_policy_output"]:
            raise ValueError("frozen demo does not match its actual source policy turn")
        if digest(turns[source_turn].get("messages")) != demo["source_messages_sha256"]:
            raise ValueError("frozen demo source-observation messages differ from external record")
        if source_turn > 0:
            actual_observation = _message_text(turns[source_turn].get("messages") or [])
            if not actual_observation or (actual_observation.startswith("Instruction:") and "status:" not in actual_observation):
                raise ValueError("later demo lacks its actual current-page observation")
            if not demo["observation_text"].startswith(actual_observation[:min(4800, len(actual_observation))]):
                raise ValueError("later demo observation is not bound to its actual source turn")
        if demo["kind"] == "successful_clear_act" and (record.get("success") is not True or record.get("asks")):
            raise ValueError("completed-success Clear evidence is missing or contains asks")
        if qualification_reason({**demo, "source_record": record}, predicates) is not None:
            raise ValueError(f"frozen demonstration fails the data-only qualification gates: {demo['task_id']}")
        if demo["source_turn"] > 0 and "initial accessibility" in demo.get("input_reformatting", ""):
            raise ValueError("later-turn demo cannot use an initial observation sidecar")
        if predicates.get("preserve_original_named_argument_demos"):
            name, arguments = extract_call(demo["raw_policy_output"])
            if name not in {"browse", "web", "browser"} or arguments != demo.get("original_named_arguments"):
                raise ValueError("frozen demo arguments differ from actual policy emission")
            expected = "<tool_call>" + json.dumps({"name": "browse", "arguments": arguments}, ensure_ascii=False) + "</tool_call>"
            if demo["assistant"] != expected or manifest["messages"][2 * index + 1]["content"] != expected:
                raise ValueError("frozen assistant prefix differs from recorded original named-argument action")
    if len({d["family"] for d in demos}) != 4:
        raise ValueError("duplicate demo task families")
    if Counter(d["kind"] for d in demos) != Counter({"successful_clear_act": 2, "isolated_credited_question": 2}):
        raise ValueError("unbalanced demonstrations")
