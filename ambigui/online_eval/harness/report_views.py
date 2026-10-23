#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""report_views.py <tasks.jsonl> <run_dir> [--out metrics.json] [--expected subsets.json]

Turn a finished run into the paper's results table: the success views and the asking
measures, per policy, with Ambiguous and Clear tasks always kept apart. It reads only
recorded files and calls no model, so it can be rerun on any finished run.

Inputs, for every policy directory <run_dir>/<policy>/ that run_online.sh wrote:
  merged/results.jsonl   pooled outcome of each task (authoritative)
  merged/summary.json    pooled summary (blocked task ids, shard status)
  merged/ask_tasks.json  every message handed to the question grading (phase 2)
  ask_quality.json       the phase-2 grading, when present
  shardN/status.json     shard exit code (124 = the shard hit its process timeout)

Every expected task gets a row. A task without an outcome is never counted as a failure
or as a measured no-ask: it is reported as missing, timed out or blocked.

Success views (a view decides which tasks enter the success denominator):
  harness    every completed task, except tasks set aside because the start page was
             blocked (env_blocked). Tasks whose start page could not be reached
             (env_unreachable) stay in, as failures.
  readme     harness, minus tasks flagged "Stops before the commitment the recording
             completed" (the recording itself stops early, so success is not comparable).
  no_flags   harness, restricted to tasks whose benchmark flags list is empty.
  action_gt  harness, restricted to tasks with action ground truth (not Android-origin).

Judge v2 (or any null verdict): an unresolved task verdict (success null) is left out of
the success rate and counted next to it ("unresolved"). Rows that were not judged
(env_unreachable, harness_error) keep success false, as under v1. For a v1 run with no
null verdict the output is exactly as before (no "unresolved" fields).

Asking measures use phase-2 grading, which excludes report-like messages (statements):
  Amb real asked       Ambiguous tasks with open items where the policy asked >= 1 real question
  Amb on-target        ... where >= 1 question targeted an open item
  Clear real asked     Clear tasks (excluding those flagged as carrying open items) where
                       the policy asked anyway (a false alarm)
  blocker recall / question precision / question F1, recomputed from per-task counts.

The rules are the ones used for the paper's tables.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

STOPS = "Stops before the commitment the recording completed"
LCI = "Labelled Clear but carries open items"
ANDROID = "Ask-only: no action ground truth, the recording is Android"
VIEWS = ("harness", "readme", "no_flags", "action_gt")


def load_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def load_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def unique(rows, key):
    d = {r[key]: r for r in rows}
    if len(d) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return d


def flag_eligible(row, view):
    flags = row.get("flags") or []
    if view == "harness":
        return True
    if view == "readme":
        return STOPS not in flags
    if view == "no_flags":
        return not flags
    if view == "action_gt":
        return row.get("has_action_ground_truth") is True and ANDROID not in flags
    raise ValueError(view)


def success_eligible(row, view):
    return (bool(row.get("result_present")) and row.get("end_reason") != "env_blocked"
            and not row.get("stops_before_commitment") and flag_eligible(row, view))


def asking_eligible(row, arm):
    if row.get("ambiguity") != arm:
        return False
    if arm == "Ambiguous":
        return bool(row.get("open_item_names"))
    return LCI not in (row.get("flags") or []) and not row.get("open_item_names")


def rate(k, n):
    return {"k": k, "n": n, "rate": k / n if n else None}


def frac(x):
    if x is None or x.get("n") is None:
        return "not measured"
    text = f"{x['k']}/{x['n']} ({100 * x['k'] / x['n']:.1f}%)" if x["n"] else "0/0 (not measured)"
    if "unresolved" in x:
        text += f", {x['unresolved']} unresolved"
    return text


def policy_records(pdir: Path, rows: dict, taskorder: list, num_shards_default: int | None):
    """One record per expected task of one policy, with null outcome fields when absent."""
    summary = load_json(pdir / "merged/summary.json", {})
    results = unique(load_jsonl(pdir / "merged/results.jsonl"), "task_id")
    if not set(results) <= set(taskorder):
        raise ValueError(f"Unexpected result for {pdir.name}")
    aq = load_json(pdir / "ask_quality.json", {})
    pt = aq.get("per_task", {}) if aq.get("applicable") is True else {}
    askrecords = load_json(pdir / "merged/ask_tasks.json", []) or []
    asktasks = unique(askrecords, "task_id") if askrecords else {}
    shard_numbers = [int(k[5:]) for k in summary.get("shard_status", {}) if k.startswith("shard") and k[5:].isdigit()]
    shard_dirs = [d for d in pdir.iterdir() if d.is_dir() and d.name.startswith("shard") and d.name[5:].isdigit()]
    num_shards = num_shards_default or max(shard_numbers, default=len(shard_dirs) or 1)
    set_aside = set()
    for sd in shard_dirs:
        set_aside |= set((load_json(sd / "scores.json", {}).get("set_aside", {}) or {}).get("task_ids", []) or [])
    out = []
    for index, tid in enumerate(taskorder, 1):
        row = rows[tid]
        result = results.get(tid)
        r = result or {}
        shardname = f"shard{(index - 1) % num_shards + 1}"
        st = load_json(pdir / shardname / "status.json", {}) or summary.get("shard_status", {}).get(shardname, {})
        blocked = tid in set_aside or tid in summary.get("env_blocked_task_ids", [])
        if result is not None:
            status = r.get("end_reason") if r.get("end_reason") in ("env_unreachable", "env_blocked") else "reachable"
        elif blocked:
            status = "env_blocked"
        elif st.get("phase1_rc") == 124:
            status = "timed_out"
        else:
            status = "missing"
        at = asktasks.get(tid)
        p = pt.get(tid)
        if p and p.get("n_judge_failures"):
            # A message the ask judge could not grade: the task is ungraded, not a
            # task whose questions hit nothing.
            p = None
        real = p.get("n_ask") if p else None
        messages = len(at.get("questions", [])) if at is not None else (len(r.get("asks", [])) if result is not None else None)
        statements = messages - real if real is not None and messages is not None else None
        if statements is not None and statements < 0:
            raise ValueError(f"Negative statement count: {pdir.name}/{tid}")
        out.append({
            "policy": pdir.name, "task_id": tid, "ambiguity": row.get("ambiguity"), "flags": row.get("flags", []),
            "has_action_ground_truth": row.get("has_action_ground_truth"),
            "open_item_names": [x.get("name") for x in row.get("open_items", []) or []],
            "result_present": result is not None, "status": status,
            "success": r.get("success") if result is not None else None,
            "end_reason": r.get("end_reason") if result is not None else ("env_blocked" if blocked else None),
            "judge_error": r.get("judge_error"), "stops_before_commitment": r.get("stops_before_commitment"),
            "judge_version": r.get("judge_version"),
            "phase2_real_questions": real,
            "phase2_on_target_questions": p.get("n_rel") if p else None,
            "phase2_statement_messages": statements,
            "phase2_has_parameters": bool(at.get("parameters")) if at is not None else None,
        })
    return out


def has_unresolved(records):
    """Whether this policy's success views carry an "unresolved" count: a run judged
    by v2, or any recorded null verdict. v1 runs keep their exact output."""
    return any(r.get("result_present") and (r.get("success") is None
               or r.get("judge_version") not in (None, "judge_v1")) for r in records)


def stats_for(records):
    """All denominators are observed counts, with expected/gaps alongside them."""
    track_unresolved = has_unresolved(records)
    stats = {"expected": len(records), "results": sum(bool(r.get("result_present")) for r in records),
             "statuses": dict(collections.Counter(r["status"] for r in records)),
             "judge_errors": sum(bool(r.get("judge_error")) for r in records if r.get("result_present")),
             "success": {}, "asking": {},
             "endings": dict(collections.Counter(r.get("end_reason") for r in records if r.get("result_present")))}
    for view in VIEWS:
        stats["success"][view] = {}
        for arm in ("Ambiguous", "Clear"):
            eligible = [r for r in records if r.get("ambiguity") == arm and flag_eligible(r, view)]
            observed = [r for r in records if r.get("ambiguity") == arm and success_eligible(r, view)]
            resolved = [r for r in observed if r.get("success") is not None]
            cell = {**rate(sum(r.get("success") is True for r in resolved), len(resolved)),
                    "expected": len(eligible), "excluded_or_missing": len(eligible) - len(observed)}
            if track_unresolved:
                cell["unresolved"] = len(observed) - len(resolved)
            stats["success"][view][arm] = cell
    graded = [r for r in records if r.get("result_present") and r.get("phase2_real_questions") is not None]
    stats["phase2_coverage"] = {"graded_tasks": len(graded), "results": stats["results"]}
    for arm in ("Ambiguous", "Clear"):
        planned = [r for r in records if asking_eligible(r, arm)]
        observed = [r for r in graded if asking_eligible(r, arm) and r.get("end_reason") != "env_blocked"]
        completed = [r for r in planned if r.get("result_present") and r.get("end_reason") != "env_blocked"]
        stats["asking"][arm] = {"asked": rate(sum(r["phase2_real_questions"] > 0 for r in observed), len(observed)),
                                "on_target": rate(sum((r.get("phase2_on_target_questions") or 0) > 0 for r in observed), len(observed)),
                                "expected": len(planned), "completed_eligible": len(completed),
                                "ungraded_completed": len(completed) - len(observed), "missing_or_blocked": len(planned) - len(completed)}
    # The scorer's gold set comes from merged/ask_tasks.json parameters, not from the
    # benchmark open_items.
    blockers = [r for r in graded if r.get("phase2_has_parameters") is True]
    total = sum(r["phase2_real_questions"] for r in graded)
    relevant = sum(r.get("phase2_on_target_questions") or 0 for r in graded)
    on_blockers = sum(r["phase2_real_questions"] for r in blockers)
    found = sum((r.get("phase2_on_target_questions") or 0) > 0 for r in blockers)
    rec = found / len(blockers) if blockers else None
    prec = relevant / total if total else None
    bp = sum(r.get("phase2_on_target_questions") or 0 for r in blockers) / on_blockers if on_blockers else None

    def f1(p, r):
        return 2 * p * r / (p + r) if p is not None and r is not None and p + r else (0.0 if p is not None and r is not None else None)
    stats["phase2_recomputed"] = {
        "total_questions": total if graded else None, "relevant_questions": relevant if graded else None,
        "asks_on_blocker_tasks": on_blockers if graded else None, "blocker_found": found if graded else None,
        "blocker_tasks": len(blockers) if graded else None, "recall": rec, "precision": prec, "precision_on_blocker": bp,
        "f1": f1(prec, rec), "f1_on_blocker": f1(bp, rec),
        "statement_messages": sum(r.get("phase2_statement_messages") or 0 for r in graded) if graded else None,
        "note": "No-question precision is undefined (null), not an observed failure. Blocker tasks use actual phase-2 parameters, not benchmark open_items."}
    return stats


def render_table(stats, order=None):
    order = order or sorted(stats)
    head = ["policy / arm", "results / expected", "Amb success harness", "Clear success harness",
            "Amb success README", "Clear success README", "Amb success no flags", "Clear success no flags",
            "Amb success action GT", "Clear success action GT", "Amb real asked", "Amb on-target",
            "Clear real asked (false alarm)", "phase2 tasks", "statements", "blocker recall", "question precision", "question F1", "judge err"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]

    def number(v):
        return "not measured" if v is None else f"{v:.3f}"
    for policy in order:
        s = stats.get(policy)
        if s is None:
            lines.append("| " + policy + " | not run |" + " |" * (len(head) - 2))
            continue
        row = [policy, f"{s['results']}/{s['expected']}"]
        for view in VIEWS:
            row += [frac(s["success"][view][arm]) for arm in ("Ambiguous", "Clear")]
        row += [frac(s["asking"]["Ambiguous"]["asked"]), frac(s["asking"]["Ambiguous"]["on_target"]),
                frac(s["asking"]["Clear"]["asked"]), f"{s['phase2_coverage']['graded_tasks']}/{s['phase2_coverage']['results']}",
                str(s["phase2_recomputed"]["statement_messages"]) if s["phase2_recomputed"]["statement_messages"] is not None else "not measured",
                number(s["phase2_recomputed"]["recall"]), number(s["phase2_recomputed"]["precision"]), number(s["phase2_recomputed"]["f1"]), str(s["judge_errors"])]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Denominators are completed eligible outcomes for success and phase-2-covered eligible outcomes for asking.",
              "Missing, timed-out and blocked rows remain explicit; they are not counted as successful or as measured no-asks.", ""]
    for policy in order:
        if policy not in stats:
            continue
        s = stats[policy]
        lines.append(f"- {policy}: statuses {s['statuses']}; phase-2 eligible gaps Amb {s['asking']['Ambiguous']['ungraded_completed']} ungraded + {s['asking']['Ambiguous']['missing_or_blocked']} absent/blocked, Clear {s['asking']['Clear']['ungraded_completed']} ungraded + {s['asking']['Clear']['missing_or_blocked']} absent/blocked; endings {s['endings']}.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="")
    ap.add_argument("--expected", default="", help="JSON {policy: [task ids in run order]} for per-policy task subsets")
    a = ap.parse_args()
    rows = unique(load_jsonl(a.tasks), "sample_id")
    order = list(rows)
    expected = load_json(a.expected, {}) if a.expected else {}
    raw = Path(a.run_dir)
    manifest = load_json(raw / "_shared_/run_manifest.json", {}) or load_json(raw / "run_manifest.json", {})
    stats = {}
    for pdir in sorted(p for p in raw.iterdir() if p.is_dir() and not p.name.startswith("_") and (p / "merged").exists()):
        taskorder = expected.get(pdir.name) or order
        records = policy_records(pdir, rows, taskorder, manifest.get("shards_per_policy"))
        stats[pdir.name] = stats_for(records)
    print(render_table(stats), end="")
    if a.out:
        Path(a.out).write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
