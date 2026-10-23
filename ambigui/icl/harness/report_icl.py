#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""report_icl.py <tasks.jsonl> <zero arm folder> <icl4 arm folder> [--out report.json]

The ICL results table: the zero-shot arm and the ICL-4 arm compared on the same tasks. It reads
only recorded files of the two arms (written by run_online.sh) and calls no model.

Inputs per arm folder (runs/<tag>/<arm>/):
  merged/results.jsonl   one outcome per task
  merged/ask_tasks.json  every task with its open items and the messages the policy sent
  ask_quality.json       phase-2 grading (question or statement, open items targeted)
  shard<i>/scores.json, shard<i>/policy_calls.jsonl   the shard's task and its policy calls
  run_manifest.json      the served policy name (or pass --served-model)

Valid outcome. A task has a valid outcome in an arm when the arm has a result for it and the
rollout was not set aside: not env_blocked (the start page could not be acted on), not
policy_unreachable (the policy server could not be reached; harness/outcomes.py) and not a
harness_error. env_unreachable stays in (as a failure), as in the online success views.

Task success. The verdict on the result row. Rows that carry a judge v2 verdict
(judge_version judge_v2_*) are read as v2: success true, false, or null = unresolved. Null is
never read as a failure. Rows without judge_version are judge v1. The two arms must use one judge
version; a mix of v1 and v2 rows is refused, never pooled. A v1 row with a judge_error has no
verdict and is counted as unresolved too.

Pairs. For each success view, the pairs are the tasks of that view with a valid outcome and a
resolved verdict in both arms. Success is reported on the pairs only, for all tasks and for
Ambiguous and Clear apart. Unresolved verdicts are excluded and counted per arm and per view.

Success views (as in the online part, harness/report_views.py):
  harness    every task, except rows the harness sets aside and rows flagged
             stops_before_commitment by the harness
  readme     harness, minus tasks flagged "Stops before the commitment the recording completed"
  no_flags   harness, restricted to tasks whose flags list is empty
  action_gt  harness, restricted to tasks with action ground truth (not Android-origin)

Question measures (online definitions, phase-2 grading; statements never count as questions),
over the asking pairs: tasks with a valid outcome in both arms and phase-2 grading in both arms.
  question precision  questions that targeted a not-yet-asked open item / questions
                      (null, "not measured", for an arm that asked nothing)
  recall              tasks with open items where at least one question targeted one /
                      tasks with open items

Prompt tokens. usage.prompt_tokens of every policy call (requests to the served policy name) in
the shards of the valid pairs (tasks with a valid outcome in both arms), as the mean per call.
In the ICL-4 arm this includes the demonstrations. Calls without a usage count are counted, not
guessed.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outcomes import SET_ASIDE, ending  # noqa: E402

STOPS = "Stops before the commitment the recording completed"
ANDROID = "Ask-only: no action ground truth, the recording is Android"
VIEWS = ("harness", "readme", "no_flags", "action_gt")
ARMS = ("zero", "icl4")
CUTS = ("all", "Ambiguous", "Clear")


def load_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def load_jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def by_id(rows, key, what):
    out = {}
    for r in rows:
        if r[key] in out:
            raise ValueError(f"duplicate {key} {r[key]!r} in {what}")
        out[r[key]] = r
    return out


def in_view(row, task, view):
    flags = task.get("flags") or []
    if row.get("stops_before_commitment"):
        return False
    if view == "harness":
        return True
    if view == "readme":
        return STOPS not in flags
    if view == "no_flags":
        return not flags
    if view == "action_gt":
        return task.get("has_action_ground_truth") is True and ANDROID not in flags
    raise ValueError(view)


def is_v2(row):
    return str(row.get("judge_version") or "").startswith("judge_v2")


def state(row):
    """('missing'|<set-aside ending>|'harness_error'|'unresolved'|'resolved', verdict)."""
    if row is None:
        return "missing", None
    end = ending(row)
    if end in SET_ASIDE or end == "harness_error":
        return end, None
    success = row.get("success")
    if success is None or (not is_v2(row) and row.get("judge_error")):
        return "unresolved", None
    return "resolved", bool(success)


def policy_tokens(arm_dir: Path, served_model: str):
    """task id -> (prompt tokens of each policy call with a usage count, calls without one)."""
    out = {}
    for shard in sorted(arm_dir.glob("shard[0-9]*")):
        rows = (load_json(shard / "scores.json", {}) or {}).get("results") or []
        if len(rows) != 1:
            continue  # one task per shard; anything else is not attributed
        known, unknown = [], 0
        for call in load_jsonl(shard / "policy_calls.jsonl"):
            if (call.get("request") or {}).get("model") != served_model:
                continue
            usage = ((call.get("response") or {}).get("usage") or {}).get("prompt_tokens")
            if call.get("status") == "ok" and isinstance(usage, int):
                known.append(usage)
            else:
                unknown += 1
        out[rows[0]["task_id"]] = (known, unknown)
    return out


def load_arm(arm_dir: Path, served_model: str | None):
    manifest = load_json(arm_dir / "run_manifest.json", {}) or {}
    model = served_model or manifest.get("policy")
    if not model:
        raise ValueError(f"no served policy name for {arm_dir}: pass --served-model")
    results = by_id(load_jsonl(arm_dir / "merged/results.jsonl"), "task_id", f"{arm_dir}/merged/results.jsonl")
    aq = load_json(arm_dir / "ask_quality.json", {}) or {}
    graded = aq.get("per_task", {}) if aq.get("applicable") is True else {}
    asked = {t["task_id"]: t for t in (load_json(arm_dir / "merged/ask_tasks.json", []) or [])}
    return {"results": results, "graded": graded, "asked": asked, "tokens": policy_tokens(arm_dir, model),
            "served_model": model}


def rate(k, n):
    return {"k": k, "n": n, "rate": (k / n if n else None)}


def build(tasks: dict, arms: dict) -> dict:
    states = {a: {tid: state(arms[a]["results"].get(tid)) for tid in tasks} for a in ARMS}
    valid = {a: {tid for tid, (s, _) in states[a].items() if s in ("resolved", "unresolved")} for a in ARMS}
    versions = {a: sorted({("judge_v2" if is_v2(arms[a]["results"][t]) else "judge_v1") for t in valid[a]}) for a in ARMS}
    used = sorted(set(versions["zero"]) | set(versions["icl4"]))
    if len(used) > 1:
        raise ValueError(f"the arms mix judge versions {versions}; re-grade both arms with one version")
    report = {"judge_version": used[0] if used else None, "tasks": len(tasks),
              "arms": {a: {"results": sum(t in arms[a]["results"] for t in tasks),
                           "states": dict(collections.Counter(s for s, _ in states[a].values())),
                           "endings": dict(collections.Counter(ending(r) for t, r in arms[a]["results"].items() if t in tasks)),
                           "served_model": arms[a]["served_model"]} for a in ARMS},
              "success": {}}
    for view in VIEWS:
        report["success"][view] = {}
        for cut in CUTS:
            ids = [t for t in tasks if cut in ("all", tasks[t].get("ambiguity"))]
            in_v = {a: [t for t in ids if in_view(arms[a]["results"].get(t) or {}, tasks[t], view)] for a in ARMS}
            pairs = [t for t in in_v["zero"] if t in in_v["icl4"]
                     and states["zero"][t][0] == "resolved" and states["icl4"][t][0] == "resolved"]
            entry = {"pairs": len(pairs)}
            for a in ARMS:
                k = sum(states[a][t][1] for t in pairs)
                entry[a] = {**rate(k, len(pairs)),
                            "unresolved": sum(states[a][t][0] == "unresolved" for t in in_v[a]),
                            "not_valid": dict(collections.Counter(states[a][t][0] for t in in_v[a]
                                                                  if states[a][t][0] not in ("resolved", "unresolved")))}
            z, i = entry["zero"]["rate"], entry["icl4"]["rate"]
            entry["icl4_minus_zero"] = (i - z) if z is not None and i is not None else None
            report["success"][view][cut] = entry
    valid_pairs = [t for t in tasks if t in valid["zero"] and t in valid["icl4"]]
    ask_pairs = [t for t in valid_pairs if t in arms["zero"]["graded"] and t in arms["icl4"]["graded"]]
    report["asking"] = {"pairs": len(ask_pairs), "valid_pairs_without_phase2": len(valid_pairs) - len(ask_pairs)}
    report["prompt_tokens"] = {"pairs": len(valid_pairs)}
    for a in ARMS:
        g, asked = arms[a]["graded"], arms[a]["asked"]
        questions = sum(g[t]["n_ask"] for t in ask_pairs)
        on_target = sum(g[t]["n_rel"] for t in ask_pairs)
        blockers = [t for t in ask_pairs if (asked.get(t) or {}).get("parameters")]
        found = sum(g[t]["n_rel"] > 0 for t in blockers)
        report["asking"][a] = {"questions": questions, "on_target_questions": on_target,
                               "question_precision": (on_target / questions if questions else None),
                               "tasks_with_open_items": len(blockers), "tasks_with_open_items_found": found,
                               "recall": (found / len(blockers) if blockers else None)}
        known = [x for t in valid_pairs for x in arms[a]["tokens"].get(t, ([], 0))[0]]
        unknown = sum(arms[a]["tokens"].get(t, ([], 0))[1] for t in valid_pairs)
        report["prompt_tokens"][a] = {"policy_calls_with_usage": len(known), "policy_calls_without_usage": unknown,
                                      "mean_per_call": (sum(known) / len(known) if known else None),
                                      "pairs_without_calls": sum(t not in arms[a]["tokens"] for t in valid_pairs)}
    zt, it = report["prompt_tokens"]["zero"]["mean_per_call"], report["prompt_tokens"]["icl4"]["mean_per_call"]
    report["prompt_tokens"]["icl4_minus_zero"] = (it - zt) if zt is not None and it is not None else None
    return report


def render(report: dict) -> str:
    def pct(x):
        return "not measured" if x is None or x.get("rate") is None else f"{x['k']}/{x['n']} ({100 * x['rate']:.1f}%)"

    def num(v, digits=3):
        return "not measured" if v is None else f"{v:.{digits}f}"

    def diff(v, scale=100.0, unit=" pts"):
        return "not measured" if v is None else f"{scale * v:+.1f}{unit}"
    lines = [f"Judge: {report['judge_version'] or 'no verdicts'}. Tasks in the task file: {report['tasks']}.", "",
             "| view | tasks | pairs | zero-shot success | ICL-4 success | ICL-4 minus zero-shot | unresolved (zero / ICL-4) |",
             "|---|---|---|---|---|---|---|"]
    for view in VIEWS:
        for cut in CUTS:
            e = report["success"][view][cut]
            lines.append(f"| {view} | {cut} | {e['pairs']} | {pct(e['zero'])} | {pct(e['icl4'])} | {diff(e['icl4_minus_zero'])} | "
                         f"{e['zero']['unresolved']} / {e['icl4']['unresolved']} |")
    a, t = report["asking"], report["prompt_tokens"]
    lines += ["", "| measure | zero-shot | ICL-4 | pairs |", "|---|---|---|---|",
              f"| question precision | {num(a['zero']['question_precision'])} | {num(a['icl4']['question_precision'])} | {a['pairs']} |",
              f"| recall on tasks that need a question | {num(a['zero']['recall'])} | {num(a['icl4']['recall'])} | {a['pairs']} |",
              f"| prompt tokens per policy call | {num(t['zero']['mean_per_call'], 0)} | {num(t['icl4']['mean_per_call'], 0)} | {t['pairs']} |",
              "", "Outcome states per arm (resolved / unresolved are valid outcomes):"]
    for arm in ARMS:
        lines.append(f"- {arm}: {report['arms'][arm]['results']} results; {report['arms'][arm]['states']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", type=Path)
    ap.add_argument("zero", type=Path)
    ap.add_argument("icl4", type=Path)
    ap.add_argument("--served-model", help="served policy name (default: from each arm's run_manifest.json)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    tasks = by_id([dict(r, _id=str(r.get("sample_id") or r.get("task_id"))) for r in load_jsonl(args.tasks)], "_id", str(args.tasks))
    arms = {"zero": load_arm(args.zero, args.served_model), "icl4": load_arm(args.icl4, args.served_model)}
    for a in ARMS:
        extra = set(arms[a]["results"]) - set(tasks)
        if extra:
            raise ValueError(f"{a} has results for tasks not in the task file: {sorted(extra)[:5]}")
    report = build(tasks, arms)
    print(render(report), end="")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
