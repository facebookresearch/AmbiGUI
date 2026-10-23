#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""merge_shards.py <policy_dir> <tasks.jsonl>

Pool a sharded policy run (shard*/scores.json, shard*/ask_tasks.json): ask_tasks
concatenated for one phase-2 pass (docs/online.md), success rates RECOMPUTED from the
pooled result rows (never averaged across shard summaries). Writes <policy_dir>/merged/{ask_tasks.json,results.jsonl,summary.json}.

Two success views are reported, because the harness's own set-aside of rows flagged
"Stops before the commitment the recording completed" never fires on this benchmark
(training/cascp_common.task_rows_from_store does not carry `flags`, so
`stops_before_commitment` is always False):
  * as_harness  -- exactly what the scorer would compute (env_blocked and policy_unreachable excluded)
  * readme      -- additionally excludes the benchmark's stops-before-commitment rows
                   (the task file's `flags`), as the benchmark's usage notes ask

Unresolved verdicts (judge v2, `success: null`) are not failures: they are left out of
every success numerator and denominator and counted next to each success rate
("unresolved"), as in the shard summaries. These fields appear when the run was judged
by v2 or any verdict is null; a v1 run's summary is exactly as before.
"""

import collections
import glob
import json
import os
import sys

STOPS = "Stops before the commitment the recording completed"
LABELLED_CLEAR_WITH_ITEMS = "Labelled Clear but carries open items"

pdir, tasks_path = sys.argv[1], sys.argv[2]
rows = {json.loads(l)["sample_id"]: json.loads(l) for l in open(tasks_path) if l.strip()}
out_dir = os.path.join(pdir, "merged")
os.makedirs(out_dir, exist_ok=True)

results, ask_tasks, endings, faults, shard_status = [], [], collections.Counter(), collections.Counter(), {}
unanswerable, unanswerable_ids, unanswerable_seen = 0, [], False
repairs = collections.Counter()
tool_calls = clean_calls = 0
for sd in sorted(glob.glob(os.path.join(pdir, "shard*"))):
    name = os.path.basename(sd)
    sp = os.path.join(sd, "scores.json")
    st = os.path.join(sd, "status.json")
    shard_status[name] = json.load(open(st)) if os.path.exists(st) else {"status": "missing"}
    if not os.path.exists(sp):
        continue
    s = json.load(open(sp))
    results += s.get("results", [])
    if "unanswerable_no_curated_intent" in s:
        unanswerable_seen = True
        unanswerable += s.get("unanswerable_no_curated_intent") or 0
        unanswerable_ids += s.get("unanswerable_no_curated_intent_task_ids") or []
    hr = s.get("harness_report") or {}
    endings.update(hr.get("endings") or {})
    faults.update(hr.get("faults") or {})
    repairs.update(hr.get("repairs_by_kind") or {})
    tool_calls += hr.get("tool_calls") or 0
    clean_calls += hr.get("clean_calls") or 0
    for ap in glob.glob(os.path.join(sd, "ask_tasks*.json")):
        ask_tasks += json.load(open(ap))

ids = [r["task_id"] for r in results]
dupes = [k for k, v in collections.Counter(ids).items() if v > 1]
missing = sorted(set(rows) - set(ids))


def rate(xs):
    n = len(xs)
    return {"n": n, "k": sum(xs), "rate": (sum(xs) / n if n else None)}


#: Judge v2 run, or a null verdict anywhere: report "unresolved" beside each success rate.
TRACK_UNRESOLVED = any(r.get("success") is None or r.get("judge_version") not in (None, "judge_v1") for r in results)


def success_rate(rs):
    """Success over resolved verdicts; a null (unresolved) verdict is not a failure."""
    out = rate([bool(r["success"]) for r in rs if r.get("success") is not None])
    if TRACK_UNRESOLVED:
        out["unresolved"] = sum(r.get("success") is None for r in rs)
    return out


def strat(rs):
    asked = [r for r in rs if r["n_ask"] > 0]
    acted = [r for r in rs if r["n_ask"] == 0]
    return {
        "n": len(rs),
        "success": success_rate(rs),
        "ask_rate": rate([r["n_ask"] > 0 for r in rs]),
        "act_rate": rate([r["n_ask"] == 0 for r in rs]),
        "success_if_asked": success_rate(asked),
        "success_if_acted": success_rate(acted),
    }


def flags(r):
    return (rows.get(r["task_id"]) or {}).get("flags") or []


blocked = [r for r in results if r.get("end_reason") == "env_blocked"]
# ICL copy: a rollout whose policy server could not be reached is set aside like env_blocked.
outage = [r for r in results if r.get("end_reason") == "policy_unreachable"]
SET_ASIDE = ("env_blocked", "policy_unreachable")
harness_scored = [r for r in results if r.get("end_reason") not in SET_ASIDE and not r.get("stops_before_commitment")]
readme_scored = [r for r in harness_scored if STOPS not in flags(r)]
views = {}
for name, rs in (("as_harness", harness_scored), ("readme", readme_scored)):
    views[name] = {
        "ambiguous": strat([r for r in rs if r["ambiguity"] == "Ambiguous"]),
        "clear": strat([r for r in rs if r["ambiguity"] == "Clear"]),
    }
# Ask-arm view per the README: Ambiguous rows with open items vs Clear rows (false alarm),
# not set aside for blocked start pages only (asking is measurable on partial rows).
ask_scored = [r for r in results if r.get("end_reason") not in SET_ASIDE]
ask_arm = [r for r in ask_scored if r["ambiguity"] == "Ambiguous" and (rows.get(r["task_id"]) or {}).get("open_items")]
control = [r for r in ask_scored if r["ambiguity"] == "Clear"]
ask_view = {
    "asked_when_it_should (Ambiguous with open items)": rate([r["n_ask"] > 0 for r in ask_arm]),
    "false_alarm (Clear)": rate([r["n_ask"] > 0 for r in control]),
    "false_alarm (Clear, excluding 'Labelled Clear but carries open items')": rate(
        [r["n_ask"] > 0 for r in control if LABELLED_CLEAR_WITH_ITEMS not in flags(r)]
    ),
    "note": "n_ask counts every send_msg_to_user, statements included; phase-2 ask_quality separates them",
}
summary = {
    "n_expected": len(rows),
    "n_results": len(results),
    "missing_task_ids": missing,
    "duplicate_task_ids": dupes,
    "set_aside_env_blocked": len(blocked),
    "env_blocked_task_ids": [r["task_id"] for r in blocked],
    "set_aside_policy_unreachable": len(outage),
    "policy_unreachable_task_ids": [r["task_id"] for r in outage],
    "readme_stops_before_commitment_excluded": len(harness_scored) - len(readme_scored),
    "success_views": views,
    "ask_view": ask_view,
    "endings": dict(endings),
    "faults": dict(faults),
    "repairs_by_kind": dict(repairs),
    "strict_call_rate": (clean_calls / tool_calls if tool_calls else None),
    "judge_errors": sum(1 for r in results if r.get("judge_error")),
    "shard_status": shard_status,
}
if TRACK_UNRESOLVED:
    summary["judge_version"] = sorted({str(r.get("judge_version")) for r in results})
    summary["unresolved"] = sum(r.get("success") is None for r in results)
if unanswerable_seen:
    # Simulated-user replies on Ambiguous rows with no curated open item (docs/online.md).
    summary["unanswerable_no_curated_intent"] = unanswerable
    summary["unanswerable_no_curated_intent_task_ids"] = sorted(set(unanswerable_ids))
# Judge health over the pooled run. The per-shard guard needs 10 scored tasks, so it
# never runs with one task per shard; the same rule is applied here and printed
# (merge.log). Unresolved v2 verdicts are not judge exceptions and are not counted.
_scored = [r for r in results if r.get("success") is not None]
_errors = sum(1 for r in _scored if r.get("judge_error"))
if len(_scored) >= 10 and (_errors / len(_scored) > 0.05 or (_errors and not any(r.get("success") is True for r in _scored))):
    print(f"[merge] WARNING judge health: {_errors}/{len(_scored)} scored tasks carry a judge error; "
          "the success rates may measure the judge, not the policy")
json.dump(ask_tasks, open(os.path.join(out_dir, "ask_tasks.json"), "w"))
with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
    for r in results:
        f.write(json.dumps(r, default=str) + "\n")
json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in summary.items() if k not in ("shard_status", "env_blocked_task_ids")}, indent=1, default=str)[:4000])
