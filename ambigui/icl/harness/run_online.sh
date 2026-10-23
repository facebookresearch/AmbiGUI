#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# One arm of the matched zero-shot vs ICL-4 comparison on this machine: dependency checks, then
# the frozen policy over every task of the task file (phase 1: live browser rollouts and task
# success), then the question grading (phase 2: ask quality).
#   bash ambigui/icl/harness/run_online.sh <tag> <zero|icl4>
# Run both arms with the same tag, task file, simulated user and judge. The policy (served name,
# Hugging Face repo and revision, tensor parallel) comes from ICL_CONFIG; it is served here by vLLM
# unless POLICY_URL points at a server that already serves it under that name.
#
# Required environment:
#   AMBIGUI_TASKS        benchmark task file (JSONL, one row per task, with open_items and max_steps)
#   ANSWERER_ENDPOINT    simulated user: OpenAI-compatible /v1 URL (serving the config's answerer_model)
#   JUDGE_URL            task judge, describer and ask judge: /v1 URL (serving the config's judge_model)
#   ICL_CONDITION_DIR    icl4 arm only: folder written by python -m ambigui.icl.prepare_condition
# Optional: ICL_CONFIG (default configs/icl/q38_27b_matched84.json), POLICY_URL, OUT_ROOT (default
# ./runs), PAR (shards at once; default all; the reported runs used 1), GATE_CASES, POLICY_PORT,
# VLLM_PY, WEIGHTS_DIR, HF_TOKEN, ICL_VLLM_SITE_PACKAGES, the API keys of
# ambigui/icl/src/evaluation/endpoint_auth.py, BRIGHTDATA_PROXY_SERVER (+ _USERNAME, _PASSWORD).
# Exit codes: 2 usage or ICL-4 condition check, 5 judge endpoint, 6 gate check, 7 policy serve,
# 8 dependency preflight, 9 merge.
set -uo pipefail
if [ $# -lt 2 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "usage: bash ambigui/icl/harness/run_online.sh <tag> <zero|icl4>   (environment: see the header of this file and docs/icl.md)"
  [ $# -lt 2 ] && [ "${1:-}" != "-h" ] && [ "${1:-}" != "--help" ] && exit 2
  exit 0
fi
TAG="$1"; ARM="$2"
case "$ARM" in zero|icl4) ;; *) echo "arm must be zero or icl4" >&2; exit 2;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"
ICL="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
SRC="$ICL/src"
PY="${PYTHON_EXEC:-python3}"
# Absolute: the shards run from src/.
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")";; esac
CFG="${ICL_CONFIG:-$REPO_ROOT/configs/icl/q38_27b_matched84.json}"
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
mkdir -p "${OUT_ROOT:-$PWD/runs}/$TAG"
ROOT="$(cd "${OUT_ROOT:-$PWD/runs}/$TAG" && pwd)"   # absolute: the shards run from src/
exec > >(tee -a "$ROOT/job_$ARM.log") 2>&1
echo "[job] host=$(hostname) tag=$TAG arm=$ARM $(date -u +%FT%TZ)"

cfg() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$CFG" "$1"; }
POLICY="$(cfg served_model)"; HFREPO="$(cfg model)"; HFREV="$(cfg model_revision)"; TP="$(cfg policy_tp)"
ANSWERER_MODEL="$(cfg answerer_model)"; JUDGE_MODEL="$(cfg judge_model)"
: "${AMBIGUI_TASKS:?set AMBIGUI_TASKS to the benchmark task file}"
: "${ANSWERER_ENDPOINT:?set ANSWERER_ENDPOINT to the simulated user /v1 URL}"
DESCRIBER_MODEL="$JUDGE_MODEL"
ASK_JUDGE_MODEL="$JUDGE_MODEL"
TASKS="$(cd "$(dirname "$AMBIGUI_TASKS")" && pwd)/$(basename "$AMBIGUI_TASKS")"
# Every task of the file, one task per shard (the reported runs' setting). Blank lines are not tasks.
LIMIT="$(grep -c '[^[:space:]]' "$TASKS")"
MAX_TOKENS=8192
POLICY_PORT="${POLICY_PORT:-8400}"
# The reported runs recorded every policy request and response losslessly, in both arms.
export RECORD_POLICY_CALLS=1 POLICY_MAX_CONTEXT=65536
# The policy server of the reported runs: BF16, eager mode, at most 8 sequences (policy_serve.sh
# adds tensor parallel, 65536 context, 4 images, utilisation 0.90, seed 1234, the checkpoint's template).
export POLICY_SERVE_EXTRA="${POLICY_SERVE_EXTRA:---dtype bfloat16 --enforce-eager --max-num-seqs 8}"

export PYTHONPATH="$HERE:$SRC:$REPO_ROOT"
export DESCRIBER_MODEL ASK_JUDGE_MODEL
export ANSWERER_ENDPOINT ANSWERER_MODEL JUDGE_MODEL

if [ "$ARM" = icl4 ]; then
  : "${ICL_CONDITION_DIR:?set ICL_CONDITION_DIR (python -m ambigui.icl.prepare_condition)}"
  ICL_CONDITION_DIR="$(cd "$ICL_CONDITION_DIR" && pwd)" || exit 2
  # The headroom guard reads four source files of the installed vLLM 0.23.0 (never imports it).
  if [ -z "${ICL_VLLM_SITE_PACKAGES:-}" ]; then
    ICL_VLLM_SITE_PACKAGES="$("${VLLM_PY:-python3}" -c 'import importlib.util,os; s=importlib.util.find_spec("vllm"); print(os.path.dirname(list(s.submodule_search_locations)[0]))' 2>/dev/null)"
  fi
  [ -n "$ICL_VLLM_SITE_PACKAGES" ] || { echo "[job] FATAL icl4 needs vLLM 0.23.0 sources: set ICL_VLLM_SITE_PACKAGES or VLLM_PY"; exit 2; }
  ICL_VLLM_SITE_PACKAGES="$(cd "$ICL_VLLM_SITE_PACKAGES" && pwd)" || { echo "[job] FATAL ICL_VLLM_SITE_PACKAGES is not a folder"; exit 2; }
  export ICL_VLLM_SITE_PACKAGES
  # The demonstrations must have been checked against this task file, and none may overlap it.
  "$PY" -m ambigui.icl.condition "$ICL_CONDITION_DIR" "$TASKS" \
    || { echo "[job] FATAL the ICL-4 condition does not pass its check against $TASKS"; exit 2; }
fi

# ---- browser check: the local Chromium reaches a public page (reported, not fatal)
"$PY" - <<'PYEOF'
import time
from playwright.sync_api import sync_playwright
try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page(); t = time.time()
        r = pg.goto("https://www.wikipedia.org", timeout=45000)
        print(f"[job] browser check: chromium {b.version} status={r.status if r else None} title={pg.title()!r} {time.time()-t:.1f}s")
        b.close()
except Exception as e:
    print(f"[job] browser check FAILED: {type(e).__name__}: {str(e)[:1500]}")
PYEOF

# ---- judge: an existing endpoint that answers a real request
: "${JUDGE_URL:?set JUDGE_URL to the task judge /v1 URL}"
"$PY" "$HERE/judge_probe.py" "$JUDGE_URL" "$JUDGE_MODEL" || { echo "[job] FATAL judge endpoint $JUDGE_URL"; exit 5; }
DESCRIBER_URL="${DESCRIBER_URL:-$JUDGE_URL}"
ASK_JUDGE_URL="${ASK_JUDGE_URL:-$JUDGE_URL}"
export JUDGE_URL DESCRIBER_URL ASK_JUDGE_URL

# ---- gate check (optional; the reported runs used one): recorded messages through the
# simulated user and the ask judge, with expected outcomes. A failed gate runs no rollouts.
GATE_RC=skipped
if [ -n "${GATE_CASES:-}" ]; then
  GATE_CASES="$(cd "$(dirname "$GATE_CASES")" && pwd)/$(basename "$GATE_CASES")"
  "$PY" "$HERE/gate_check.py" "$GATE_CASES" "$ROOT/gate_check_$ARM.json"
  GATE_RC=$?
  echo "[job] gate check rc=$GATE_RC"
  [ "$GATE_RC" = 0 ] || { echo "[job] FATAL gate check failed; not running rollouts"; exit 6; }
fi

# ---- manifest. Values reach Python through the environment, never pasted into its source;
# user names and passwords in URLs are not recorded.
ARM="$ARM" CFG="$CFG" POLICY="$POLICY" HFREPO="$HFREPO" HFREV="$HFREV" TASKS="$TASKS" LIMIT="$LIMIT" \
MAX_TOKENS="$MAX_TOKENS" GATE_RC="$GATE_RC" ICL_CONDITION_DIR="${ICL_CONDITION_DIR:-}" \
  "$PY" - "$ROOT/run_manifest_$ARM.json" "$REPO_ROOT" <<'PYEOF'
import hashlib, json, os, platform, subprocess, sys, urllib.parse
from importlib import metadata
E = os.environ
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
def ver(name):
    try: return metadata.version(name)
    except metadata.PackageNotFoundError: return None
def url(value):
    if not value: return value
    u = urllib.parse.urlsplit(value)
    if not u.hostname: return value
    return urllib.parse.urlunsplit(u._replace(netloc=u.hostname + (f":{u.port}" if u.port else "")))
try:
    commit = subprocess.run(["git", "-C", sys.argv[2], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
except OSError:
    commit = None
condition = None
if E["ARM"] == "icl4":
    c = json.load(open(os.path.join(E["ICL_CONDITION_DIR"], "condition.json")))
    condition = {k: c.get(k) for k in ("condition_fingerprint", "demo_fingerprint", "canonical_message_digest", "prompt_prefix_sha256",
                                        "system_sha256", "processor_fingerprint", "demo_source_rule", "demonstration_sources")}
m = {
 "arm": E["ARM"], "condition": condition, "config": json.load(open(E["CFG"])), "config_sha256": sha(E["CFG"]),
 "policy": E["POLICY"], "policy_hf_repo": E["HFREPO"], "policy_hf_revision": E["HFREV"],
 "policy_url": url(E.get("POLICY_URL")) or "local vLLM (harness/policy_serve.sh)",
 "policy_serve_extra": E["POLICY_SERVE_EXTRA"],
 "answerer_model": E["ANSWERER_MODEL"], "answerer_endpoint": url(E["ANSWERER_ENDPOINT"]),
 "judge_model": E["JUDGE_MODEL"], "judge_kind": "lookup", "judge_url": url(E["JUDGE_URL"]),
 "describer_model": E["DESCRIBER_MODEL"], "describer_url": url(E["DESCRIBER_URL"]),
 "ask_judge_model": E["ASK_JUDGE_MODEL"], "ask_judge_kind": "set", "ask_judge_url": url(E["ASK_JUDGE_URL"]),
 "tasks_file": E["TASKS"], "tasks_sha256": sha(E["TASKS"]), "limit": int(E["LIMIT"]),
 "step_cap": "per row: the row's max_steps",
 "shards": int(E["LIMIT"]), "parallel_shards": int(E.get("PAR") or E["LIMIT"]),
 "shard_process_timeout_s": int(E.get("POLICY_TIMEOUT", "3600")),
 "start_page_wait": "domcontentloaded (60000 ms) then load best effort (10000 ms)",
 "max_tokens": int(E["MAX_TOKENS"]), "image_budget": 4, "episode_deadline_s": 600.0,
 "parser": "auto", "concurrency_per_shard": 1, "temperature": 0.0, "seed": 0,
 "observation_mode": "som_state", "repo_commit": commit,
 "transport": "local chromium, direct; fallback proxy " + ("configured" if E.get("BRIGHTDATA_PROXY_SERVER") else "not configured"),
 "policy_max_context": 65536, "policy_call_recording": True, "gate_check_rc": E["GATE_RC"],
 "packages": {p: ver(p) for p in ("browsergym-core", "playwright", "openai", "pillow", "numpy", "transformers", "tokenizers",
                                  "jinja2", "pydantic", "vllm")},
 "python": platform.python_version(), "host": platform.node(), "platform": platform.platform(),
}
json.dump(m, open(sys.argv[1], "w"), indent=1); print(json.dumps(m, indent=1))
PYEOF

# Resume: a shard finished in an earlier attempt of this run (same OUT_ROOT, tag and arm) is
# kept, not rerun. A shard that did not finish, or that ended env_unreachable or
# policy_unreachable (the policy server could not be reached), is run again (harness/outcomes.py).
try_resume() { # <out dir> <shard>
  ! "$PY" "$HERE/outcomes.py" rerun "$1/shard$2"
}

# Task judge version (docs/icl.md, "Task judge version"). Unset JUDGE_VERSION keeps the v1 task judge, as in the
# reported runs. JUDGE_VERSION=judge_v2_with_intent/2.0.3 needs all three files below.
JUDGE_V2_ARGS=()
if [ -n "${JUDGE_VERSION:-}" ]; then
  : "${CURATED_SIDECAR:?JUDGE_VERSION needs CURATED_SIDECAR (the curated task rows)}"
  : "${JUDGE_CONFIG:?JUDGE_VERSION needs JUDGE_CONFIG (start from configs/icl/judge_v2.json)}"
  : "${JUDGE_JOURNAL_ROOT:?JUDGE_VERSION needs JUDGE_JOURNAL_ROOT (a new, empty folder)}"
  abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
  JUDGE_V2_ARGS=(--judge-version "$JUDGE_VERSION" --curated-sidecar "$(abspath "$CURATED_SIDECAR")"
                 --judge-config "$(abspath "$JUDGE_CONFIG")" --judge-journal-root "$(abspath "$JUDGE_JOURNAL_ROOT")")
fi

OUT="$ROOT/$ARM"; mkdir -p "$OUT"
{
  echo "[$ARM] start $(date -u +%T)"
  N="$LIMIT"; PAR="${PAR:-$N}"; P_URL="${POLICY_URL:-}"; LIST=""
  for i in $(seq 1 "$N"); do
    mkdir -p "$OUT/shard$i"
    if try_resume "$OUT" "$i"; then echo "[$ARM/s$i] kept (finished in an earlier attempt)"; else LIST="$LIST $i"; fi
  done
  if [ -z "${LIST// /}" ]; then
    echo "[$ARM] every shard finished in an earlier attempt"
  elif [ -z "$P_URL" ]; then
    bash "$HERE/policy_serve.sh" "$POLICY" "${POLICY_CHECKPOINT:-$HFREPO}" "$HFREV" "$TP" "$POLICY_PORT" "$OUT/serve"
    SRV_RC=$?
    if [ "$SRV_RC" != 0 ]; then
      echo "[$ARM] serve failed (rc=$SRV_RC) -> no rollouts (not a model score)"
      tail -n 400 "$OUT/serve/vllm_policy_$POLICY.log" > "$OUT/serve_tail.log" 2>/dev/null
      echo "{\"status\":\"policy_not_servable\",\"stage\":\"serve\",\"rc\":$SRV_RC}" > "$OUT/status.json"; exit 7
    fi
    P_URL="http://127.0.0.1:$POLICY_PORT/v1"
  fi
  stop_policy_serve() { # only the server this script started: its recorded PID and that PID's children
    [ -f "$OUT/serve/policy_vllm.pid" ] || return 0
    local SPID; SPID="$(cat "$OUT/serve/policy_vllm.pid")"
    pkill -TERM -P "$SPID" 2>/dev/null; kill "$SPID" 2>/dev/null; sleep 10
    pkill -KILL -P "$SPID" 2>/dev/null; kill -9 "$SPID" 2>/dev/null; sleep 2
    tail -n 400 "$OUT/serve/vllm_policy_$POLICY.log" > "$OUT/serve_tail.log" 2>/dev/null
    rm -f "$OUT/serve/policy_vllm.pid"
  }
  PF_RC=0; POLICY_OK=1
  if [ -n "${LIST// /}" ]; then
    # The shaped probe: the harness's own system prompt, tool schema, tool_choice and a
    # screenshot against the policy server (in the icl4 arm with the demonstrations and the guard).
    if [ "$ARM" = icl4 ]; then
      CASCP_SHOT_DIR="$OUT/preflight_recording/frames" POLICY_URL="$P_URL" POLICIES="$POLICY" MAX_TOKENS="$MAX_TOKENS" \
        "$PY" "$HERE/preflight_icl.py" "$ICL_CONDITION_DIR" "$TASKS" > "$OUT/preflight.json" 2> "$OUT/preflight.err"
    else
      CASCP_SHOT_DIR="$OUT/preflight_recording/frames" POLICY_URL="$P_URL" POLICIES="$POLICY" MAX_TOKENS="$MAX_TOKENS" \
        "$PY" "$HERE/preflight_roles.py" > "$OUT/preflight.json" 2> "$OUT/preflight.err"
    fi
    PF_RC=$?
    POLICY_OK=$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d["policies"][sys.argv[2]]["ok"]))' \
      "$OUT/preflight.json" "$POLICY" 2>/dev/null || echo 0)
    echo "[$ARM] preflight rc=$PF_RC policy_ok=$POLICY_OK $("$PY" -c 'import json,sys; p=json.load(open(sys.argv[1]))["policies"][sys.argv[2]]; print(json.dumps({k: p.get(k) for k in ("s", "finish", "parsed", "error")})[:300])' "$OUT/preflight.json" "$POLICY" 2>/dev/null)"
  fi
  if [ "$PF_RC" != 0 ]; then echo '{"status":"dependency_preflight_failed"}' > "$OUT/status.json"; cat "$OUT/preflight.json"; stop_policy_serve; exit 8; fi
  if [ "$POLICY_OK" != 1 ]; then
    echo "[$ARM] not servable now -> no rollouts (not a model score)"
    echo '{"status":"policy_not_servable","stage":"request"}' > "$OUT/status.json"; stop_policy_serve; exit 8
  fi
  # One task per shard, at most PAR running at once: an interruption costs only the tasks in
  # flight; every finished shard is kept and skipped on the next attempt.
  for i in $LIST; do
    rm -rf "$OUT/shard$i"; mkdir -p "$OUT/shard$i"  # a shard that is run again starts empty
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
    (
      cd "$SRC" || exit 1
      EVAL_ARGS=(--served-url "$P_URL" --served-model "$POLICY"
        --eval-rollouts "$TASKS"
        --judge-url "$JUDGE_URL" --judge-model "$JUDGE_MODEL" --judge-kind lookup
        --describer-url "$DESCRIBER_URL" --describer-model "$DESCRIBER_MODEL"
        --answerer-endpoint "$ANSWERER_ENDPOINT" --answerer-model "$ANSWERER_MODEL"
        --max-tokens "$MAX_TOKENS" --shard "$i/$N"
        "${JUDGE_V2_ARGS[@]}"
        --out "$OUT/shard$i/scores.json")
      if [ "$ARM" = icl4 ]; then
        RUNNER=("$ICL/run_icl_eval.py" --condition-dir "$ICL_CONDITION_DIR" --icl-output-root "$OUT")
      else
        RUNNER=("$HERE/run_eval.py")
      fi
      CASCP_SHOT_DIR="$OUT/shard$i/frames" POLICY_URL="$P_URL" timeout -k 60 "${POLICY_TIMEOUT:-3600}" "$PY" "${RUNNER[@]}" \
        "${EVAL_ARGS[@]}" > "$OUT/shard$i/run.log" 2>&1
      RC=$?
      echo "{\"status\":\"done\",\"phase1_rc\":$RC,\"probe_retry\":0}" > "$OUT/shard$i/status.json"
      echo "[$ARM/s$i] phase1 rc=$RC rollouts=$(grep -c ' amb=' "$OUT/shard$i/run.log")"
    ) &
    sleep 2
  done
  wait
  stop_policy_serve
  if [ "$ARM" = icl4 ]; then
    "$PY" "$HERE/merge_icl.py" "$OUT" "$TASKS" "$ICL_CONDITION_DIR" > "$OUT/merge.log" 2>&1
  else
    "$PY" "$HERE/merge_shards.py" "$OUT" "$TASKS" > "$OUT/merge.log" 2>&1
  fi
  MERGE_RC=$?
  echo "[$ARM] merge rc=$MERGE_RC"; head -c 4000 "$OUT/merge.log"; echo
  P2_RC=skipped
  if [ -s "$OUT/merged/ask_tasks.json" ]; then
    ( cd "$SRC" && "$PY" "$HERE/run_ask_quality.py" --tasks "$OUT/merged/ask_tasks.json" --out "$OUT/ask_quality.json" --workers 8 )
    P2_RC=$?
    echo "[$ARM] phase2 rc=$P2_RC"
    "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("[ask_quality]", json.dumps({k: d.get(k) for k in ("n_asks", "n_statements", "n_asks_scored", "n_judge_failures", "n_gate_unreadable")}), json.dumps(d.get("report"))[:800])' "$OUT/ask_quality.json" 2>/dev/null
  else
    echo "[$ARM] no merged ask_tasks; phase 2 skipped"
  fi
  echo "{\"status\":\"done\",\"merge_rc\":$MERGE_RC,\"phase2_rc\":\"$P2_RC\"}" > "$OUT/status.json"
  echo "[$ARM] end $(date -u +%T)"
  [ "$MERGE_RC" = 0 ] || exit 9
} 2>&1 | tee "$OUT/policy.log" | sed -u "s/^/[$ARM] /"
ARM_RC=${PIPESTATUS[0]}
cp "$ROOT/run_manifest_$ARM.json" "$OUT/run_manifest.json" 2>/dev/null
echo "[job] done $(date -u +%FT%TZ) rc=$ARM_RC; results in $OUT"
exit "$ARM_RC"
