#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# One online evaluation run on this machine: dependency checks, then every policy over the
# task set (phase 1: live browser rollouts and task success), then the question grading
# (phase 2: ask quality).
#   bash run_online.sh <tag> <policy>[,<policy>...]
# A policy is either an open-weights slug from configs/online/open_policies.tsv (served
# here by vLLM, one at a time) or the name an OpenAI-compatible endpoint at POLICY_URL
# serves (API policies run in parallel). <model>@pN is prompt arm pN of <model>
# (harness/data/prompts/ask_pN.txt). See docs/online.md for every setting.
#
# Required environment:
#   AMBIGUI_TASKS        benchmark task file (JSONL, one row per task, with max_steps)
#   ANSWERER_ENDPOINT    simulated user: OpenAI-compatible /v1 URL
#   ANSWERER_MODEL       simulated user: served model name
#   JUDGE_MODEL          task judge served name (also describer and ask judge by default)
#   JUDGE_URL            task judge /v1 URL (also the describer and the ask judge by default)
#   POLICY_URL           /v1 URL for API policies (not needed for open policies)
set -uo pipefail
if [ $# -lt 2 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "usage: bash run_online.sh <tag> <policy>[,<policy>...]   (environment: see the header of this file and docs/online.md)"
  [ $# -lt 2 ] && [ "${1:-}" != "-h" ] && [ "${1:-}" != "--help" ] && exit 2
  exit 0
fi
TAG="$1"; POLICIES="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
SRC="$HERE/../src"
PY="${PYTHON_EXEC:-python3}"
mkdir -p "${OUT_ROOT:-$PWD/runs}/$TAG"
ROOT="$(cd "${OUT_ROOT:-$PWD/runs}/$TAG" && pwd)"   # absolute: the shards run from src/
exec > >(tee -a "$ROOT/job.log") 2>&1
echo "[job] host=$(hostname) tag=$TAG policies=$POLICIES $(date -u +%FT%TZ)"

: "${AMBIGUI_TASKS:?set AMBIGUI_TASKS to the benchmark task file}"
: "${ANSWERER_ENDPOINT:?set ANSWERER_ENDPOINT to the simulated user /v1 URL}"
: "${ANSWERER_MODEL:?set ANSWERER_MODEL to the simulated user served name}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen3-vl-235b-a22b-instruct}"
DESCRIBER_MODEL="${DESCRIBER_MODEL:-$JUDGE_MODEL}"
ASK_JUDGE_MODEL="${ASK_JUDGE_MODEL:-$JUDGE_MODEL}"
TASKS="$(cd "$(dirname "$AMBIGUI_TASKS")" && pwd)/$(basename "$AMBIGUI_TASKS")"
[ -z "${POLICY_TASKS_DIR:-}" ] || POLICY_TASKS_DIR="$(cd "$POLICY_TASKS_DIR" && pwd)"
[ -z "${GATE_CASES:-}" ] || GATE_CASES="$(cd "$(dirname "$GATE_CASES")" && pwd)/$(basename "$GATE_CASES")"
# Every task of the file, one task per shard (the paper runs' setting).
LIMIT="$(grep -c . "$TASKS")"
MAX_TOKENS=8192
OPEN_TABLE="$REPO_ROOT/configs/online/open_policies.tsv"
POLICY_PORT="${POLICY_PORT:-8400}"

export PYTHONPATH="$HERE:$SRC"
export DESCRIBER_MODEL ASK_JUDGE_MODEL
export ANSWERER_ENDPOINT ANSWERER_MODEL JUDGE_MODEL

# ---- browser check: the local Chromium reaches a public page
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

# ---- task judge version (docs/judge_v2.md). Unset JUDGE_VERSION keeps the v1 task judge, as in
# the recorded runs. JUDGE_VERSION=judge_v2_with_intent/2.0.3 needs the three files below. They
# are checked once here, before any policy server starts, with the paths made absolute.
JUDGE_V2_ARGS=()
if [ -n "${JUDGE_VERSION:-}" ]; then
  : "${CURATED_SIDECAR:?JUDGE_VERSION needs CURATED_SIDECAR (the curated task rows)}"
  : "${JUDGE_CONFIG:?JUDGE_VERSION needs JUDGE_CONFIG (start from configs/online/judge_v2.json)}"
  : "${JUDGE_JOURNAL_ROOT:?JUDGE_VERSION needs JUDGE_JOURNAL_ROOT (a folder for the judge journal)}"
  for f in "$CURATED_SIDECAR" "$JUDGE_CONFIG"; do [ -f "$f" ] || { echo "[job] FATAL judge v2 file not found: $f"; exit 5; }; done
  mkdir -p "$JUDGE_JOURNAL_ROOT" || { echo "[job] FATAL cannot create $JUDGE_JOURNAL_ROOT"; exit 5; }
  CURATED_SIDECAR="$(cd "$(dirname "$CURATED_SIDECAR")" && pwd)/$(basename "$CURATED_SIDECAR")"
  JUDGE_CONFIG="$(cd "$(dirname "$JUDGE_CONFIG")" && pwd)/$(basename "$JUDGE_CONFIG")"
  JUDGE_JOURNAL_ROOT="$(cd "$JUDGE_JOURNAL_ROOT" && pwd)"
  export JUDGE_VERSION CURATED_SIDECAR JUDGE_CONFIG JUDGE_JOURNAL_ROOT
  JUDGE_V2_ARGS=(--judge-version "$JUDGE_VERSION" --curated-sidecar "$CURATED_SIDECAR"
                 --judge-config "$JUDGE_CONFIG" --judge-journal-root "$JUDGE_JOURNAL_ROOT")
  # Judge v2 check, once: version, config, served names and curated sidecar (no model call).
  ( cd "$SRC" || exit 1
    "$PY" -c "import argparse,sys; from evaluation.judge_v2_adapter import add_activation_arguments, activation_from_args
p=argparse.ArgumentParser(); add_activation_arguments(p)
[p.add_argument('--'+f) for f in ('judge-kind','judge-url','judge-model','describer-url','describer-model')]
activation_from_args(p.parse_args(sys.argv[1:])); print('[job] judge v2 config and curated sidecar OK')" \
      "${JUDGE_V2_ARGS[@]}" --judge-kind lookup --judge-url "$JUDGE_URL" --judge-model "$JUDGE_MODEL" \
      --describer-url "$DESCRIBER_URL" --describer-model "$DESCRIBER_MODEL" ) \
    || { echo "[job] FATAL judge v2 check failed; no policy server started"; exit 5; }
fi

# ---- gate check (optional): recorded messages through the answerer and the ask judge
GATE_RC=skipped
if [ -n "${GATE_CASES:-}" ]; then
  "$PY" "$HERE/gate_check.py" "$GATE_CASES" "$ROOT/gate_check.json"
  GATE_RC=$?
  echo "[job] gate check rc=$GATE_RC"
  if [ "$GATE_RC" != 0 ]; then
    echo "[job] FATAL gate check failed; not running rollouts"
    POLICIES=""
  fi
fi

# ---- manifest
"$PY" - "$ROOT/run_manifest.json" "$REPO_ROOT" <<EOF
import hashlib, json, os, platform, subprocess, sys
from importlib import metadata
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
def ver(name):
    try: return metadata.version(name)
    except metadata.PackageNotFoundError: return None
try:
    commit = subprocess.run(["git", "-C", sys.argv[2], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
except OSError:
    commit = None
open_table = "$OPEN_TABLE"
m = {
 "policies": "$POLICIES".split(","), "answerer_model": "$ANSWERER_MODEL", "answerer_endpoint": "$ANSWERER_ENDPOINT",
 "judge_model": "$JUDGE_MODEL", "judge_kind": "lookup", "judge_url": "$JUDGE_URL",
 "describer_model": "$DESCRIBER_MODEL", "describer_url": "$DESCRIBER_URL",
 "ask_judge_model": "$ASK_JUDGE_MODEL", "ask_judge_kind": "set", "ask_judge_url": "$ASK_JUDGE_URL",
 "judge_serve": "external OpenAI-compatible endpoint",
 "tasks_file": "$TASKS", "tasks_sha256": sha("$TASKS"), "limit": int("$LIMIT"),
 "step_cap": "per row: the row's max_steps",
 "shards_per_policy": int("$LIMIT"),
 "parallel_shards_per_policy": int(os.environ.get("PAR", "$LIMIT")),
 "harness_probe_tasks": 0,
 "judge_livecheck_timeout_s": 900.0,
 "start_page_wait": "domcontentloaded (timeout BG_GOTO_TIMEOUT_MS=60000 ms) then load best effort (BG_LOAD_GRACE_MS=10000 ms); BrowserGym default is load within 10 s",
 "max_tokens": int("$MAX_TOKENS"), "image_budget": 4, "episode_deadline_s": 600.0,
 "parser": "auto", "concurrency_per_policy": 1, "temperature": 0.0, "seed": 0,
 "observation_mode": "som_state",
 "repo_commit": commit,
 "transport": "local chromium, direct; fallback proxy " + ("configured" if os.environ.get("BRIGHTDATA_PROXY_SERVER") else "not configured"),
 "policy_max_context": int(os.environ.get("POLICY_MAX_CONTEXT", "65536")),
 "policy_call_recording": os.environ.get("RECORD_POLICY_CALLS") == "1",
 "gate_check_rc": "$GATE_RC",
 "judge_version": os.environ.get("JUDGE_VERSION") or "judge_v1",
 "judge_v2_config_sha256": sha(os.environ["JUDGE_CONFIG"]) if os.environ.get("JUDGE_VERSION") and os.path.isfile(os.environ.get("JUDGE_CONFIG", "")) else None,
 "judge_v2_curated_sidecar_sha256": sha(os.environ["CURATED_SIDECAR"]) if os.environ.get("JUDGE_VERSION") and os.path.isfile(os.environ.get("CURATED_SIDECAR", "")) else None,
 "open_policies": [l.strip() for l in open(open_table) if l.strip() and not l.lstrip().startswith("#") and l.split("|")[0].strip() in [x.split("@")[0] for x in "$POLICIES".split(",")]],
 "ask_prompt_variants": {v: open("$HERE/data/prompts/ask_" + v + ".txt").read().strip() for v in sorted({x.split("@")[1] for x in "$POLICIES".split(",") if "@" in x})},
 "open_policy_serve": "vLLM on this machine's GPUs (harness/policy_serve.sh); checkpoint = Hugging Face repo at the pinned revision; tensor-parallel and max new tokens per turn from the open-policy table (overrides max_tokens above); max-model-len = min(65536, the model's max_position_embeddings); image limit 4 = the image budget; chat template from the checkpoint; no reasoning parser; temperature 0, seed 0",
 "packages": {p: ver(p) for p in ("browsergym-core", "playwright", "openai", "requests", "pillow", "numpy", "vllm", "transformers", "torch")},
 "python": platform.python_version(), "host": platform.node(), "platform": platform.platform(),
}
json.dump(m, open(sys.argv[1], "w"), indent=1); print(json.dumps(m, indent=1))
EOF

open_spec() { # <slug> -> "<hf repo> <revision> <tp> <max new tokens>", empty if not an open policy
  awk -F'|' -v s="$1" '/^[[:space:]]*#/ {next} {for (k=1;k<=NF;k++) gsub(/[[:space:]]/,"",$k)} $1==s {print $2" "$3" "$4" "$5; exit}' "$OPEN_TABLE" 2>/dev/null
}
# Resume: a shard finished in an earlier attempt of this run (same OUT_ROOT and tag) is
# kept, not rerun. A finished shard that ended env_unreachable is not accepted and is run again.
try_resume() { # <out dir> <shard>
  grep -q '"phase1_rc": *0' "$1/shard$2/status.json" 2>/dev/null \
    && ! grep -q '"end_reason": *"env_unreachable"' "$1/shard$2/scores.json" 2>/dev/null
}

# ---- per-policy run (phase 1 + phase 2): API policies in parallel, open policies one at a time
run_policy() {
  local POLICY="$1"; local OUT="$ROOT/$POLICY"; mkdir -p "$OUT"
  {
    echo "[$POLICY] start $(date -u +%T)"
    # A prompt arm is <model>@<variant>: the model is served or called as <model>, and the
    # variant (data/prompts/ask_<variant>.txt) replaces the ask paragraph of its system prompt.
    local BASE="${POLICY%%@*}" VAR=""
    case "$POLICY" in *@*) VAR="${POLICY#*@}"; export ASK_PROMPT_VARIANT="$VAR";; esac
    # A per-policy task subset ($POLICY_TASKS_DIR/<policy>.jsonl) replaces the run's task
    # file for this policy, one task per shard.
    local P_TASKS="$TASKS"
    local N="$LIMIT"
    if [ -n "${POLICY_TASKS_DIR:-}" ] && [ -f "$POLICY_TASKS_DIR/$POLICY.jsonl" ]; then
      P_TASKS="$POLICY_TASKS_DIR/$POLICY.jsonl"; N=$(grep -c . "$P_TASKS")
      echo "[$POLICY] task subset: $N rows from $P_TASKS"
    fi
    local PAR="${PAR:-$N}" i
    local SPEC P_URL="${POLICY_URL:-}" P_MAXTOK="$MAX_TOKENS" SKIP_RESUME=0 LIST HFREPO HFREV TP
    LIST=$(seq 1 "$N")
    SPEC=$(open_spec "$BASE")
    if [ -n "$SPEC" ] && [ -n "$VAR" ]; then
      # Served once for all of its prompt arms, before the arms start (see below).
      read -r HFREPO HFREV TP P_MAXTOK <<< "$SPEC"
      P_URL="http://127.0.0.1:$POLICY_PORT/v1"
    elif [ -n "$SPEC" ]; then
      read -r HFREPO HFREV TP P_MAXTOK <<< "$SPEC"
      # Resume first, so a model whose every shard finished in an earlier attempt is not
      # loaded onto the GPUs again.
      LIST=""
      for i in $(seq 1 "$N"); do
        mkdir -p "$OUT/shard$i"
        if try_resume "$OUT" "$i"; then echo "[$POLICY/s$i] kept (finished in an earlier attempt)"; else LIST="$LIST $i"; fi
      done
      SKIP_RESUME=1
      if [ -z "${LIST// /}" ]; then
        echo "[$POLICY] every shard finished in an earlier attempt; not serving"
      else
        bash "$HERE/policy_serve.sh" "$POLICY" "${POLICY_CHECKPOINT:-$HFREPO}" "$HFREV" "$TP" "$POLICY_PORT" "$OUT/serve"
        local SRV_RC=$?
        if [ "$SRV_RC" != 0 ]; then
          echo "[$POLICY] serve failed (rc=$SRV_RC) -> no rollouts (not a model score)"
          tail -n 400 "$OUT/serve/vllm_policy_$POLICY.log" > "$OUT/serve_tail.log" 2>/dev/null
          echo "{\"status\":\"policy_not_servable\",\"stage\":\"serve\",\"rc\":$SRV_RC}" > "$OUT/status.json"; return
        fi
        P_URL="http://127.0.0.1:$POLICY_PORT/v1"
      fi
    elif [ -z "$P_URL" ]; then
      echo "[$POLICY] not in the open-policy table and POLICY_URL is unset -> no rollouts"
      echo '{"status":"policy_not_servable","stage":"config"}' > "$OUT/status.json"; return
    fi
    local PF_RC=0 POLICY_OK=1
    if [ -z "$SPEC" ] || [ -n "${LIST// /}" ]; then
      # For an open policy this is also the shaped probe: the harness's own system prompt,
      # tool schema, tool_choice and a screenshot, against its serve.
      POLICY_URL="$P_URL" POLICIES="$BASE" MAX_TOKENS="$P_MAXTOK" \
        "$PY" "$HERE/preflight_roles.py" > "$OUT/preflight.json" 2> "$OUT/preflight.err"
      PF_RC=$?
      POLICY_OK=$("$PY" -c "import json;d=json.load(open('$OUT/preflight.json'));print(int(d['policies']['$BASE']['ok']))" 2>/dev/null || echo 0)
      echo "[$POLICY] preflight rc=$PF_RC policy_ok=$POLICY_OK $("$PY" -c "import json;p=json.load(open('$OUT/preflight.json'))['policies']['$BASE'];print(json.dumps({k:p.get(k) for k in ('s','finish','parsed','error')})[:300])" 2>/dev/null)"
    fi
stop_policy_serve() {
      [ -f "$OUT/serve/policy_vllm.pid" ] || return 0
      kill "$(cat "$OUT/serve/policy_vllm.pid")" 2>/dev/null; sleep 5; pkill -f "served-model-name $POLICY" 2>/dev/null; sleep 5
      tail -n 400 "$OUT/serve/vllm_policy_$POLICY.log" > "$OUT/serve_tail.log" 2>/dev/null
      rm -f "$OUT/serve/policy_vllm.pid"
    }
    if [ "$PF_RC" != 0 ]; then echo '{"status":"dependency_preflight_failed"}' > "$OUT/status.json"; cat "$OUT/preflight.json"; stop_policy_serve; return; fi
    if [ "$POLICY_OK" != 1 ]; then
      echo "[$POLICY] not servable now -> no rollouts (not a model score)"
      echo '{"status":"policy_not_servable","stage":"request"}' > "$OUT/status.json"; stop_policy_serve; return
    fi
    # SHARDS slices of the task set (round-robin, eval_cascp_online --shard), at most PAR
    # running at once. With SHARDS = number of tasks, each slice is one task, so an
    # interruption costs only the tasks in flight: every finished slice is kept and
    # skipped on the next attempt.
    for i in $LIST; do
      mkdir -p "$OUT/shard$i"
      if [ "$SKIP_RESUME" = 0 ] && try_resume "$OUT" "$i"; then
        echo "[$POLICY/s$i] kept (finished in an earlier attempt)"
        continue
      fi
      rm -rf "$OUT/shard$i"; mkdir -p "$OUT/shard$i"  # a shard that ended env_unreachable is rerun
      while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
      (
        cd "$SRC" || exit 1
        CASCP_SHOT_DIR="$OUT/shard$i/frames" POLICY_URL="$P_URL" timeout -k 60 "${POLICY_TIMEOUT:-4800}" "$PY" "$HERE/run_eval.py" \
          --served-url "$P_URL" --served-model "$BASE" \
          --eval-rollouts "$P_TASKS" \
          --judge-url "$JUDGE_URL" --judge-model "$JUDGE_MODEL" --judge-kind lookup \
          --describer-url "$DESCRIBER_URL" --describer-model "$DESCRIBER_MODEL" \
          --answerer-endpoint "$ANSWERER_ENDPOINT" --answerer-model "$ANSWERER_MODEL" \
          --max-tokens "$P_MAXTOK" --shard "$i/$N" \
          "${JUDGE_V2_ARGS[@]}" \
          --out "$OUT/shard$i/scores.json" > "$OUT/shard$i/run.log" 2>&1
        RC=$?
        echo "{\"status\":\"done\",\"phase1_rc\":$RC,\"probe_retry\":0}" > "$OUT/shard$i/status.json"
        echo "[$POLICY/s$i] phase1 rc=$RC rollouts=$(grep -c ' amb=' "$OUT/shard$i/run.log")"
      ) &
      sleep 2
    done
    wait
    stop_policy_serve
    "$PY" "$HERE/merge_shards.py" "$OUT" "$P_TASKS" > "$OUT/merge.log" 2>&1
    echo "[$POLICY] merge rc=$?"; head -c 4000 "$OUT/merge.log"; echo
    if [ -s "$OUT/merged/ask_tasks.json" ]; then
      ( cd "$SRC" && "$PY" "$HERE/run_ask_quality.py" --tasks "$OUT/merged/ask_tasks.json" --out "$OUT/ask_quality.json" --workers 8 )
      echo "[$POLICY] phase2 rc=$?"
      "$PY" -c "import json;d=json.load(open('$OUT/ask_quality.json'));print('[ask_quality]',json.dumps({k:d.get(k) for k in ('n_asks','n_statements','n_asks_scored','n_judge_failures','n_gate_unreadable')}),json.dumps(d.get('report'))[:800])" 2>/dev/null
    else
      echo "[$POLICY] no merged ask_tasks; phase 2 skipped"
    fi
    echo '{"status":"done"}' > "$OUT/status.json"
    echo "[$POLICY] end $(date -u +%T)"
  } 2>&1 | tee "$OUT/policy.log" | sed -u "s/^/[$POLICY] /"
}
PIDS=""
VARIANT_OPEN=$(for P in ${POLICIES//,/ }; do case "$P" in *@*) b="${P%%@*}"; [ -n "$(open_spec "$b")" ] && echo "$b";; esac; done | sort -u)
if [ "$(echo "$VARIANT_OPEN" | grep -c .)" -gt 1 ]; then
  echo "[job] FATAL prompt arms of more than one open model in one run: $VARIANT_OPEN"; POLICIES=""
elif [ -n "$VARIANT_OPEN" ]; then
  read -r VREPO VREV VTP VTOK <<< "$(open_spec "$VARIANT_OPEN")"
  if ! bash "$HERE/policy_serve.sh" "$VARIANT_OPEN" "${POLICY_CHECKPOINT:-$VREPO}" "$VREV" "$VTP" "$POLICY_PORT" "$ROOT/_serve_$VARIANT_OPEN"; then
    echo "[job] $VARIANT_OPEN serve failed -> its prompt arms get no rollouts (not a model score)"
    KEEP=""
    for P in ${POLICIES//,/ }; do
      if [ "${P%%@*}" = "$VARIANT_OPEN" ]; then mkdir -p "$ROOT/$P"; echo '{"status":"policy_not_servable","stage":"serve"}' > "$ROOT/$P/status.json"
      else KEEP="$KEEP,$P"; fi
    done
    POLICIES="${KEEP#,}"
  fi
fi
for P in ${POLICIES//,/ }; do
  if [ -n "$(open_spec "$P")" ]; then run_policy "$P"   # the GPUs hold one open policy at a time
  else run_policy "$P" & PIDS="$PIDS $!"; sleep 5; fi
done
for p in $PIDS; do wait $p; done
if [ -n "$VARIANT_OPEN" ] && [ -f "$ROOT/_serve_$VARIANT_OPEN/policy_vllm.pid" ]; then
  kill "$(cat "$ROOT/_serve_$VARIANT_OPEN/policy_vllm.pid")" 2>/dev/null; sleep 5
  mkdir -p "$ROOT/_shared_"; tail -n 400 "$ROOT/_serve_$VARIANT_OPEN/vllm_policy_$VARIANT_OPEN.log" > "$ROOT/_shared_/serve_tail_$VARIANT_OPEN.log" 2>/dev/null
fi
for P in ${POLICIES//,/ }; do echo "===== $P"; grep -E "^\[$P\]|\[harness\]|FATAL|\[eval\]" "$ROOT/$P/policy.log" | head -40; cat "$ROOT/$P/status.json" 2>/dev/null; echo; done
# The max-model-len each open policy was actually served with (policy_serve.sh writes it).
"$PY" - "$ROOT" <<'PYEOF'
import glob, json, os, sys
root = sys.argv[1]; path = os.path.join(root, "run_manifest.json")
m = json.load(open(path))
served = {}
for f in glob.glob(os.path.join(root, "*", "serve", "served_max_model_len.txt")) + glob.glob(os.path.join(root, "_serve_*", "served_max_model_len.txt")):
    d = os.path.dirname(f); name = os.path.basename(os.path.dirname(d)) if os.path.basename(d) == "serve" else os.path.basename(d)[len("_serve_"):]
    served[name] = int(open(f).read().strip())
m["served_max_model_len"] = served
json.dump(m, open(path, "w"), indent=1)
PYEOF
mkdir -p "$ROOT/_shared_"; cp "$ROOT/run_manifest.json" "$ROOT/_shared_/" 2>/dev/null; cp "$ROOT/gate_check.json" "$ROOT/_shared_/" 2>/dev/null
echo "[job] done $(date -u +%FT%TZ); results in $ROOT"
