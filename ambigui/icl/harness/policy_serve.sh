#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Serve the frozen ICL policy with vLLM on this machine's GPUs.
#   bash policy_serve.sh <slug> <hf repo id | local dir> <revision | -> <tp> <port> <log_dir>
# <slug> is the name the policy is served and recorded under. A Hugging Face repo id is
# downloaded at the pinned <revision> (a commit hash); a local directory is served as is
# ("-" as the revision). Exits 0 once /v1/models lists <slug>; the server's pid is in
# <log_dir>/policy_vllm.pid. Exit 6: weights could not be fetched. Exit 7: the serve
# never came up (root cause printed).
#
# Environment: VLLM_PY (python with vllm installed, default python3), PYTHON_EXEC (python
# for the helper steps, default python3), WEIGHTS_DIR (download root, default
# ./weights), HF_TOKEN for gated repositories, POLICY_MAX_CONTEXT (run_online.sh sets 65536),
# POLICY_SERVE_EXTRA (extra vLLM flags; run_online.sh sets the reported runs' flags).
set -uo pipefail
SLUG="$1"; SRC="$2"; REV="$3"; TP="$4"; PORT="$5"; LOG_DIR="$6"; mkdir -p "$LOG_DIR"
VPY="${VLLM_PY:-python3}"
PY="${PYTHON_EXEC:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$LOG_DIR/vllm_policy_$SLUG.log"
WEIGHTS_DIR="${WEIGHTS_DIR:-$PWD/weights}"

VENV=(env -u PYTHONPATH -u PYTHONHOME "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1"
  "VLLM_NO_USAGE_STATS=1" "DO_NOT_TRACK=1"
  "VLLM_USE_FLASHINFER_MOE_FP8=0" "VLLM_USE_FLASHINFER_MOE_FP16=0")
"${VENV[@]}" "$VPY" -c "import torch, vllm, transformers; n=torch.cuda.device_count(); print('[serve:$SLUG] torch', torch.__version__, 'gpus', n, 'vllm', vllm.__version__, 'transformers', transformers.__version__); import sys; sys.exit(0 if n >= $TP else 9)" \
  || { echo "[serve:$SLUG] FATAL fewer than $TP GPUs visible"; exit 7; }

if [ -d "$SRC" ]; then
  DEST="$SRC"
else
  DEST="$WEIGHTS_DIR/$SLUG"
  [ -f "$DEST/config.json" ] || "$PY" "$HERE/fetch_weights.py" "$SRC" "$REV" "$DEST" \
    || { echo "[serve:$SLUG] FATAL weights fetch failed"; exit 6; }
fi

# The weight identity is recorded before serving.
"$PY" - "$DEST" "$LOG_DIR/base_checkpoint_identity.json" "$SRC" "$REV" <<'PYEOF'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); files={}
for path in sorted(root.iterdir()):
    if not path.is_file(): continue
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''): h.update(data)
    files[path.name]=h.hexdigest()
pathlib.Path(sys.argv[2]).write_text(json.dumps({'source':sys.argv[3],'revision':sys.argv[4],'files_sha256':files},indent=2)+'\n')
PYEOF

# The context bound is the model's own, capped at 64k: vLLM refuses to start above
# max_position_embeddings, and every request here fits well inside 64k.
MAXLEN=$("$PY" - "$DEST" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
vals = []
for k in ("text_config", "llm_config", "language_config"):
    if isinstance(c.get(k), dict):
        vals.append(c[k].get("max_position_embeddings"))
vals.append(c.get("max_position_embeddings"))
vals = [v for v in vals if isinstance(v, int) and v > 0]
print(min([65536] + vals))
PYEOF
)
MAXLEN=${POLICY_MAX_CONTEXT:-$MAXLEN}
CT=(); [ -f "$DEST/chat_template.jinja" ] && CT=(--chat-template "$DEST/chat_template.jinja")
# Extra vLLM flags from run_online.sh (the reported ICL runs: --dtype bfloat16 --enforce-eager --max-num-seqs 8).
read -r -a EXTRA <<< "${POLICY_SERVE_EXTRA:-}"
echo "[serve:$SLUG] vLLM tp=$TP max-model-len=$MAXLEN image-limit=4 chat-template=${CT[*]:-tokenizer} extra=${EXTRA[*]:-none}"
"${VENV[@]}" "$VPY" -m vllm.entrypoints.openai.api_server \
  --model "$DEST" --served-model-name "$SLUG" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" --trust-remote-code \
  --limit-mm-per-prompt "{\"image\": 4}" \
  --gpu-memory-utilization 0.90 --seed 1234 "${CT[@]}" "${EXTRA[@]}" > "$LOG" 2>&1 &
PID=$!; echo $PID > "$LOG_DIR/policy_vllm.pid"
for i in $(seq 1 120); do
  if curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$SLUG\""; then
    echo "[serve:$SLUG] ready after $((i*10))s"; exit 0
  fi
  kill -0 $PID 2>/dev/null || break
  sleep 10
done
kill $PID 2>/dev/null; sleep 5
echo "[serve:$SLUG] FAILED to come up; root-cause lines:"
grep -n -E "Error|error:|Exception|out of memory|Assert|not supported|Unrecognized|failed" "$LOG" \
  | grep -v -E "Triton|UserWarning|INFO |Engine core initialization failed" | head -30 | cut -c1-400
exit 7
