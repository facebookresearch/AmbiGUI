# AmbiGUI online evaluation

This part of the repository runs GUI agents on live websites and measures two things: whether
the agent finishes the task, and whether it asks the user a clarifying question when the
instruction leaves an important choice open. It is the code behind the paper's online
(live-browser) results.

## What it does

Each task gives the agent an instruction and a start page. An instruction is **ambiguous**
when it leaves open a choice that changes the outcome (for example which size or date), and
**clear** otherwise. The benchmark marks each task Ambiguous or Clear; these two groups are
always reported separately and never pooled.

One evaluation run has two phases.

1. **Phase 1, rollouts.** A real Chromium browser opens the start page. The model under test
   (the **policy**) sees a screenshot with numbered element marks and the page state, and
   answers with one tool call per turn: click, type, scroll, go to a URL, send a message to the
   user, or report the task complete or infeasible. One full attempt at a task is a
   **rollout** (also called an episode).
   - When the policy sends a message to the user, a **simulated user** replies. The simulated
     user (in the code, the "answerer") is a language model that knows what the person really
     wanted: the task's **open items**, which are the choices the instruction left open,
     each with the value the person chose. It answers only what was asked. A message that is
     a report rather than a question gets a fixed refusal.
   - After the rollout, a **task judge** decides success. The judge reads the actions; when it
     needs to know what was on screen, it asks a vision model (the **describer**) questions
     about the screenshots, then decides.
     Version 2 of the task judge also checks the outcome against the task's intended
     `open_items` values; see [docs/judge_v2.md](judge_v2.md). Set `JUDGE_VERSION`,
     `CURATED_SIDECAR`, `JUDGE_CONFIG` and `JUDGE_JOURNAL_ROOT` to use it; unset, the v1 judge
     of the recorded runs is used.
2. **Phase 2, question grading.** An **ask judge** reads every message the policy sent. It
   decides whether each message is a real question or a statement, and which open items each
   question targets. This runs after the rollouts, so it can be rerun under another judge
   without repeating the browser work.

Finally `report_views.py` turns the recorded files into the results table.

Code layout:

| path | what it is |
|---|---|
| `ambigui/online_eval/harness/run_online.sh` | the launcher: checks, phase 1 for every policy, merge, phase 2 |
| `ambigui/online_eval/harness/run_eval.py` | runs `src/scripts/eval_cascp_online.py` (the rollout loop) with three recorded patches (see its docstring) |
| `ambigui/online_eval/harness/run_ask_quality.py` | runs `src/scripts/score_online_ask_quality.py` (phase 2) |
| `ambigui/online_eval/harness/merge_shards.py` | pools the shards of one policy, recomputing rates from the pooled rows |
| `ambigui/online_eval/harness/report_views.py` | the paper's results table, from recorded files only |
| `ambigui/online_eval/harness/policy_serve.sh`, `fetch_weights.py` | serve open-weights models with vLLM from pinned Hugging Face revisions |
| `ambigui/online_eval/harness/preflight_roles.py`, `judge_probe.py`, `gate_check.py` | checks that every model endpoint answers before any browser opens |
| `ambigui/online_eval/harness/prompt_variant.py`, `data/prompts/` | the prompt arms p0 to p5 of the prompt sweep |
| `ambigui/online_eval/harness/policy_call_recording.py` | optional lossless journal of every policy call |
| `ambigui/online_eval/harness/verify_merge.py` | checks a fine-tuned checkpoint after its adapter is merged |
| `ambigui/online_eval/src/` | the rollout loop, live browser environment, simulated user, ask judge, task judge and scoring |
| `configs/online/open_policies.tsv` | the open-weights policies, with pinned revisions and per-model generation budgets |

## Install

Python 3.12 on Linux.

```bash
pip install -e ".[online]"                 # or, exactly: pip install -r requirements/online.lock && pip install --no-deps -e .
PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.playwright.dev python -m playwright install --with-deps chromium
```

This installs Chromium 125 (build 1117), the browser of Playwright 1.44 and of the paper runs.
BrowserGym 0.14.3 requires exactly Playwright 1.44. That version's built-in download address no
longer serves files, so `PLAYWRIGHT_DOWNLOAD_HOST` points it at Playwright's current download
server, which still serves the same build. `--with-deps` also installs the system libraries
Chromium needs, with `apt-get` and root rights. Without them, leave out `--with-deps` and install
those libraries yourself (`python -m playwright install-deps --dry-run chromium` lists them).

To serve open-weights models on this machine (GPUs needed), also install vLLM:
`pip install -e ".[online-serve]"` (vLLM 0.23.0, the version of the paper runs). It is often
easier to install vLLM in its own environment and point `VLLM_PY` at that Python.

Docker: `docker build -f docker/online.Dockerfile -t ambigui-online .` (from the repository root) builds an image with the harness, the locked
packages and the browser. Models are not inside the image; they are reached over HTTP.

### Tasks

The runner reads a benchmark task file in JSONL form: one task per line with at least
`sample_id`, `instruction`, `start_url`, `ambiguity`, `open_items`, `flags`,
`has_action_ground_truth` and `max_steps` (the per-task turn cap: 3 x the length of the recorded
trajectory). Build it from the pointer-only benchmark release in `benchmark/` against your own
copy of the source datasets (`ambigui_online_benchmark.jsonl`).

Each task has two lists of open items, as in the gold file:

- `open_items`: the user's intended values for the details the instruction leaves open. The
  simulated user answers from this list, and the ask judge grades questions against it.
- `items_from_decision_points`: items read off the annotators' decision records. Never merge
  this list into `open_items`.

**Difference from the paper runs.** In the paper's online runs, the simulated user saw one merged
list (both kinds of item). The released file keeps them apart, so with the released file the
simulated user answers from `open_items` only. This is a different set-up from the paper runs:
re-running with it is a new measurement, not a reproduction of the paper's trajectories. For the
same reason, a correct rebuild does not match the paper's 84-task file (SHA-256
`4efd151060567ab55a85a272fbbae26934b2d05cd1e7f66d27ec6e5ad74ab84f`), which also carried fields the
release drops. Check your rebuild with `sha256sum -c benchmark/manifest/CHECKSUMS.sha256`.

## Serve the models

Every model is reached through an **OpenAI-compatible endpoint**: an HTTP server that speaks
the OpenAI Chat Completions API at a base URL ending in `/v1`. vLLM provides one for
open-weights models; hosted providers provide one for their models (directly or through a
translating proxy you run yourself). For each role you give a base URL, a served model name
and, if the endpoint needs one, an API key.

| role | URL | model name | key (optional) | paper model |
|---|---|---|---|---|
| policy (API) | `POLICY_URL` | the policy name on the command line | `POLICY_API_KEY` | frontier models |
| policy (open weights) | started by the runner | slug in `configs/online/open_policies.tsv` | none | see the table |
| simulated user | `ANSWERER_ENDPOINT` | `ANSWERER_MODEL` | `ANSWERER_API_KEY` | Llama-3.2-90B-Vision-Instruct |
| task judge | `JUDGE_URL` | `JUDGE_MODEL` | `JUDGE_API_KEY` | Qwen3-VL-235B-A22B-Instruct |
| describer | `DESCRIBER_URL` (default: judge) | `DESCRIBER_MODEL` (default: judge) | `DESCRIBER_API_KEY` | same as the judge |
| ask judge | `ASK_JUDGE_URL` (default: judge) | `ASK_JUDGE_MODEL` (default: judge) | `ASK_JUDGE_API_KEY` | same as the judge |

A key is matched to an endpoint by its URL. `AMBIGUI_API_KEY` is used for any endpoint without
its own key; with no key set, the placeholder `EMPTY` is sent, which vLLM accepts. Keys only
change the `Authorization` header, never a request's content (`src/evaluation/endpoint_auth.py`).

**Policy, open weights.** `run_online.sh` serves each open policy itself, one at a time, with
`harness/policy_serve.sh`: it downloads the Hugging Face repo at the pinned revision from
`configs/online/open_policies.tsv` and starts vLLM with tensor parallel from the table, BF16,
`--max-model-len` = min(65536, the model's maximum positions), at most 4 images per request,
the checkpoint's own chat template, no reasoning parser, `--seed 1234`. Gated repositories
(Gemma) need `HF_TOKEN`. To serve one by hand:

```bash
bash ambigui/online_eval/harness/policy_serve.sh qwen3-vl-8b-instruct Qwen/Qwen3-VL-8B-Instruct \
  0c351dd01ed87e9c1b53cbc748cba10e6187ff3b 1 8400 runs/serve_logs
```

**Policy, API model.** Set `POLICY_URL` (and `POLICY_API_KEY`) and name the policy exactly as
the endpoint serves it. The harness sends `tools` with `tool_choice="none"` and reads the tool
call from the reply text with the `auto` parser: a Qwen3-Coder XML call first, then a
Gemma-style call, then a JSON object `{"name": ..., "arguments": ...}`. See "Known differences" for how the paper runs connected
hosted models.

**Simulated user.** The paper used Llama-3.2-90B-Vision-Instruct
(`meta-llama/Llama-3.2-90B-Vision-Instruct`, revision
`e305d2a43a4adc6987308fe7d896fb8ec5f1a5d8`) through a hosted OpenAI-compatible API. To serve it
yourself with vLLM:

```bash
vllm serve meta-llama/Llama-3.2-90B-Vision-Instruct --revision e305d2a43a4adc6987308fe7d896fb8ec5f1a5d8 \
  --served-model-name llama3.2-90b-vision-instruct --tensor-parallel-size 8 --max-model-len 16384 --port 8500
export ANSWERER_ENDPOINT=http://127.0.0.1:8500/v1 ANSWERER_MODEL=llama3.2-90b-vision-instruct
```

The simulated user's prompts are text only and its replies are short; these serve flags are a
working choice, not a setting copied from the paper runs.

**Judge.** One model plays the task judge, the describer and the ask judge:
Qwen3-VL-235B-A22B-Instruct, temperature 0, reasoning off. Set `JUDGE_URL` to an endpoint
serving it; `run_online.sh` sends it one real request before any browser opens. To serve it
yourself with vLLM (the paper's main runs used a hosted deployment, see "Known differences"):

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --revision 710c13861be6c466e66de3f484069440b8f31389 \
  --served-model-name qwen3-vl-235b-a22b-instruct --tensor-parallel-size 8 --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 48}' --max-num-seqs 32 --enforce-eager --port 8300
```

## Run

### One task

```bash
export AMBIGUI_TASKS=/path/to/full84.jsonl
export ANSWERER_ENDPOINT=http://127.0.0.1:8500/v1 ANSWERER_MODEL=llama3.2-90b-vision-instruct
export JUDGE_URL=http://127.0.0.1:8300/v1 JUDGE_MODEL=qwen3-vl-235b-a22b-instruct
bash ambigui/online_eval/harness/run_online.sh one_task qwen3-vl-8b-instruct
```

The run takes every task of the file, one task per shard. For one task, put its line alone in a
file and point `AMBIGUI_TASKS` at it.

### The full set

```bash
PAR=8 bash ambigui/online_eval/harness/run_online.sh full84 qwen3-vl-8b-instruct,qwen3-vl-8b-thinking
python ambigui/online_eval/harness/report_views.py "$AMBIGUI_TASKS" runs/full84 --out runs/full84/metrics.json
```

Each task is its own shard, so an interruption costs only the tasks in flight. Rerunning the same command with
the same `OUT_ROOT` and tag keeps every finished shard and reruns the rest, including any
shard whose start page was unreachable. `PAR` is how many shards of one policy run at once,
each with its own browser.

Outputs, under `runs/<tag>/<policy>/`: `shardN/scores.json` (per-task outcomes),
`shardN/frames/` (screenshots and `trajectory.json` per rollout), `shardN/ask_tasks.json`
(messages for phase 2), `merged/results.jsonl` and `merged/summary.json` (pooled),
`ask_quality.json` (phase 2), and `run_manifest.json` (models, endpoints, settings, package
versions, git commit) at the run root.

### Settings

| variable | default | meaning |
|---|---|---|
| `PAR` | number of tasks | shards of one policy running at once |
| `POLICY_TIMEOUT` | 4800 | process timeout of one shard, in seconds (exit code 124) |
| `RECORD_POLICY_CALLS` | unset | `1` keeps a lossless journal of every policy request and response |
| `POLICY_TASKS_DIR` | unset | directory of `<policy>.jsonl` task subsets that replace the task file for that policy |
| `JUDGE_VERSION`, `CURATED_SIDECAR`, `JUDGE_CONFIG`, `JUDGE_JOURNAL_ROOT` | unset (v1 task judge) | select task judge v2 and give its three inputs; see `docs/judge_v2.md` |
| `GATE_CASES` | unset | JSON file of recorded messages to replay through the simulated user and ask judge before running (see `gate_check.py`) |
| `OUT_ROOT` | `./runs` | where runs are written |
| `POLICY_CHECKPOINT` | unset | a local directory to serve instead of the table's Hugging Face repo |
| `BRIGHTDATA_PROXY_SERVER` (+ `_USERNAME`, `_PASSWORD`) | unset | optional fallback proxy, used only after a direct attempt meets a blocker |
| `PROXY_ROUTE_LABEL` | module label | name of that proxy route in the records |
| `POLICY_PORT` | 8400 | local port of the open-policy vLLM server; pick a free one when another server uses it |
| `POLICY_MAX_CONTEXT` | the table's context length | `--max-model-len` of the open-policy server (the matched SFT4 runs used 65536) |
| `VLLM_PY` | `python3` | the Python that has vLLM installed |
| `PYTHON_EXEC` | `python3` | the Python for the harness helper steps (weight fetch, merge checks) |
| `WEIGHTS_DIR` | `./weights` | where open-policy weights are downloaded |
| `HF_TOKEN` | unset | Hugging Face token for gated repositories |
| `MERGED_FROM_BASE`, `ADAPTER` | unset | with `POLICY_CHECKPOINT`: check a merged fine-tune against its base and adapter (`verify_merge.py`) |

Fixed for every run (the paper runs' settings): every task of the file, one task per shard;
temperature 0 and seed 0 for policy calls; 8192 new tokens per turn for API policies (open
policies use the table); a 600-second wall-clock limit per rollout; `som_state` observations; at
most 4 images per policy request (older screenshots are dropped); the start page awaited until
`domcontentloaded` for up to 60 s, then up to 10 s more for `load`; a 900-second read timeout
for the one-request judge check each shard (and the ask-quality pass) makes; a 900-second HTTP timeout
for simulated-user and ask-judge calls; the `lookup` task-judge rubric and the `set`
ask-judge rubric; one rollout at a time per shard.

## Reproduce the paper's online results

All paper runs used the 84-task file above with the fixed settings, the
Llama-3.2-90B-Vision-Instruct simulated user and the Qwen3-VL-235B-A22B-Instruct judge, and
were scored with `report_views.py`.

| paper result | policies | command differences |
|---|---|---|
| Frontier models | Claude, GPT, Grok and Gemini models, plus Muse Spark 1.3 (an internal RL-trained model, not public) | API policies through `POLICY_URL`; `PAR=8` |
| Frontier, corrected Clear rows | the same, 14 declared rows rerun | `POLICY_TASKS_DIR` with one `<policy>.jsonl` per policy holding exactly the declared rows; the corrected view replaces only those 14 rows of the frontier run |
| Open-weights batch (scale, reasoning, generation and GUI-data contrasts) | the 18 open policies of `open_policies.tsv` except `qwen3.8-27b` | open policies; `PAR=10`; max new tokens per model from the table |
| Prompt sweep, Qwen3.8-27B | `qwen3.8-27b@p0` ... `qwen3.8-27b@p5` | one run with all six arms; the model is served once; `PAR=6` |
| Prompt sweep, Muse Spark 1.3 | `<model>@p0` ... `<model>@p5` | internal model, not reproducible publicly; the arms are applicable to any policy |
| Matched Qwen3.8-27B BASE vs SFT4 | `qwen3.8-27b` and the merged SFT4 checkpoint | see `docs/online_protocol.md`; serve SFT4 with `POLICY_CHECKPOINT=<merged dir> MERGED_FROM_BASE=<base dir> ADAPTER=<adapter dir>`; `RECORD_POLICY_CALLS=1`; `POLICY_MAX_CONTEXT=65536` |

No trained weights are released with this repository, and the training code is not part of it.
The SFT4 row therefore cannot be reproduced from this repository; the same protocol runs on any
fine-tuned checkpoint you merge yourself (`docs/online_protocol.md`).

Prompt arms change only the ask paragraph of the policy's system prompt
(`harness/data/prompts/README.md`); `p0` is the baseline paragraph, word for word.

Results from the live web differ between days: sites change, go down or block automated
browsers. Compare policies only on runs made together, and pair them on tasks both could
reach.

## Success views

A **success view** decides which tasks enter the success denominator. All views use only
completed outcomes; Ambiguous and Clear tasks are reported apart.

1. **Harness view.** Every completed task, as the harness scores it. Tasks set aside because
   the start page showed a blocker (`env_blocked`, for example a bot wall) are excluded. Tasks
   whose start page could not be reached (`env_unreachable`) stay in, as failures.
2. **README view.** The harness view minus tasks flagged "Stops before the commitment the
   recording completed" (the reference recording itself stops early, so success is not
   comparable). On the 84-task file: 23 Ambiguous and 36 Clear tasks.
3. **Strict no-flags view.** The harness view restricted to tasks whose benchmark `flags`
   list is empty: 18 Ambiguous and 21 Clear tasks with the released file (the paper's file had 20
   Clear; see below).

`report_views.py` also reports an **action ground-truth view**: tasks with a recorded action
sequence (not Android-origin), 18 Ambiguous and 27 Clear. The 24 Android-origin Ambiguous
tasks have no action ground truth, so success over all 84 tasks is not the main execution result.

Asking measures come from phase 2 and exclude statements:

- **Amb real asked**: share of Ambiguous tasks with open items where the policy asked at least one real question
  (41 tasks with the released file; 42 in the paper, see below).
- **Amb on-target**: share where at least one question targeted an open item.
- **Clear real asked (false alarm)**: share of Clear tasks where the policy asked anyway; the
  12 Clear tasks flagged as carrying open items are excluded (30 remain).
- **Blocker recall**: share of tasks with open items where at least one question targeted one.
  **Question precision**: share of real questions that targeted a not-yet-asked open item.
  **Question F1**: their harmonic mean. A run that asked nothing has no precision (shown as
  "not measured"), which is not the same as a precision of zero.

## How unreachable environments are counted

Before the policy acts, the start page is loaded and checked.

- **Unreachable** (the start page did not open: the browser could not launch it, or the page
  never loaded): the rollout ends `env_unreachable` before any turn. It stays in every success
  view and counts as a **failure**; it is not removed. On a rerun of the same tag, a shard that
  ended `env_unreachable` is run again rather than kept, and the merged result is the rerun's.
- **Blocked** (the page loaded but cannot be worked on: a 403 bot wall, a CDN error page, a
  captcha interstitial, or a page with nothing to click or type): the rollout ends
  `env_blocked` and is **set aside**. It is excluded from the success denominators and from
  the asking measures, and listed in `merged/summary.json` under `env_blocked_task_ids`.
- **Harness fault** (the page opened but the harness failed to take its screenshot, or the
  harness code raised): the rollout ends `harness_error` and counts as a failure; the count of
  such endings is in the summary's `endings`.
- **No outcome** (the shard process hit `POLICY_TIMEOUT`, exit code 124, or never ran): the task
  is reported as `timed_out` or `missing`. It is never counted as a failure or as a measured
  "did not ask".

Every expected task keeps its own row in the report, so reachable, unreachable, blocked,
timed-out and missing tasks stay visible per policy.

With the fallback proxy configured, a start page that is blocked on the direct route and does
not open through the proxy is `env_blocked` (set aside), as without a proxy. Each result row
records the route that reached the start page (`transport`: `direct` or the proxy label).

## What the judges receive, and what each row records

- The task judges (v1 and v2) receive the action trace without per-action failure marks. They
  decide from the outcome shown in the screenshots, so a model is not penalised for a failed
  action itself. The harness report's tool-call fidelity rate (`strict_call_rate`) counts an
  action whose recorded `result` starts with `error` as a rejected call; it is a separate
  statistic and does not change any verdict.
- Each result row also records `task_time_evidence`: the UTC time and date at which the rollout
  started. Judge v2 reads this date for relative dates such as "today" or "tomorrow".
- Task 406da915 is Ambiguous and has no curated open item. It is rolled out like any other task.
  A question on it gets no answer from the simulated user (the policy reads "(no answer)"); these
  replies are counted as `unanswerable_no_curated_intent` in the shard and merged summaries.
  Under judge v2 its verdict is unresolved, with the reason "no curated intent".
- Under judge v2, an unresolved verdict (`success: null`) is left out of every success rate and
  counted next to it (`unresolved`), in the shard summary, in `merged/summary.json` and in the
  results table. Rows that are not judged (`env_unreachable`, `harness_error`) keep
  `success: false`, under v1 and v2 alike.

## Run sets use different simulated users and judges: never mix them

The simulated user and the judges shape every number: a different simulated user answers
questions differently, and a different judge scores success differently. Each run set of the
paper fixed its own pair, for example:

- frontier, open-weights batch, prompt sweeps and matched SFT4 runs: Llama-3.2-90B-Vision-Instruct
  simulated user with the Qwen3-VL-235B-A22B-Instruct judge;
- the AWS online runs: a GLM-4.6V simulated user.

Never put results from different run sets in one table, chart or average. Every run records its
simulated user and judges in `run_manifest.json`; check them before comparing.

## Versions used in the reported runs

Filled from recorded evidence only: each run's `run_manifest.json`, the open-policy serve lines
in `policy.log`, and the job logs' version and browser-check lines. "Not recorded" means no file
of the run states it.

| component | release pin | open-weights batch | matched Qwen3.8-27B BASE vs SFT4 | frontier models, corrected Clear rows, prompt sweeps |
|---|---|---|---|---|
| vLLM | `vllm==0.23.0` (`[online-serve]`) | 0.23.0 | 0.23.0 | not recorded (frontier policies are hosted APIs; the sweeps' open-policy serve logs carry no version line) |
| transformers | not pinned (installed with vLLM) | 5.14.1 | 5.14.1 | not recorded |
| torch / CUDA | not pinned (installed with vLLM) | 2.11.0+cu130 | 2.11.0+cu130 | not recorded |
| Chromium | the build Playwright 1.44.0 installs: 125.0.6422.26 | 125.0.6422.26 (job browser check) | 125.0.6422.26 | not recorded |
| Playwright | `playwright==1.44.0` | not recorded | not recorded | not recorded |
| BrowserGym | `browsergym-core==0.14.3` | not recorded | not recorded | not recorded |
| open-policy serving flags | `policy_serve.sh`: tensor parallel from the table, `--max-model-len` 65536 unless `POLICY_MAX_CONTEXT`, `--limit-mm-per-prompt` 4 images, `--seed 1234`, `--gpu-memory-utilization 0.90`, the table's chat template | recorded per policy: tp 1 or 2, max-model-len 65536, image limit 4, chat template from the tokenizer or the table's file | policy max context 65536 (manifest) | sweeps: not recorded beyond "vLLM on the job's GPUs" |
| task judge / describer / ask judge | — | hosted Qwen3-VL-235B-A22B-Instruct, reasoning off (manifest) | same | same |

## Known differences from the paper runs

These are the places where this public version cannot be byte-identical to how the paper runs
were made. The scoring code is unchanged. Re-scoring the recorded paper runs with it gives
identical numbers when the paper's 84-task file is used. With the released task file, two
denominators per policy differ: Ambiguous "asked" and "on-target" use 41 tasks, not 42
(task 406da915 has no curated open item, so it no longer counts as an Ambiguous task with open
items; a policy that asked on it also loses 1 from the numerator), and Clear "success no flags"
uses 21 tasks, not 20 (task 1564d78a's only flag came from a model panel, which is not
released). The Clear false-alarm denominator stays 30.

The paper runs could not tell a start page blocked on both network routes apart from an
unreachable one when the fallback proxy was used (one known case, counted as `env_unreachable`).

1. **Hosted models went through a translating adapter.** In the paper runs, the hosted
   frontier policies were reached through a local adapter that turned the harness's Chat
   Completions request into the provider's own API. For those policies it attached the one
   `browse` tool natively (the harness sends `tool_choice="none"`; the adapter sent `auto`) and
   wrote each native tool call back into the reply text as `{"name": ..., "arguments": ...}`
   ahead of any prose. It also dropped the `seed` parameter where the provider had none, and
   dropped `temperature`/`top_p` when a provider rejected them. That adapter is not part of
   this release. A public OpenAI-compatible endpoint receives the harness's request as is, so
   a hosted model may write its tool calls differently and parse differently.
2. **Simulated user and judge serving.** The paper's simulated user was a hosted
   Llama-3.2-90B-Vision-Instruct API (which ignores `seed`), and the main runs' judge was a
   shared hosted deployment of Qwen3-VL-235B-A22B-Instruct (reasoning off, temperature 0)
   behind an OpenAI-compatible bridge. A different server or engine version can produce different
   tokens at temperature 0.
3. **Network route.** In the paper runs the browser reached the web through an institutional
   forward proxy, via the fallback-proxy path. Here the browser goes direct unless you set a
   proxy. Sites that block by network origin can be reachable in one setting and not the other.
4. **The original frontier run** used an earlier revision of the simulated user, which answered
   genuine questions on tasks with nothing left open with a fixed reply instead of the
   nothing-open prompt used here (and in all later runs). The 14 corrected Clear rows were
   rerun with the current behaviour.
5. **Open-weights checkpoints** are pinned to Hugging Face revisions whose configuration,
   tokenizer, processor, template and index files are byte-identical to the copies the paper
   runs loaded, with three documented exceptions in `configs/online/open_policies.tsv`
   (formatting-only tokenizer differences; two unused template files; gated Gemma 3 repositories
   checked by file names and sizes only).
6. **Not public:** Muse Spark 1.3, the internal RL-trained frontier model, and its prompt sweep cannot be rerun.
7. **Composite and repaired views are not rebuilt by this code.** Three published views were
   assembled after the runs with tools that are not in this release:
   - the corrected-Clear frontier view: the 14 re-run Clear rows replace the same rows of
     the original frontier run. Score the re-run itself with
     `report_views.py <tasks> <run_dir> --expected <json>`, where the JSON maps each policy to
     the list of task ids it ran (`{"<policy>": ["<task id>", ...]}`), so tasks outside the
     subset are not counted as missing; the composite then takes, per policy, the original
     run's result rows with those 14 rows replaced by the re-run's rows;
   - phase-2 grading repairs: the ask-quality grading of some policies was redone after the
     run (open batch: `gemma-4-31b-it` and `qwen3-vl-4b-thinking`; prompt sweeps: Qwen3.8-27B
     arms p3 and p5, Muse Spark 1.3 arms p0 and p1);
   - top-up merges: the prompt sweeps' missing shards were run again in top-up jobs and merged
     into the main sweep runs.
8. **No wire records for the open batch.** The open-weights batch did not keep a journal of
   its policy requests and responses (`RECORD_POLICY_CALLS` was off); only the trajectories
   and scores remain.
