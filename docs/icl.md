# AmbiGUI in-context learning (ICL) baseline

> **This is an ablation, not a main experiment.** It asks one narrow question: does putting four
> worked examples in the prompt of an unchanged model change how it acts and asks? The main
> results of AmbiGUI are the offline and online evaluations. ICL results are
> known to be hard to reproduce: they can change with the choice and order of the examples, the
> exact prompt text, and the serving engine. A run with this code is a new measurement, not a
> reproduction of the reported numbers (see "Differences from the reported runs").
> The ICL numbers in version 6 of the paper (a 90-task cohort) cannot be reproduced with this code:
> the release runs the matched 84-task protocol described here (see "Reported results").

This part tests whether a **frozen** policy model asks better clarifying questions when its
prompt shows it a few examples, without any training. The weights never change; only the prompt
does. It runs two arms of the same online evaluation:

| arm | what the policy sees |
|---|---|
| zero-shot (`zero`) | the online harness prompt, unchanged |
| ICL-4 (`icl4`) | the same prompt with four fixed demonstrations (two good actions on clear tasks, two good questions on ambiguous tasks) inserted after the system message |

Everything else is matched: the same tasks, served model and engine, settings, simulated user and
judges. The difference in success, question quality and prompt length between the arms is the
effect of the demonstrations.

The code is self-contained in `ambigui/icl/`. It ships its own copy of the online evaluation
harness (`harness/` and `src/`, taken from the online part, see [docs/online.md](online.md)) and
reads nothing from the other parts.

| file | role |
|---|---|
| `prepare.py`, `demos.py`, `judging.py`, `journal.py`, `parsing.py`, `common.py` | select, build and check the four demonstrations (CPU, plus four ask-judge calls: two per question) |
| `prepare_condition.py`, `condition.py` | freeze the ICL-4 condition (`condition.json`), check it against the run's task file, and insert the demonstrations into each policy request |
| `headroom/` | the strict headroom guard of the ICL arm (see "Keeping the 8192-token budget") |
| `run_icl_eval.py` | phase 1 of one ICL shard: the unchanged harness runner with the condition installed |
| `harness/run_online.sh` | one arm end to end: checks, policy serving, phase 1 (rollouts and task success), merge, phase 2 (question grading) |
| `harness/preflight_icl.py`, `harness/merge_icl.py` | the ICL arm's preflight and merge: the harness's own scripts with the condition installed and its identity stamped |
| `harness/outcomes.py` | which recorded endings are valid outcomes, and which shards a rerun must redo |
| `harness/report_icl.py` | the results table: the two arms compared on paired tasks (see "Reported results") |
| `harness/*` (others), `src/` | the online harness, as in the online part |

## How one arm works

1. **Checks.** The local Chromium opens a public page; the judge endpoint answers one real request;
   every dependency role (simulated user, task judge, describer, ask judge) and the policy answer a
   real request (`preflight_roles.py`; in the ICL arm the policy probe already carries the
   demonstrations and passes the headroom guard).
2. **Phase 1.** Every task of the task file runs as its own shard (one browser process each, at
   most `PAR` at once). The policy acts in a live BrowserGym browser, up to the task's `max_steps`
   turns. When it sends a message, the simulated user answers from the task's `open_items`. When
   the rollout ends, the task judge (`lookup` rubric, with the describer) decides success. A
   rollout that ends because the policy server cannot be reached (connection refused, connection
   error, timeout, gateway error) ends as `policy_unreachable`: it is not judged and is set aside
   like `env_blocked`, never counted as a policy failure.
3. **Merge.** The shards are merged into `merged/results.jsonl` and `merged/summary.json`.
4. **Phase 2.** The ask judge (`set` rubric) grades every message the policy sent: question or
   statement, and which open items a question targets (`ask_quality.json`).

Every policy request and response, in both arms, is recorded losslessly in each shard's
`policy_calls.jsonl` (screenshots as files in `policy_image_blobs/`).

## The ICL-4 condition

The only change in the ICL arm is in `condition.py`. Every request to the policy model gets the
8 demonstration messages (user, assistant, four times) inserted right after the system message:

```
[system: unchanged harness prompt] [8 demonstration messages] [the live task: unchanged]
```

It is applied on every turn to the request the harness builds (never stored in the history, so
it never accumulates). The request must otherwise be exactly the harness's: the same system
prompt (no prompt variant), the same single `browse` tool schema with `tool_choice="none"`,
8192 output tokens, temperature 0, seed 0, at most 4 images, no streaming; anything else is
refused rather than changed. The demonstrations are text only (no screenshots), and the
assistant turns are the original action text, not native tool calls. Requests to the simulated
user and the judges are never touched.

`condition.json` pins the demonstration messages, system prompt, tool schema and the policy
checkpoint's processor files by sha256 and carries a fingerprint over all of it. Every ICL shard,
preflight record and merged output is stamped with that fingerprint (`prompt_condition_identity.json`,
`integration_identity.json`); the merge refuses shards written under a different condition.

The insertion runs **outside** the lossless recorder, so `policy_calls.jsonl` holds the expanded
request the server actually received.

## Keeping the 8192-token budget

vLLM 0.23.0 checks a prompt's length before it expands each screenshot into image tokens, and
afterwards silently lowers the output budget if the expanded prompt leaves fewer than 8192 tokens.
The demonstrations make every ICL prompt 1,963 tokens longer (with the reported demonstrations), so without a check an ICL turn
could get a smaller output budget than the matched zero-shot turn, unnoticed.

The ICL arm therefore installs a strict guard (`headroom/strict_headroom.py`) **inside** the
recorder. For every policy request it counts the fully expanded prompt exactly as the server will:
the checkpoint's chat template and tokenizer, and the server's own image-resize and image-token
functions (executed from the installed vLLM 0.23.0 and transformers 5.14.1 source files, checked
against the sha256 values in `headroom/native_sources.json`; vLLM is never imported). If
`expanded prompt tokens + 8192 > 65536` the request is **rejected**: nothing is trimmed, no image
is dropped and the budget is never lowered. A rejected request ends that rollout as a
`generation_error` with the reason in `end_detail`. Every check is logged in the shard's
`strict_headroom_checks.jsonl`: `status` `accepted` with the token count, or `rejected` with the
reason and the evidence (written before the request is refused).

The guard counts tokens the way vLLM 0.23.0 does with its default processor and chat-template
settings, as `harness/policy_serve.sh` starts it. When `POLICY_URL` points at another server,
nothing checks that the server matches; use the same vLLM version and settings. In the reported
run, the guard's count equalled the server's `usage.prompt_tokens` on every recorded response.

The zero-shot arm has no guard, as in the reported runs.

## How the four demonstrations are selected

Code: `demos.py` (selection, building, checks) and `prepare.py` (runs it once, on CPU).

**Source.** Rollouts of earlier policies on **training** tasks only: "harvest" runs, saved as
`<run>/harvest/<chunk>.tgz`. The code that produced them is not part of this release.
Only chunk 0 of each of the four harvest runs named in the config is read. A harvest rollout is
used as a source of *single turns*, not as a model of a good full trajectory.

**Which records may be used.** A record is rejected when:

| rule | why |
|---|---|
| its task is not in the training task file with `split == "train"` | demonstrations come from training tasks only |
| its task's `source` is not an approved source: `mind2web`, `meta-gui` or `osworld` (the older spellings `multimodal-mind2web` and `osworld-human` are read as `mind2web` and `osworld`) | only the approved sources may be used; rejection reason `source_not_approved` |
| its task family (the ID before `__`) or its normalised instruction matches any evaluated task (`--tasks`) or any row of the benchmark files given as exclusions (`--exclusion-tasks`) | no overlap with anything evaluated |
| it ended with an error, a judge error, `harness_error`, `generation_error`, `truncated`, `env_unreachable`, `env_blocked` or `unfinished`, or any turn hit the token limit | not a clean rollout |
| its recorded instruction is not the full task text, or its label differs from the task file | not the same task |

**Which turns may be used.**

| kind | label | conditions |
|---|---|---|
| successful act | Clear | the rollout was judged a success, asked nothing, and ended by itself (`finished` or `commit_withheld`), not cut off; the turn is a page action (`click`, `fill`, `select_option`, ...) whose dispatch succeeded every time it was sent |
| credited question | Ambiguous | the turn sends a message that the rollout's ask judge credited with at least one open item of the task; the first such question only; questions that ask for passwords, codes, keys or other login secrets are rejected |

Other conditions: the turn's own page observation must be available (the first turn may use the
recorded initial page text, because its screenshot is not stored; if the archive has no initial
page text, the first turn is shown with the line "(training screenshot omitted; textual
demonstration)" instead); long observations are cut to a 4,800-character prefix plus up to two
950-character windows around the elements the action addresses (up to about 6,800 characters); the action is
kept exactly as the policy wrote it (its original named arguments), only the reasoning is removed;
a failed earlier action of the same rollout is never shown as correct.

**Choosing four.** In the fixed order *act, question, act, question*, each slot takes the first
candidate in a seeded hash order (seed 20261022) whose task family, instruction and action text are
all new. No evaluated task, label or outcome is consulted.

**Fresh check.** Each of the two chosen questions is sent to the ask judge again. If it is not
judged a question that targets an open item, preparation stops (no replacement is searched for).

**Message format.** Each demonstration is one user message and one assistant message:

```
user:      Training demonstration 1/4 (Clear: successful policy action; screenshot omitted).
           Instruction: <task instruction>
           Recorded observation:
           <page text of that turn>
assistant: <tool_call>{"name": "browse", "arguments": {...the original action...}}</tool_call>
```

The files written are `demonstrations.json` (the four demonstrations with full source evidence:
archive checksums, record hashes, original policy output, every rejection, the source rule and
the sha256 of the task file they were checked against), `demo_messages.json` (the 8 messages),
`demo_source_records.json` (the four source records), `tools.json` and `system_prompt.txt`.

`verify_frozen` (in `demos.py`) re-checks all of this. `prepare` runs it right after building.
`prepare_condition` copies the files into the condition and records the source rule, the
qualification gates and the task file's sha256 in `condition.json` (so they are part of the
condition fingerprint). The ICL arm runs `verify_frozen` again before it starts
(`python -m ambigui.icl.condition`, called by `run_online.sh`, and again in the preflight and in
every shard), against the task file the run actually uses: that file must be the one the
demonstrations were checked against, no demonstration may share an ID, task family or normalised
instruction with any of its tasks, and every demonstration must come from an approved source.
Otherwise the arm does not start.

## Data you need

None of these files ship with this repository.

| input | what it is | how to get it |
|---|---|---|
| online benchmark task file (`AMBIGUI_TASKS`, also `--tasks` of `prepare`) | JSON Lines, one task per line, with `sample_id`, `instruction`, `start_url`, `ambiguity`, `open_items`, `max_steps`, `flags` and `has_action_ground_truth` (the last two for the success views) | build it from the pointer-only benchmark release (`benchmark/`) and your own copy of the source datasets (`ambigui_online_benchmark.jsonl`, see [docs/online.md](online.md), "Tasks"). Sources are `mind2web`, `meta-gui` and `osworld`. The simulated user and the ask judge use `open_items` only; `items_from_decision_points` is never merged in |
| training task file (`--train-tasks`) | the training tasks, same row format, with `split == "train"` and a `source` | **not released**: the release does not include the training task files or a way to build them. You need your own training tasks in the same row format as the online task file, with `split == "train"` and an approved `source` |
| benchmark files for the exclusion check (`--exclusion-tasks`, optional) | further benchmark files no demonstration may overlap | for example the offline gold file built from `benchmark/` |
| harvest chunks (`--harvest-root`) | `<run>/harvest/0.tgz` for the four runs named in the config | **not released**, and the code that wrote them is not part of this release. Each chunk is a `.tgz` holding one `record.json` per rollout, in the format `demos.py` reads (`build`). You must produce your own rollouts in that format; your four demonstrations will differ from the reported ones |
| policy checkpoint folder (`--checkpoint` of `prepare_condition`) | the policy's processor, tokenizer and template files; `prepare_condition` refuses files whose sha256 differs from the values the config pins for the revision (`processor_files_sha256`) | `python ambigui/icl/harness/fetch_weights.py Qwen/Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 weights/qwen3.8-27b-base-online` (the folder `run_online.sh` serves from, so the weights are downloaded once) |

The reported demonstrations themselves (their messages and source rollouts) are not released.

## Install

Python 3.12 on Linux.

```bash
pip install -r requirements/icl.lock && pip install --no-deps -e .     # or: pip install -e ".[icl]"
PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.playwright.dev python -m playwright install --with-deps chromium
```

To serve the policy on this machine (one 80 GB GPU in the reported runs), install vLLM 0.23.0,
best in its own environment, and point `VLLM_PY` at that Python: `pip install -e ".[icl-serve]"`.
The ICL arm's headroom guard reads four source files of that vLLM installation; `run_online.sh`
finds them through `VLLM_PY`, or set `ICL_VLLM_SITE_PACKAGES` to its `site-packages` folder (for
example after `pip install --no-deps --target /opt/vllm-src vllm==0.23.0` when the policy is
served elsewhere).

**Library versions of the reported ICL runs.** These are the versions the reported runs used.
Other versions (of these libraries, the GPU or the driver) may change outputs, even at
temperature 0.

| where | library or component | version |
|---|---|---|
| policy server (shared by both arms) | vLLM | 0.23.0 (V1 engine, eager mode, FLASH_ATTN attention backend for the language model and the vision encoder) |
| policy server | transformers / tokenizers | 5.14.1 / 0.22.2 |
| policy server | torch | 2.11.0+cu130 (CUDA 13.0) |
| policy server | triton | 3.6.0 |
| policy server | flashinfer | 0.6.12 (installed; not the selected attention backend) |
| policy server | NVIDIA CUDA runtime / cuBLAS / cuDNN / NCCL | 13.0.96 / 13.1.0.3 / 9.19.0.56 / 2.28.9 |
| policy server | pillow (image preprocessing on the server) | 12.3.0 |
| policy server | Python | 3.12 |
| policy server | GPU | 1 x A100 80 GB class; driver version not recorded |
| headroom guard (ICL arm) | transformers / tokenizers / jinja2, and the vLLM 0.23.0 source files it reads | 5.14.1 / 0.22.2 / 3.1.6 |
| browser side | Chromium / Playwright / BrowserGym | 125.0.6422.26 / 1.44.0 / 0.14.3 |

The versions of the other client-side packages (for example openai and pillow in the harness
process) were not recorded for the reported runs; `requirements/icl.lock` pins the versions this
release was verified with.

Docker: `docker build -f docker/icl.Dockerfile -t ambigui-icl .` builds an image with this part,
its locked packages, the browser and the vLLM 0.23.0 source files the guard reads. Models are not
inside the image.

**Models.**

| role | model | how it is reached |
|---|---|---|
| policy | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16, no adapters, served name `qwen3.8-27b-base-online` | served by `harness/policy_serve.sh` on this machine; or set `POLICY_URL` to a server that serves it under that name with the same settings |
| simulated user | Llama-3.2-90B-Vision-Instruct, served name `llama3.2-90b-vision-instruct` | an OpenAI-compatible endpoint at `ANSWERER_ENDPOINT` |
| task judge, describer, ask judge | Qwen3-VL-235B-A22B-Instruct, served name `qwen3-vl-235b-a22b-instruct`, temperature 0 | an OpenAI-compatible endpoint at `JUDGE_URL` |

How to self-host the simulated user and the judge with vLLM, and how API keys are matched to
endpoints: [docs/online.md](online.md), "Serve the models".

## Run

```bash
export JUDGE_URL=http://127.0.0.1:8300/v1 ANSWERER_ENDPOINT=http://127.0.0.1:8500/v1
export AMBIGUI_TASKS=data/ambigui_online_benchmark.jsonl
CFG=configs/icl/q38_27b_matched84.json

# 1. Select, build and check the four demonstrations once (CPU; four ask-judge calls).
python -m ambigui.icl.prepare --config $CFG --train-tasks <your training task file> \
  --tasks $AMBIGUI_TASKS --exclusion-tasks data/gold_eval_set.jsonl \
  --harvest-root <folder with your harvest chunks> --out runs/icl_demos

# 2. Freeze the ICL-4 condition (CPU, no model calls).
python -m ambigui.icl.prepare_condition --config $CFG --demos-dir runs/icl_demos \
  --checkpoint weights/qwen3.8-27b-base-online --out runs/icl_condition

# 3. Both arms, with the same tag and endpoints.
bash ambigui/icl/harness/run_online.sh matched zero
ICL_CONDITION_DIR=runs/icl_condition bash ambigui/icl/harness/run_online.sh matched icl4

# 4. The results table (no model calls).
python ambigui/icl/harness/report_icl.py $AMBIGUI_TASKS runs/matched/zero runs/matched/icl4 \
  --out runs/matched/icl_report.json
```

Each arm writes to `runs/<tag>/<arm>/`: `run_manifest.json`, `preflight.json`,
`shard<i>/` (`scores.json`, `run.log`, `policy_calls.jsonl`, `ask_tasks.json`; in the ICL arm also
`prompt_condition_identity.json` and `strict_headroom_checks.jsonl`), `merged/` (`results.jsonl`,
`summary.json`, `ask_tasks.json`) and `ask_quality.json`. Rerunning the same command keeps every
finished shard and reruns only shards that are missing or did not finish, and shards that ended
`env_unreachable` or `policy_unreachable` (`harness/outcomes.py`; records from before the
`policy_unreachable` ending are read the same way when their `end_detail` is a connection error,
timeout or gateway error). A request refused by the headroom guard stays a `generation_error` and
is not rerun.

`run_online.sh` exits non-zero when the arm cannot run: 2 (usage, or the ICL-4 condition fails its
check against the task file), 5 (judge endpoint), 6 (gate check), 7 (policy serve), 8 (dependency
preflight), 9 (merge). The browser check only reports; it does not stop the arm.

Fixed settings (both arms):

| setting | value |
|---|---|
| policy serving | vLLM 0.23.0, BF16, tensor parallel 1, max model length 65536, 4 images per request, GPU memory 0.90, at most 8 sequences, eager mode, `--seed 1234`, the checkpoint's chat template, no reasoning parser |
| generation | 8192 new tokens per turn; temperature 0; request seed 0; `tool_choice="none"`, reply parsed with the `auto` parser |
| observation | `som_state`, image budget 4 (older screenshots dropped) |
| turns | the task's own `max_steps` |
| deadlines | 600 s per rollout, 3600 s per shard process (`POLICY_TIMEOUT`); one task per shard; the reported runs ran one shard at a time (`PAR=1`) |
| start page | wait for `domcontentloaded` (60 s), then up to 10 s for `load` |
| task judge / ask judge | `lookup` (with describer) / `set` |
| recording | every policy request and response, both arms (`RECORD_POLICY_CALLS=1`) |
| gate check | optional `GATE_CASES`: recorded messages replayed through the simulated user and the ask judge before any rollout (`harness/gate_check.py`); the reported runs used one; its case file (recorded messages) is not released |
| optional fallback proxy | `BRIGHTDATA_PROXY_SERVER` (+ `_USERNAME`, `_PASSWORD`), used only after a direct attempt meets a blocker |

### Environment variables

Every environment variable the ICL code reads.

| variable | read by | default | meaning |
|---|---|---|---|
| `AMBIGUI_TASKS` | `run_online.sh` | required | the online benchmark task file |
| `ANSWERER_ENDPOINT` | `run_online.sh`, harness | required | simulated user `/v1` URL |
| `JUDGE_URL` | `run_online.sh`, harness, `prepare` | required | task judge `/v1` URL (also describer and ask judge unless set below) |
| `ICL_CONDITION_DIR` | `run_online.sh` | required for `icl4` | folder written by `prepare_condition` |
| `ICL_CONFIG` | `run_online.sh` | `configs/icl/q38_27b_matched84.json` | run config (policy, simulated user and judge names) |
| `ICL_VLLM_SITE_PACKAGES` | `run_online.sh`, headroom guard | found from `VLLM_PY` | `site-packages` folder of an installed vLLM 0.23.0 whose source files the guard reads (a relative path is made absolute) |
| `VLLM_PY` | `run_online.sh`, `policy_serve.sh` | `python3` | Python with vLLM 0.23.0 installed |
| `PYTHON_EXEC` | `run_online.sh`, `policy_serve.sh` | `python3` | Python for the harness and helper steps (a relative path is made absolute) |
| `POLICY_URL` | `run_online.sh`, harness | unset (serve locally) | `/v1` URL of a server that already serves the policy under its served name |
| `POLICY_PORT` | `run_online.sh` | `8400` | port of the local policy server |
| `POLICY_CHECKPOINT` | `run_online.sh` | the config's Hugging Face repo | local checkpoint folder to serve instead of downloading |
| `POLICY_SERVE_EXTRA` | `run_online.sh`, `policy_serve.sh` | `--dtype bfloat16 --enforce-eager --max-num-seqs 8` | extra vLLM flags (the default is the reported runs' setting) |
| `POLICY_TIMEOUT` | `run_online.sh` | `3600` | seconds per shard process |
| `WEIGHTS_DIR` | `policy_serve.sh` | `./weights` | download folder for the policy checkpoint |
| `HF_TOKEN` | `fetch_weights.py` | unset | Hugging Face token, for gated repositories only |
| `OUT_ROOT` | `run_online.sh` | `./runs` | output root |
| `PAR` | `run_online.sh` | all shards | shards running at once (the reported runs used 1) |
| `GATE_CASES` | `run_online.sh` | unset (no gate check) | case file for the optional gate check |
| `DESCRIBER_URL`, `ASK_JUDGE_URL` | `run_online.sh`, harness | `JUDGE_URL` | describer and ask judge `/v1` URLs |
| `POLICY_API_KEY`, `JUDGE_API_KEY`, `DESCRIBER_API_KEY`, `ANSWERER_API_KEY`, `ASK_JUDGE_API_KEY`, `AMBIGUI_API_KEY` | `src/evaluation/endpoint_auth.py` | unset (sends `EMPTY`) | API key per endpoint; `AMBIGUI_API_KEY` for any endpoint without its own |
| `BRIGHTDATA_PROXY_SERVER`, `BRIGHTDATA_PROXY_USERNAME`, `BRIGHTDATA_PROXY_PASSWORD` | `src/evaluation/web_transport.py` | unset | optional fallback proxy, used only after a direct attempt meets a blocker |
| `PROXY_ROUTE_LABEL` | `harness/run_eval.py` | the module's own label | name of that proxy in the records |
| `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` (and lower-case forms) | `src/evaluation/web_transport.py` | | cleared before the browser starts, so it goes direct |
| `TMPDIR` | `harness/preflight_roles.py` | `/tmp` | where the preflight writes its test screenshot |
| `JUDGE_VERSION`, `CURATED_SIDECAR`, `JUDGE_CONFIG`, `JUDGE_JOURNAL_ROOT` | `run_online.sh` | unset (v1 task judge) | select task judge v2 and give its three inputs (see "Task judge version") |

Set by `run_online.sh` itself (do not set them): `RECORD_POLICY_CALLS=1`, `POLICY_MAX_CONTEXT=65536`,
`ANSWERER_MODEL`, `JUDGE_MODEL`, `DESCRIBER_MODEL`, `ASK_JUDGE_MODEL` (from the config),
`CASCP_SHOT_DIR`, `POLICIES` and `MAX_TOKENS` (per step). `ASK_PROMPT_VARIANT` must be unset: the
ICL arm refuses to run with it.

## Reported results

The ICL results come from the **84 online benchmark tasks** (the online part's task set), run as
the matched zero-shot vs ICL-4 comparison above.

> **Placeholder.** The reported matched 84-task ICL results were produced with demonstrations
> selected before the approved-source rule of this release existed. A run with this release selects
> different demonstrations, so it is a new measurement, not a reproduction of the reported ICL-4
> numbers. The zero-shot arm has no demonstrations.

| measure | zero-shot | ICL-4 | ICL-4 minus zero-shot |
|---|---|---|---|
| paired success (tasks valid in both arms) | placeholder | placeholder | placeholder |
| question precision | placeholder | placeholder | placeholder |
| recall on tasks that need a question | placeholder | placeholder | placeholder |
| prompt tokens per policy call | placeholder | placeholder | placeholder |
| paired tasks (n) | placeholder | | |

`python ambigui/icl/harness/report_icl.py <task file> runs/<tag>/zero runs/<tag>/icl4 --out <file>`
computes the table from the two arms' recorded files (no model calls). It prints one success row
per view and per task group (all, Ambiguous, Clear), the question measures and the prompt tokens,
and writes every count to `--out`.

**How the table is defined.**

| part | definition |
|---|---|
| valid outcome | the arm has a result for the task, and the rollout was not set aside: not `env_blocked` (start page could not be acted on), not `policy_unreachable` (policy server could not be reached), not a `harness_error`. `env_unreachable` stays in, as a failure, as in the online success views |
| task success | the verdict on the result row (the in-run task judge: v1 by default, v2 with `JUDGE_VERSION`, see "Task judge version"). A row with a judge v2 verdict (`judge_version` `judge_v2_*`) is read as v2: true, false, or null = **unresolved**. Unresolved is never a failure: it is excluded and counted per arm and per view. A row without `judge_version` is judge v1 (a v1 row with a `judge_error` has no verdict and is counted as unresolved too). The two arms must carry one judge version; a mix is refused, never pooled |
| success views | the same views as the other online results ([docs/online.md](online.md), "Success views"): harness view, README view, strict no-flags view and action ground-truth view, with Ambiguous and Clear tasks reported apart |
| pairs | per view: the tasks of that view with a valid outcome and a resolved verdict in both arms. Success is computed on the pairs only, and the number of pairs is reported |
| question precision, recall on tasks that need a question | as in the online part (phase-2 grading, `ask_quality.json`): question precision = questions that targeted a not-yet-asked open item / questions; recall = tasks with open items where at least one question targeted one / tasks with open items. Statements never count as questions. An arm that asked nothing has no precision ("not measured"), not a precision of zero. Computed over the tasks with a valid outcome in both arms that phase 2 graded in both arms |
| prompt tokens | `usage.prompt_tokens` of each policy call in `policy_calls.jsonl` (in the ICL arm this includes the demonstrations), as the mean per call over the tasks with a valid outcome in both arms. Calls without a usage count are counted, not guessed |

### Task judge version

The in-run task judge is v1 by default, as in the reported runs. To use the intent-aware judge v2
(`judge_v2_with_intent/2.0.3`; see [docs/judge_v2.md](judge_v2.md) for what it checks), set all four:

| variable | meaning |
|---|---|
| `JUDGE_VERSION` | `judge_v2_with_intent/2.0.3` |
| `CURATED_SIDECAR` | the task rows with the curated `open_items` (the online task file built from `benchmark/` has this shape) |
| `JUDGE_CONFIG` | start from `configs/icl/judge_v2.json`; set the judge and describer endpoints and your serving backend |
| `JUDGE_JOURNAL_ROOT` | a new, empty folder for the judge request/response journal |

`run_online.sh` passes them to every shard of both arms (the code is the ICL part's own copy in
`src/evaluation/judge_v2_*.py` and `score_rollout_v2.py`). A v2 verdict that cannot be decided is
null (unresolved), never a failure; a rollout set aside (blocked start page, policy server
unreachable, harness error) gets no v2 verdict. Use the same judge version in both arms.
`harness/report_views.py` (copied from the online part) gives the per-arm success views and asking
measures; `harness/report_icl.py` gives the paired table.

## Differences from the reported runs

1. **Task file.** The reported runs used the online part's 84-task file, in which the simulated
   user saw one merged list of open items (`open_items` plus `items_from_decision_points`). The
   released task file keeps the two lists apart and the harness uses `open_items` only, so a run
   with it is a new measurement, not a reproduction (see [docs/online.md](online.md), "Tasks").
2. **Demonstrations.** The reported ICL-4 run used four demonstrations selected before the
   approved-source rule existed, from harvests of unreleased round-1 checkpoints. Their exclusion
   check ran against a 107-row held-out file of training-pool tasks and an exclusion index built
   from the offline benchmark file, not against the 84-task file; a later check against the 84
   tasks found no overlap by ID, task family or normalised instruction. This code accepts only
   approved-source training tasks and checks against the task file you run, so a run with it
   selects different demonstrations and is a new measurement; yours come from your own harvests.
3. **Model serving.** The reported runs reached the simulated user and the judge through shared
   hosted deployments (see [docs/online.md](online.md), "Known differences"); the policy was served
   by vLLM 0.23.0 with the settings above, shared by both arms. A different server or engine version
   can produce different tokens at temperature 0.
4. **Network route.** The reported runs reached the web through an institutional forward proxy;
   here the browser goes direct unless you set a proxy.
5. **Shard scheduling.** The reported runs ran on a cluster scheduler with retries of interrupted
   shards; here `run_online.sh` runs every shard on one machine and keeps finished shards on rerun.
   An interrupted shard, or one whose page or policy server could not be reached, is rerun; a
   rollout that lost its policy server is never scored as a policy failure.
6. **Code changes.** Options, modes and files that no reported run used were removed; the scoring
   and request paths are unchanged.
