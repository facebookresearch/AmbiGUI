# AmbiGUI offline evaluation

## What this part does

A GUI agent is a model that operates a website or an app for a person: it reads the screen
and the person's request, and takes actions such as clicking or typing. Some requests leave
out a detail the agent needs ("book a table" — for how many?). A good agent then asks the
person before acting; when nothing is missing, it acts without asking. AmbiGUI measures
whether an agent makes that choice well.

The *offline* evaluation measures this on recorded trajectories, without a live browser. A
*trajectory* is one recorded task: the person's request, the screens they saw, and the
actions they took, step by step. At every recorded step the model under test (the
*policy*) is shown the request, the current screenshot and the recorded history so far, and
must either:

- give the action to take now, or
- ask one clarifying question.

The recorded history is used at every step, not the model's own earlier actions, so one
early mistake cannot spoil the rest of the task. This is called *teacher forcing*.

When the policy asks, another language model plays the person and replies. This model is
the *simulated user*, also called the *answerer*. It is given the request and the task's
*open items*: the details the request leaves unstated, each with the value the person
actually chose. The policy sees the reply and is asked again at the same step, until it
acts. After ten questions in a row at one step, the task is *cut off*: it stops, and its
remaining steps count as wrong.

A third model, the *ask judge*, later reads every recorded question and says which open
items it asked about. Finally the metrics are computed from the recorded files alone.

So one evaluation of one policy (one *pass*) has three stages:

| stage | program | models it needs | what it writes |
|---|---|---|---|
| run | `offline_evaluation.run.run_ask_protocol` | policy + answerer | `substrate.jsonl`, `turns.jsonl`, `asks.jsonl`, `inference.jsonl`, `run.json` |
| judge | `offline_evaluation.run.judge_asks` | ask judge | adds verdicts to `asks.jsonl`, judge calls to `inference.jsonl` |
| score | `offline_evaluation.run.score_ask_protocol` | none (no network) | `report.json`, `outcomes.jsonl`, `emissions.jsonl` |

The stages are separate programs so that a failed judge never costs the policy's GPU time
again, and so that a changed metric can be recomputed from the files without calling any
model.

### Where the code is

- `ambigui/offline_eval/src/` — the evaluation code. Re-scoring the recorded frontier run
  set with it is byte-identical. It is not identical to the code behind every published
  table: some flags, spare serves and fallbacks were removed, and the open-weight run sets
  were scored with an earlier credit rule (see "Known differences").
  Its packages import each other by top-level name, so this folder must be on `PYTHONPATH`
  when the stage programs are run directly. The launcher below does this for you.
- `ambigui/offline_eval/src/configs/models/` — the *model catalogue*: one YAML file per
  policy, with whether its weights take images (`vision`), image size and default
  output-token budget.
- `ambigui/offline_eval/launch.py` — a launcher that runs each stage from one *run-set*
  file (below). It only builds the stage's command-line flags; it adds no behaviour.
- `configs/offline/models.yaml` — which public weights answer for each model name, pinned
  to one Hugging Face commit.
- `configs/offline/run_sets/*.yaml` — one file per *run set* (defined below).

### Run sets: never mix them

A *run set* is a group of passes that share one answerer and one ask judge. The answerer
writes the words the policy reads, so it changes every number about asking — how often the
policy asks again, whether it acts on the reply, even its action accuracy after a reply.
**Each published run set used a different answerer model. Numbers from different run sets
must never be put in one table or chart, compared directly, or averaged.**

| run set | answerer (served name → weights) | ask judge (served name → weights) | config |
|---|---|---|---|
| frontier | `llama3.2-90b-vision-instruct` → Llama-3.2-90B-Vision-Instruct | the same model and name | `frontier_llama90.yaml` |
| open-weight, GLM answerer | `glm_4_6v` → GLM-4.6V (`llama32_90b_vision` for the two GLM policies) | `judge70b` | `open_weight_glm46v.yaml` |
| open-weight, earlier | `answerer32b` → Qwen3-VL-32B-Instruct (`answerer27b` → Gemma-3-27B for the Qwen3-VL-32B policy) | `judge70b` (`judge27b` for the two Llama-3.3-70B policies) | `open_weight_qwen3vl32b.yaml` |

A model never answers or judges for itself: where a policy is the run set's own answerer or
judge model, that one role is replaced for that policy only (the `per_policy` blocks in the
config files). Those passes' ask numbers are not comparable with the rest of their set.

## Install

Python 3.12 (as `pyproject.toml` requires). From the repository root:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements/offline.lock     # the exact versions this release was checked with
pip install --no-deps -e .                   # or: pip install -e ".[offline]" for unpinned versions
```

`requirements/offline.lock` pins every package. The `[offline]` extra in `pyproject.toml`
lists only the direct dependencies (`requests`, `PyYAML`, `pillow`, `openai`).

With Docker:

```bash
docker build --build-arg AMBIGUI_GIT_COMMIT="$(git rev-parse HEAD)" -f docker/offline.Dockerfile -t ambigui-offline .
docker run --rm ambigui-offline --help
```

The image holds the code and its packages only. Models are reached over HTTP, and the
benchmark, screenshots and outputs are mounted as volumes (see the header of `docker/offline.Dockerfile`).

### The models are served separately

Every model — policy, answerer and judge — is reached through an *OpenAI-compatible
endpoint*: an HTTP server that answers the OpenAI chat-completions API at a base URL ending
in `/v1`. Two routes work:

1. **Serve Hugging Face weights yourself** with vLLM (or any compatible server), at the
   pinned revision and under the served name in `configs/offline/models.yaml`, for example:

   ```bash
   vllm serve Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
       --served-model-name qwen3-vl-8b-instruct --port 8000 --tensor-parallel-size 1
   ```

   For a policy that sees screenshots, the server must accept as many images per request as
   the run sends (`--limit-mm-per-prompt '{"image": 2}'` in recent vLLM; the run checks this
   against `serve_image_limit` before it starts).
2. **Use a hosted OpenAI-compatible API.** Give its base URL as the endpoint and put its
   key in the environment variable for that role. A key is never written into a config
   file or passed on the command line.

| role | base URL | key (optional; unset = no key, as a local vLLM server expects) |
|---|---|---|
| policy | `--endpoint` or `AMBIGUI_POLICY_BASE_URL` | `AMBIGUI_POLICY_API_KEY` |
| answerer | `answerer.endpoint` in the run-set file (default `AMBIGUI_ANSWERER_BASE_URL`) | `AMBIGUI_ANSWERER_API_KEY` |
| judge | `judge.endpoint` in the run-set file (default `AMBIGUI_JUDGE_BASE_URL`) | `AMBIGUI_JUDGE_API_KEY` |

**The served name matters.** Each server must answer to the exact name in
`configs/offline/models.yaml` (`--served-model-name`). For a policy that is its catalogue
key. For the answerer it is the name the run set recorded, because the answerer picks its
output-token budget from that name: `glm_4_6v` gets 8192 tokens (GLM-4.6V reasons before it
answers), while names not in its table, such as `answerer32b` and
`llama3.2-90b-vision-instruct`, get the default 1500. Serving the same weights under another
name changes the answerer's budget and so its replies.

HTTP calls to the policy, the judge and the simulated user, and every start-up check,
ignore proxy settings in the environment (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`), so the
check and the real calls take the same route. Run where the endpoints are directly
reachable. With no key set, no `Authorization` header is sent, for any role.

### Environment variables

| variable | used by | meaning |
|---|---|---|
| `AMBIGUI_BENCHMARK` | run-set files | the gold evaluation file (`gold_eval_set.jsonl`) |
| `AMBIGUI_IMAGE_ROOT` | run-set files | the folder the steps' `image_path` values are relative to |
| `AMBIGUI_OUT_ROOT` | run-set files | where run directories are written (default `runs`) |
| `AMBIGUI_POLICY_BASE_URL` | `launch run` | the policy endpoint when `--endpoint` is not given |
| `AMBIGUI_ANSWERER_BASE_URL`, `AMBIGUI_JUDGE_BASE_URL` | run-set files | the simulated user's and the ask judge's endpoints |
| `AMBIGUI_ALT_ANSWERER_BASE_URL`, `AMBIGUI_ALT_JUDGE_BASE_URL` | open-weight run-set files | the replacement simulated user or judge for a policy that is the run set's own answerer or judge model |
| `AMBIGUI_POLICY_API_KEY`, `AMBIGUI_ANSWERER_API_KEY`, `AMBIGUI_JUDGE_API_KEY` | the three stages | keys for hosted endpoints; never written to the command line or `run.json` |
| `AMBIGUI_ANSWERER_HTTP_TIMEOUT_S` | simulated user | seconds allowed for one simulated-user HTTP call (default 60) |
| `AMBIGUI_GIT_COMMIT` | `run_manifest.json` | the commit to record where there is no `.git` folder (the Docker image) |

### The benchmark

The policy is posed the AmbiGUI gold evaluation file, `gold_eval_set.jsonl` (298 tasks:
143 *Ambiguous* tasks, where asking is the right behaviour, and 155 *Clear* tasks, where
acting is). The file the benchmark build writes has 3,115 recorded steps (Mind2Web 1,032,
META-GUI 2,083, OSWorld 0); the recorded runs were made on an internal copy with 3,248
steps. It is rebuilt from your own copy of the public source
datasets by the benchmark part of this release (`benchmark/`). Each step's `image_path` is
relative to an image root folder that you pass to the run. A frame that cannot be found
under it stops the run.

**Getting the screenshots.** The benchmark build writes the task rows only; it ships no
screenshots. You place each frame under the image root yourself, at the step's
`image_path`, using the step pointers in `benchmark/manifest/pointers.jsonl`:

| `source` | `image_path` | where the frame comes from |
|---|---|---|
| `meta-gui` | `meta-gui/screenshots/<sample_id>/step_NN.png` | META-GUI's public `test.tar`: the file `<dialog_dir>/<turn_dir>/<action_key>.png`, with `dialog_dir` from the row's `native` block and `turn_dir`, `action_key` from step NN of its `steps` list |
| `mind2web` | `mind2web/screenshots/<sample_id>/step_NN.png` | Mind2Web: the `screenshot` of the row whose `annotation_id` is the row's `native.annotation_id` and whose `action_uid` is step NN's `action_uid` |
| `osworld` | (none) | OSWorld rows have no recorded steps, so the offline evaluation has nothing to pose and skips them |

The run checks that the frame of every step it poses exists, even with `--screens off`
(the frames are recorded in `substrate.jsonl`). With `--limit N` only the first N steps are
posed, and only their frames are needed.

Set, for every command below:

```bash
export AMBIGUI_BENCHMARK=/path/to/gold_eval_set.jsonl
export AMBIGUI_IMAGE_ROOT=/path/to/screenshots
export AMBIGUI_OUT_ROOT=$PWD/runs          # optional; this is the default
```

## Run inference

Start the policy, answerer and judge servers, then:

```bash
CFG=configs/offline/run_sets/open_weight_glm46v.yaml     # or open_weight_qwen3vl32b.yaml
python -m ambigui.offline_eval.launch run --config $CFG \
    --model qwen3-vl-8b-instruct --screens on --endpoint http://localhost:8000/v1
```

- `--model` is a catalogue key; `--served-model` defaults to the same name.
- `--screens on|off` chooses whether screenshots are attached (*screens-on*: the policy
  sees the screen; *screens-off*: text only). Both are passes of the same benchmark.
- The output goes to `$AMBIGUI_OUT_ROOT/<run_set>/<model>/screens-<on|off>/`.
- `--limit N` poses only the first N steps (in the order task source, task id, step).
- The launcher prints the exact flags of each stage before it runs. The same flags can be
  given to `python -m offline_evaluation.run.run_ask_protocol` directly (with
  `PYTHONPATH=ambigui/offline_eval/src`).

Before any work, the run sends one real request to every server it depends on and stops
if any is down, unreachable, or serving under another name. Decoding is greedy
(temperature 0). The fixed seed (`seed` in the run-set file, default 0) is sent to the
policy and the judge only; the simulated user sends no seed.

**Failed policy calls.** A policy call that fails (HTTP 429 or 5xx, timeout, unreadable
body) is not retried: the step gets an empty reply and is scored wrong. `report.json`
counts these steps as `calls.n_errored`. No retry is made because the recorded runs made
none, and a retry would change behaviour against them.

**Output budget.** The policy's output-token budget is the catalogue entry's
`max_tokens`, unless the run-set file lists the pass in `policy_max_tokens`; the launcher
then passes that value as `--max-tokens`, and `run.json` records it
(`max_tokens_source`). The open-weight files list the budget every recorded pass used.

Each stage also writes `run_manifest.json` beside its outputs: the git commit (and whether
the checkout had uncommitted changes), the run-set file's SHA-256, the model ids and
revisions, package versions, the seed, the host, and the exact flags of every stage. No
stage reads it.

### Smoke check

Add `--limit 30` to the `run` command above to pose only the first 30 steps, then judge and
score as usual. It proves the servers and paths are right; it is not a paper result.

## Judge and score

```bash
python -m ambigui.offline_eval.launch judge --config $CFG --model qwen3-vl-8b-instruct --screens on
python -m ambigui.offline_eval.launch score --config $CFG --model qwen3-vl-8b-instruct --screens on
```

The judge stage rewrites `asks.jsonl` with each question's verdict. It writes nothing
until every question has a verdict, so an outage leaves the directory as it was. The judge
is always posed text only. Its token budget comes from the catalogue entry named by
`judge.registry_key` in the run-set file.

The score stage needs no model and no network. It refuses a run that is not complete, has
unjudged questions, or was not *staged* (see "Dialogue gating" below). It prints a summary
and writes `report.json`.

## Reproduce the offline paper results

### Frontier run set (Claude, GPT, Grok, Gemini, Muse Spark 1.3)

Policies: five frontier models through their providers' APIs. Answerer and judge:
Llama-3.2-90B-Vision-Instruct through a hosted OpenAI-compatible API, served as
`llama3.2-90b-vision-instruct`, text only, temperature 0, 1500 output tokens. All five
passes cover the same 298 tasks / 3,248 recorded steps (the internal copy).

**Exact reproduction of the scores (no model calls).** The recorded per-model run
directories (recorded inference plus judge verdicts) are not shipped with this repository.
Given them, the score stage re-scores them exactly:

```bash
for m in <the five pass names>; do
  python -m ambigui.offline_eval.launch score \
      --config configs/offline/run_sets/frontier_llama90.yaml --model $m --screens on
done
```

with each pass's `substrate.jsonl`, `turns.jsonl`, `asks.jsonl` (judged), `inference.jsonl`
and `run.json` placed in `$AMBIGUI_OUT_ROOT/frontier_llama90/<model>/screens-on/`. The
resulting `report.json`, `outcomes.jsonl` and `emissions.jsonl` are byte-identical to the
published ones (checked for all five models before release).

**Re-judging** the recorded questions needs the Llama-3.2-90B judge endpoint
(`AMBIGUI_JUDGE_BASE_URL`, `AMBIGUI_JUDGE_API_KEY`) and replaces the recorded verdicts
with new ones; see "Known differences".

**Re-running inference** needs each provider's OpenAI-compatible endpoint and key
(`AMBIGUI_POLICY_*`) and the answerer endpoint. The catalogue has an entry for four of the
five frontier policies (`claude-5-1-fable`, `gpt-6-1-sol`, `grok-4-20-reasoning`,
`gemini-3-8-flash`), under the names the recorded run used, with the recorded settings:
Claude 1024 output tokens; the other three send no output-token limit (`max_tokens: null`,
the provider's default, as recorded); screenshots at their original size
(`max_image_dim: null`). Pass the provider's own model name with `--served-model`. The fifth
policy, Muse Spark 1.3, is an internal model and has no entry. A re-run is not expected to reproduce the
published emissions exactly; see "Known differences".

### Corrected Clear-row view (the primary offline view of the paper)

The paper's primary offline table uses a corrected view of the frontier runs. In the
recorded runs, the simulated user answered questions on Clear tasks that have no curated
open item with a reply that invented a preference. For 88 such Clear tasks, the part of
each run after the policy's first question (the "suffix") was regenerated with the later
simulated-user behaviour (the statement check and the nothing-open reply) and spliced back
onto the unchanged recorded start of the run (the "prefix"). 83 suffixes were regenerated
(Claude: 10, GPT: 19, Grok: 39, Gemini: 7, Muse Spark 1.3: 8); five Claude cases keep their
original suffix because the exact screenshots they were sent are missing. Every other task,
every Ambiguous record and every first question is unchanged.

With this release only the re-scoring of such a corrected run is possible (`launch score`
on the corrected run directories, which are not shipped). The suffix generator and the
splicer are not released: they were a one-off repair run outside this code base, and their
exact settings were not kept in a form that can be released. Re-running inference with
this code produces a new, uniformly current run, not this corrected view.

The partial open-weight snapshot the paper reports as a frozen, descriptive table is not
reproducible with this release either, because the statement-gated runner that produced it
is not released.

### Open-weight run sets

Policies: the open-weight models in the catalogue (Qwen-VL / Qwen3.x, Gemma, InternVL,
UI-TARS, ...), each served with vLLM from `configs/offline/models.yaml`. For each model and
each screen condition:

```bash
CFG=configs/offline/run_sets/open_weight_glm46v.yaml     # or open_weight_qwen3vl32b.yaml
for screens in on off; do
  python -m ambigui.offline_eval.launch run   --config $CFG --model <key> --screens $screens --endpoint http://localhost:8000/v1
  python -m ambigui.offline_eval.launch judge --config $CFG --model <key> --screens $screens
  python -m ambigui.offline_eval.launch score --config $CFG --model <key> --screens $screens
done
```

`vision: false` marks a text-only checkpoint: `llama-3.3-70b-instruct`,
`llama-3.3-70b-instruct-awq`, `minimax-m2.5` and `qwen-agentworld-35b-a3b`. The run refuses
`--screens on` for them at start-up. `agentcpm-gui` and `uix-qwen2-mind2web` did not load
in the serving stack used for the recorded runs. `muse-glimmer-30b` (two passes of the
GLM-answerer run set) is an internal model: it has no catalogue entry and is **internal,
not reproducible**.

**Re-scoring the recorded open-weight runs is not exact.** The open-weight run sets were
scored with an earlier credit rule that allowed look-ahead credit across a change of
request. This release's scorer gives look-ahead credit only within the same request, and
refuses those recorded runs: GLM-4.6V answerer 62 of 64 passes, the earlier ("pre") set
60 of 62, the "post" set 24 of 26. Those tables therefore cannot be re-scored exactly
with this release. The published numbers are unchanged.

### Expected cost

Scoring one recorded pass takes a few seconds on one CPU core. A run makes one policy call
per recorded step (3,115 per pass with the rebuilt file; the recorded passes had 3,248), plus
one more per question asked, plus two simulated-user calls per question (the statement
gate, then the answer; each retried up to twice inside the client). Its GPU needs are
those of serving the model (tensor-parallel sizes in `configs/offline/models.yaml`).
Judging makes one judge call per question while the task still has unspent open items;
later questions get no call.

## Metric definitions

All metrics are computed by the score stage from the recorded files.

### Which steps and items are used

- **Dialogue gating (`visible_from_step`).** Some tasks are conversations: the person's
  request arrives over several turns. A turn is shown to the policy only from the step given
  by its `visible_from_step`, so the policy is never shown a request the person had not made
  yet. A pass where every step was posed this way is *staged*; the score stage refuses to
  report a pass that is not.
- **Clarifying turns are removed for the ask condition.** Turns whose role is
  `clarifying_question` or `clarifying_answer` are never shown. They are the recording's own
  question and answer — exactly what the policy is being tested on — so showing them would
  hand the policy the answer.
- **Open items only.** Asking is graded against each task's `open_items` (313 items on 141
  tasks). The file's other list, `items_from_decision_points`, is never read and never
  merged with `open_items`: it has no answers a simulated user can give, and on the rows
  carrying both, the two lists name different details.
- **Every step with a recorded action is scored.** A step the run never reached (after a
  cut-off) counts as wrong, not as missing. Steps whose recorded action has no equivalent in
  the policy's action format (for example a key press) are *inexpressible* and reported as a
  count, not scored.

### Acting (`act` in `report.json`)

- `operation_match`: the action type (click, type, select, ...) matches the recording.
- `target_match_strict` / `target_match_tolerant`: the named element matches the recorded
  one exactly (ignoring case and spacing) / loosely (at least half of the recorded
  element's content words). Over steps whose recording names an element. These and
  `operation_match` read the policy's **first** candidate only; step success uses any
  candidate.
- `text_match`: the typed or selected text matches, ignoring case and spacing. Over steps
  that record one.
- `step_success_own_step`: the right action on the right element with the right text,
  checked against that step's own recorded action. **This is the headline action metric.**
  Credit is spend-once: each recorded action can be credited once, so a correct own-step
  action earns nothing if an earlier step's alternative already spent that action.
- `step_success_with_lookahead`: the same, but a step is also credited when one of the
  policy's alternative actions matches a recorded action later in the trajectory, only
  within the same request (the same staged instruction). `n_credited_ahead` counts those
  steps. Always report it beside the own-step number.
- `calls.n_errored`: steps whose policy call failed (scored wrong). `statement_gate`:
  the simulated user's statement-gate calls, with `n_errored` (the call failed) and
  `n_unreadable` (no verdict); both are treated as questions. Both blocks appear for runs
  made by this release; a recorded run keeps its recorded report layout.

### Asking (`ask_behaviour`, `ask_content`)

- `n_tasks_that_asked`, `n_questions`, `mean_asks_per_task`, `mean_calls_per_step`.
- `n_overask_tasks`: *Clear* tasks on which the policy asked anyway.
- `n_not_a_request`: questions the judge said were not a request for information at all
  (a report, a refusal, an offer of help). These earn nothing.
- `n_tasks_cut_off`, `n_unreached_steps`: tasks stopped after ten questions in a row, and
  the steps they left unscored-as-wrong.
- Ask content, four rules from loosest to strictest. A task counts as correct (a *true
  positive*) when:
  - `any` — the policy asked anything on it;
  - `at_least_one` — its questions covered at least one of its open items;
  - `full_set` — they covered all of its open items;
  - `full_set_clean` — they covered all of them and no question covered none.

  For each rule: recall = true positives / *Ambiguous* tasks that have open items
  (*gradeable* tasks); precision = true positives / tasks the policy asked on (gradeable
  ones plus *Clear* ones, so over-asking lowers precision); F1 is their harmonic mean.
  `ask_*_any` measures only *whether* the policy asked, so it is an ask rate, not a measure
  of question quality; prefer `at_least_one`, `full_set` and `full_set_clean`.

### Timing (`ask_timing`)

Each open item has a window whose `latest_step` is the last step at which asking still
helps (one step before the recorded action that fixed the value). `asked_in_time` is the
fraction of items whose question came by then.

- Only items with `never_committed == false` are timed. When `never_committed` is true the
  recording never fixed the value, so the window runs to the end and "in time" would be
  trivially true; these items are counted (`n_never_committed`) and excluded.
- Windows come from two sources, reported separately and never averaged: `recorded`
  (written by hand, n = 12) and `proposed` (proposed by a model reading the recording).

### Using the answer (`answer_use`)

The fraction of credited answers that appear (case-insensitive substring) in an action the
policy emitted at or after the step it asked. Reported beside the ask scores, not folded
into them.

## Versions used in the reported runs

Filled from recorded evidence only (each run's `run.json` and `report.json`, and the recorded
summaries of the open-weight run sets). "Not recorded" means no file of the run states it.

| component | release pin | frontier run set (original and corrected Clear-row views) | open-weight run sets (GLM-4.6V and Qwen3-VL-32B answerers) |
|---|---|---|---|
| vLLM | not pinned by this part (models are served separately) | none: every policy, the simulated user and the judge were hosted APIs | not recorded |
| transformers, torch, CUDA | not pinned by this part | not applicable (hosted) | not recorded |
| OpenAI client | `openai==3.35.1` (requirements/offline.lock) | not recorded (the provider adapters are not in the release) | not recorded |
| request settings that affect outputs | temperature 0, seed 0 (policy and judge), 10 asks per turn, 1 frame per step, `max_tokens` from the catalogue or the run set's `policy_max_tokens`, `max_image_dim` from the catalogue | not recorded in run.json; provider emissions show Claude `max_output_tokens` 1024 (one call 4096), Gemini none (three calls 16384), GPT, Grok and Muse Spark 1.3 none, no temperature or seed sent, original PNG frames | recorded in run.json: temperature 0, seed 0, `max_asks` 10, `frame_cap` 1, per-pass `max_tokens` (GLM set: 512 x26, 2048 x8, 4096 x11, 8192 x19 over 64 passes, 30 differing from the catalogue; earlier sets: the catalogue values; all listed in the run-set files), `max_image_dim` 768, `serve_image_limit` 2 (screens on) or 0 (off) |
| serving flags | — | not applicable | not recorded (the grid table records tensor-parallel size per model only) |

## Known differences from the paper runs

- **Answerer per run set.** Each run set used a different answerer (table above). Never
  mix run sets.
- **Hugging Face revisions.** The recorded open-weight runs loaded local copies of the
  weights, and the commit of each copy was not recorded. `configs/offline/models.yaml` pins
  the Hugging Face commit current on 2026-10-23. Where a repository changed after the
  recorded runs, results can differ.
- **Inferred judge weights.** The open-weight run files record the judge only by served
  name (`judge70b`, `judge27b`) and do not record which catalogue entry set the judge's
  token budget. The weights and budgets in the configs are inferred and marked as such.
- **Decoding is not bit-reproducible.** Even at temperature 0 with a fixed seed, a served
  model can produce different text on a re-run (batching, hardware, server version).
  Expect small differences when re-running inference or judging. Scoring recorded files is
  exact.
- **Frontier inference.** The recorded frontier runs reached each provider's Responses API
  through adapters outside this code, with per-provider output budgets (Claude 1024 tokens
  with one retry at 4096; GPT, Grok and Muse Spark 1.3 at the provider default; Gemini at the default
  with three retries at 16,384), no temperature and no seed, and each screenshot sent as its
  original file (JPEG-capped above 3.8 MB). This code sends a Chat Completions request with
  the catalogue entry's budget (none when it is `null`), temperature 0 and a seed, and
  re-encodes each frame, and it does not repeat the adapter's retries. A re-run is a new
  measurement, not a replay.
- **Screenshots.** The META-GUI frames in META-GUI's public `test.tar` are byte-identical to
  the frames the recorded frontier runs sent (all 2,083 META-GUI steps checked against the
  file hashes those runs recorded). The Mind2Web frames of the recorded runs were PNG files
  prepared from the dataset; Mind2Web ships each screenshot as a JPEG, so the bytes
  differ, and pixel identity could not be checked. Screens-on numbers on Mind2Web tasks can
  therefore differ from the recorded ones. The open-weight runs read a copy of the frames
  whose hashes were not recorded.
- **Answerer prompt and nothing-open tasks.** This release ships the later wording of the
  answerer's open-items prompt (`rubrics/reasoning_decide/shared/answer_from_open_items.md`)
  with the statement check before the answerer (`evaluation/message_kind.py`). This
  release's offline runner never tells the answerer that a task left nothing open: a
  question on a task with no open items is answered from the instruction alone
  (`answer_user_intent.md`). The recorded run sets differ:

  | run set | open-items prompt | statement check | question on a task with no open items |
  |---|---|---|---|
  | frontier (original view) | earlier wording | no | answered from the instruction alone |
  | GLM-4.6V answerer | later wording (shipped) | not recorded | the runner passed the nothing-open signal, so the answerer said there was no preference to state |
  | Qwen3-VL-32B answerer | not recorded | not recorded | the earlier Clear-row reply, which could invent a preference |

  Re-running inference is therefore a new measurement for every run set. Re-scoring is
  exact for the frontier run set only (next item).
- **Open-weight scoring version.** The open-weight run sets were scored with an earlier
  credit rule that allowed look-ahead credit across a change of request. This release's
  scorer allows it only within the same request and refuses those recorded runs (GLM 62
  of 64, earlier set 60 of 62, post set 24 of 26), so their tables cannot be re-scored
  exactly with this release. Their reports also hold three counters this release does not
  write for them (`calls.n_errored`, `n_questions_nothing_open`,
  `n_questions_unsupported`).
- **What was replaced.** Besides the links to the authors' infrastructure, the release
  removed the `--max-asks`, `--frame-cap`, `--workers`, `--timeout` and `--screens auto`
  flags (fixed at the recorded values), spare serves, and the judge-budget fallback.
- **Model ids in `run.json`.** `hf_path` is now the Hugging Face id, so a re-run's
  `run.json` differs from the recorded string.
