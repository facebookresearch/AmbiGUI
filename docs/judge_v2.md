# Online task judge v2 (intent-aware)

The online task judge decides whether a rollout completed the task. Version 1 (`judge_v1`) reads the
instruction, the actions and the screenshots. It does not see the values the user actually wanted
for the details the instruction leaves open, so it can pass a complete answer for the wrong value
(for example, the weather for the wrong city).

Version 2 (`judge_v2_with_intent/2.0.3`) also receives the task's curated `open_items` (each
item's `name`, `question` and `answer`) and checks the final outcome against every one of them:

- a shown mismatch with an intended value is a **failure**;
- if the intended values, the screenshots or a usable judge reply are missing, the verdict is
  **unresolved** (`success: null`), not a failure;
- `items_from_decision_points` and window metadata never enter the judge prompt.

The judge's rationale is saved with every verdict (`judge_rationale`), together with the version
tag (`judge_version`) and a journal of every judge request and response.

## Which version is used where

| caller | version |
|---|---|
| online evaluation (`ambigui/online_eval/src/scripts/eval_cascp_online.py`) | v1 by default; v2 when `--judge-version judge_v2_with_intent/2.0.3` is given |
| judge preflight (`ambigui/online_eval/harness/preflight_roles.py`) | the same flags; v2 checks the task-judge route only |

The paper's online task-success scores are being re-graded with v2. Once that re-grade is
complete, v2 is the primary task-success verdict and the v1 verdicts are kept as an archived
version. Unresolved v2 verdicts are left out of success rates, and
their count is reported next to each rate.

## Running online evaluation with v2

Keep every other setting the same and add:

```sh
--judge-version judge_v2_with_intent/2.0.3 \
--curated-sidecar /abs/path/ambigui_online_benchmark.jsonl \
--judge-config   /abs/path/judge_v2.json \
--judge-journal-root /abs/path/judge_journals
```

- `--curated-sidecar`: rows with an exact `sample_id` (or `task_id`), the exact `instruction`
  and the curated `open_items`. The online file built from `benchmark/` already has this shape:
  its `open_items` holds only the curated intended values.
- `--judge-config`: start from `configs/online/judge_v2.json`. Set the judge and describer
  endpoints and your serving backend. Both roles use Qwen3-VL-235B-A22B-Instruct with reasoning
  off. The `--judge-url`/`--judge-model` and describer flags must match the config exactly; the
  judge never substitutes another model or endpoint and never retries on its own.
- `--judge-journal-root`: a folder for the request/response journal (each verdict gets its own
  new subfolder).
- Served name: the config's `model` must equal `--judge-model` (and `--describer-model`)
  exactly. It may be written in any letter case, for example `qwen3-vl-235b-a22b-instruct`
  as in `run_online.sh`, but it must name Qwen3-VL-235B-A22B-Instruct.
- API key: the v2 judge sends the same `Authorization` header as v1, chosen by
  `evaluation/endpoint_auth.py` (`JUDGE_API_KEY`, `DESCRIBER_API_KEY`, else `AMBIGUI_API_KEY`;
  no header when none is set). The key is never written to the journal.
- Each v2 judge request has a 120-second timeout and is not retried. A request that fails or
  times out makes that verdict unresolved.
- Only rollouts that v1 would judge go to the v2 judge. Rows that are not judged
  (unreachable or blocked start page, harness error) keep the v1 outcome. An Ambiguous task with
  no curated open item is unresolved, with the reason "no curated intent".
- Relative dates are resolved from `task_time_evidence`, the UTC date at which the rollout
  started (recorded in each result row).

The v2 judge uses the same limits as v1: 1200 tokens (lookup), 600 (trace) and 600 (describer), at most 10 frames chosen by the v1
frame selector, frames sent as JPEG (width 1000, quality 80), and the full action history. A
missing or unreadable frame makes the verdict unresolved; no frame is silently dropped. Relative
dates (today, tomorrow) are only resolved from the date recorded with the rollout, never from the
date the judge runs. Like v1, the judge receives the action trace without per-action failure
marks and decides from the outcome shown in the screenshots.

## Re-grading stored rollouts

The paper's v2 verdicts were produced by re-running only the task judge on the stored trajectories
(same instruction, screenshots and actions; no new rollouts). The trajectories themselves are not
released, so that re-grade cannot be repeated from this repository; new runs use the flags above.
