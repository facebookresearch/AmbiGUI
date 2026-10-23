# AmbiGUI benchmark — data card

This card describes the four files that `build.py` writes. The repository ships them as pointers
and our own labels (`annotations/`, `manifest/`); you rebuild the full files from your own copy
of the source datasets (see [README.md](README.md)).

| file | rows | one row is |
|---|---:|---|
| `gold_eval_set.jsonl` | 298 | one task, in the form the offline evaluation reads |
| `gold_benchmark_with_targets.jsonl` | 298 | the same task, with the full annotator record (`gold`) and per-step action targets |
| `open_items.windows.jsonl` | 313 | one open item: its proposed ask window |
| `ambigui_online_benchmark.jsonl` | 84 | one task for the online (live browser) evaluation |

Gold rows: 143 Ambiguous / 155 Clear; 150 Mind2Web, 115 META-GUI, 33 OSWorld; 313 open items on
141 rows (all Ambiguous); 3,115 recorded steps (1,032 Mind2Web, 2,083 META-GUI, none for OSWorld).
Online rows: 45 Mind2Web (37 also gold rows, 8 not) and 39 META-GUI (all gold rows); 42
Ambiguous / 42 Clear.

## Producer classes

The last column of each table says who or what made the field. The full record, with evidence,
is in `manifest/label_provenance.yaml`.

| class | meaning |
|---|---|
| Human | human research participants in the annotation study (final labels only); for the 12 recorded ask windows, the owner and the annotators |
| Code | deterministic code over the human labels or the source values; no model |
| Gemma | Gemma-4-31B (approved producer) |
| Owner | annotated by the dataset owner |
| Source | a source value, rebuilt from your copy of the source by `build.py`; not shipped in the repository |
| Pointer | an id or index into the source |

Types: `str`, `int`, `bool`, `obj` (JSON object), `list`, `null`. "S" in a type means the value
is a string that holds a number or a boolean (`"3"`, `"False"`).

## `gold_eval_set.jsonl`

| field | type | meaning | producer |
|---|---|---|---|
| `sample_id` | str | source-native task id: Mind2Web `annotation_id`, META-GUI `dialog_id`, OSWorld task id | Pointer |
| `ambiguity` | str | `Ambiguous` (the agent should ask) or `Clear` (it should act) | Human |
| `ambiguity_type` | str | main type: `Parametric`, `Referential`, `Intent`, `Cross-modal`; `""` on 6 rows | Human |
| `all_types` | str | all types, comma-separated | Human |
| `severity` | str | cost of a wrong guess: `minor`, `major`, `critical` | Human |
| `risk` | str | `Reversible` or `Irreversible` | Human |
| `source` | str | `mind2web`, `meta-gui`, `osworld` | Pointer |
| `substrate` | str | `web`, `mobile`, `desktop` (one per source) | Code |
| `domain` | str | Mind2Web domain, META-GUI category, OSWorld domain | Source |
| `split` | str | Mind2Web `test_task` / `test_website` / `test_domain`; META-GUI `test`; OSWorld `cross_domain` (our tag) | Pointer |
| `n_steps` | str / null | number of recorded steps (a string here, an int in the targets file); `null` on OSWorld rows | Code |
| `site` | str | site or app name in our wording (`<website> / <subdomain>`, `META-GUI app: <category>`, `OSWorld app: <domain>`) | Source + Code |
| `instruction` | str | the task text, verbatim from the source (Mind2Web `confirmed_task`, a META-GUI user turn, the OSWorld instruction) | Source |
| `annotator_notes` | obj | annotator free text: `Looking back at the end` (`Completion strategy`, `Decisions supported`, `Options unresolved`, `Step`) on all rows; `While annotating` (`Edge cases`, `Process uncertainty`, `Fatigue`) on 87 rows | Human |
| `ask_by` | obj / null | the step by which the row's open items had to be asked about: one step before the row's first commitment. `null` on 136 rows (no commitment recorded) | Code |
| `ask_by.step` | int | that step; `-1` if the first commitment is the first action | Code |
| `ask_by.earliest_recorded_ask_before_step` | int / null | the earliest "ask before step" an annotator wrote on any decision of the row | Code |
| `ask_by.no_recorded_deadline_is_earlier` | bool | true if even that recorded deadline comes after `ask_by.step` | Code |
| `ask_by.first_commitment` | obj | the commitment that sets `ask_by`: `kind` (`decision`), `id` (decision id), `step` | Code |
| `open_items` | list | the details the task leaves open, with the answer the user would give. **The list to use**: the simulated user answers from it and the ask judge grades against it. Non-empty only on Ambiguous rows | Gemma |
| `open_items[].name` | str | short name of the item (`budget`, `location`) | Gemma |
| `open_items[].question` | str | a clarifying question about the item | Gemma |
| `open_items[].answer` | str | the value the user wants (88 answers are verbatim source spans) | Gemma |
| `open_items[].window` | obj | the ask window: when a question about the item is still useful (see "Ask windows" below) | Gemma (proposed, 301) / Human (recorded, 12) |
| `open_items[].window.latest_step` | int | the last step at which asking still helps (one step before the action that fixes the value) | Gemma / Human |
| `open_items[].window.source` | str | `proposed` (301 items, by Gemma-4-31B) or `recorded` (12 items, written by hand by the owner and annotators) | Code |
| `open_items[].window.never_committed` | bool | true if nothing in the recording fixes the value; the window then runs to the last step. Always false when `source` is `recorded` | Gemma / Human |
| `open_items[].window.basis` | str | the evidence for a proposed window; `""` when `source` is `recorded` | Gemma |
| `open_items[].detail` | obj | always `{}` (same shape as `items_from_decision_points`) | Code |
| `items_from_decision_points` | list | items read off the annotators' decision records. Supplementary: **never merge them into `open_items`** | Code over Human |
| `items_from_decision_points[].name` | str | `<decision id> · <type>` | Code |
| `items_from_decision_points[].question`, `.answer` | str | the annotator's decision question and best assumption | Human |
| `items_from_decision_points[].window` | obj | `earliest_step`, `latest_step` (one step before the annotator's commit step), `source` (`derived`), `never_committed`, `closed_before_the_run` (value fixed by the first action), `opens_at_a_recorded_step` (false when `earliest_step` is a default 0), `basis` (how it was computed) | Code |
| `items_from_decision_points[].detail` | obj | the annotator's text for that decision: `Options open`, `Why that assumption`, `Ask or guess`, `How it was resolved`, `What the agent did`, `Clarifying question` | Human |
| `decisions` | list | the annotators' decision points (places where the task left a choice open) | Human + Code |
| `decisions[].id`, `.question`, `.classification` | str | decision id (`d3`), its question, its ambiguity type | Human |
| `decisions[].earliest_step` | int / null | first step at which the choice is open (from the annotator's root-cause record) | Human |
| `decisions[].commits_at_step` | int | step at which the recording commits to an option | Human |
| `decisions[].recorded_ask_before_step` | int / null | the annotator's own "ask before step", where given | Human |
| `decisions[].derived_ask_by_step` | int | `commits_at_step - 1` | Code |
| `decisions[].interval_runs_backwards` | bool | `earliest_step` is after `commits_at_step` | Code |
| `decisions[].ask_by_is_after_the_commit` | bool | the annotator's recorded ask-before step is at or after the commit step (too late for this decision) | Code |
| `decisions[].<annotator fields>` | str | annotator text: `Ask or guess`, `Best assumption`, `Why that assumption`, `Random guess`, `How it was resolved`, `Options open`, `Steps it spans (as written)`, `Option locked in`, `Cost of the others`, `Why that is the risk`, `Evidence on screen`, `Clarifying question` | Human |
| `turns` | list | the META-GUI dialogue (`[]` on Mind2Web and OSWorld rows) | Source + Human + Code |
| `turns[].turn` | int | index into the source `dialog.json` | Pointer |
| `turns[].speaker` | str | `user` or `system` (from the source) | Source |
| `turns[].role` | str | `task`, `clarifying_question` (a system turn ending in `?`), `clarifying_answer`, `followup`, `reply`; a fixed punctuation rule | Code |
| `turns[].text` | str | the turn text, from the source | Source |
| `turns[].classification` | str | annotator class of a user turn: `Original instruction`, `Later task instruction`, `User clarification`, `Task-irrelevant turn`; `""` on system turns | Human |
| `turns[].visible_from_step` | int / null | the first step at which the turn may be shown to the agent. Show a turn only from this step on; for the ask condition, also drop `clarifying_question` / `clarifying_answer` turns | Code |
| `steps` | list | the recorded trajectory (`[]` on OSWorld rows) | Source + Code |
| `steps[].step_index` | int | our step index; `manifest/pointers.jsonl` maps it to the source step | Pointer |
| `steps[].turn_index` | int / null | position of the META-GUI `turn_<n>` folder; `null` for Mind2Web | Pointer |
| `steps[].action` | str | our rendering of the recorded action: `<op> "<value>" into "<element>"` | Source + Code |
| `steps[].element_source` | str | where the element name came from (`screen text`, `accessibility description`, `view identifier`, `recorded by the ingest` = the element text Mind2Web records, ...) | Code |
| `steps[].image_path` | str | relative screenshot path pattern `<source>/screenshots/<sample_id>/step_NN.png`, prefixed by `--pool-root`. Screenshots are not shipped | Code |
| `steps[].annotations` | list | copies of the annotator records anchored to this step (all values are strings). `kind` is `step_discovery`, `decisions`, `step_tracking`, `compounding_roots`, `assumptions` or `cross_modal_gaps`; the other keys are the fields of that record in `gold` (targets file) | Human |
| `flags` | list of str | row-level warnings computed from the labels (for example "No recorded ask-by step is before the row's first commitment"; "Ambiguous, but no gold-half open items" = an Ambiguous row with an empty `open_items`) | Code |

## `gold_benchmark_with_targets.jsonl`

Same tasks as `gold_eval_set.jsonl`, keyed by `sample_id`.

| field | type | meaning | producer |
|---|---|---|---|
| `sample_id`, `ambiguity`, `severity`, `risk`, `source`, `substrate`, `domain`, `split`, `instruction` | | as in `gold_eval_set.jsonl` | |
| `ambiguity_type` | str / null | as in `gold_eval_set.jsonl`, but `null` where that file has `""` | Human |
| `ambiguity_types_all` | list of str | all ambiguity types | Human |
| `benchmark` | str | always `gold` | Code |
| `n_steps` | int / null | number of recorded steps (an int here); `null` on OSWorld rows | Code |
| `canonical_instruction` | str / null | META-GUI rows only (the key is absent on other rows): one natural-language instruction that combines the user's task turns (rewritten on 96 rows, a source span on 19). `null` on 9 rows where no clean instruction exists | Gemma |
| `canonical_instruction_raw` | str | META-GUI rows only: the user turns listed in `canonical_turn_indices`, joined | Source |
| `canonical_turn_indices` | list of int | META-GUI rows only: the turns the annotator classed `Original instruction` or `Later task instruction` | Human |
| `review_handle` | obj | `label` (site name, as `site`), `raw_website` (Mind2Web `website`, else `null`), `substrate`, `url` (a URL made from the website name, Mind2Web only), `url_confidence` (`exact` 43 / `guessed` 107, else `null`) | Source + Code |
| `steps[].step_index`, `.image_path`, `.element_source` | | as in `gold_eval_set.jsonl` | |
| `steps[].action_repr` | str | the same rendering as `gold_eval_set` `steps[].action` | Source + Code |
| `steps[].operation`, `.gold_action.tool` | str | our action word: `click`, `read`, `input`, `swipe`, `type`, `select`, `enter`, `back`, `clear` | Code |
| `steps[].element` | str | the element name (Mind2Web DOM text; META-GUI view-hierarchy text at the tap point; "an unlabelled `<tag>`" is our wording) | Source |
| `steps[].gold_action.args` | obj | `target` (= `element`), `point` ([x, y]: Mind2Web centre of the target box, META-GUI tap point), `text` (typed text), `value` (selected value), `params` (META-GUI read/swipe parameters) | Source |
| `gold` | obj | the annotator record of the task | Human |
| `gold.decisions[]` | obj | `id`, `step` (commit step), `question`, `classification`, `options` (list), `best_assumption`, `best_assumption_reason`, `ask_or_guess`, `ask_before_step` (int / null), `clarification_question`, `discovery_answer`, `is_random_guess` (bool), `shared_steps` | Human |
| `gold.compounding_roots[]` | obj | the root cause of a decision: `id`, `earliest_step`, `decision_step`, `locked_option`, `other_options_cost`, `basis`, `reasoning`, `discovery_answer` | Human |
| `gold.assumptions[]` | obj | `id`, `step`, `observed` | Human |
| `gold.cross_modal_gaps[]` | obj | gaps between screen and instruction: `id`, `step`, `affordance`, `explanation`, `instruction_required`, `screen_exposed`, `proceeded_past_gap` | Human |
| `gold.step_discovery[]` | obj | per step: `step`, `answer`, `creates_decision`, `creates_assumption`, `creates_cross_modal` (bool) | Human |
| `gold.step_tracking[]` | obj | `id`, `step`, `affected_options`, `evidence_and_reason`, `information_source`, `discovery_answer` | Human |
| `gold.terminal_review` | obj | `step`, `completion_strategy`, `decisions_supported`, `options_unresolved` (same text as `annotator_notes` › `Looking back at the end`) | Human |
| `gold.notes` | obj | `edge_cases`, `fatigue`, `process_uncertainty` (same text as `annotator_notes` › `While annotating`) | Human |
| `gold.type_mode` | str / null | `Single dominant type` or `Exceptional ordered multiple types` | Human |
| `gold.trajectory_familiarity` | str | the annotator's own familiarity with the website or app: `Very familiar`, `Somewhat familiar`, `Not familiar`. It is about the annotator, not the task or a model | Human |
| `gold.turn_classifications[]` | obj | `turn`, `speaker`, `text` (source text as stored by the labelling tool), `classification` | Human + Source |
| `gold.dialogue_json[]` | obj | META-GUI rows: `i` (turn index), `speaker`, `role`, `text`: the dialogue as `turns` in the eval file | Source + Code |
| `gold.steps_json[]` | obj | the action list shown to annotators: `i`, `turn`, `action` (an older rendering without element for META-GUI taps), `img` (path pattern, prefixed by `--package-root`) | Source + Code |

## `open_items.windows.jsonl`

One row per open item (313 rows), keyed by (`sample_id`, `name`) = (`sample_id`, `open_items[].name`)
in `gold_eval_set.jsonl`.

| field | type | meaning | producer |
|---|---|---|---|
| `sample_id` | str | the task | Pointer |
| `name` | str | the open item's name | Pointer |
| `latest_step` | int | proposed last step at which asking still helps | Gemma |
| `commit_step` | int / null | proposed step whose action fixes the value; `null` if nothing in the recording fixes it (the item is never committed) | Gemma |
| `basis` | str | the evidence for the proposed window (steps and frames it rests on) | Gemma |

## Ask windows: `open_items.windows.jsonl` and `window.source`

`open_items.windows.jsonl` holds the **proposed** window of every one of the 313 open items:
Gemma-4-31B read the recorded steps and frames and named the step that fixes the value
(`commit_step`), the last step at which asking still helps (`latest_step`) and its evidence
(`basis`).

The gold rows carry one window per item in `open_items[].window`:

- on 301 items `window.source` is `proposed`: the window is the one in `open_items.windows.jsonl`
  (`never_committed` is true exactly when `commit_step` is `null`);
- on 12 items `window.source` is `recorded`: the gold file keeps a window written by hand by the
  owner and the annotators. It gives a `latest_step` but no commit step and no basis, so
  `never_committed` is false and `basis` is `""`. On these 12 items the windows file still has the
  proposed window, and the two differ: `basis` differs on all 12 (the gold basis is empty),
  `latest_step` on 4, and on 1 item the proposed window has no `commit_step` (never committed).

The offline evaluation reads only `open_items[].window` from the gold file; it reports
`recorded` and `proposed` windows separately and never averages them, and it times only items
with `never_committed == false`. Use `open_items.windows.jsonl` when you need the proposed window
or its commit step for all 313 items.

## `ambigui_online_benchmark.jsonl`

| field | type | meaning | producer |
|---|---|---|---|
| `sample_id`, `source`, `split`, `ambiguity`, `ambiguity_type`, `severity`, `risk` | | as in the gold files (equal to the gold row on the 76 gold rows; the 8 other rows are labelled by a person) | Human |
| `substrate` | str | always `web`: every task runs on a live website | Code |
| `recorded_substrate` | str | where the recording was made: `web` (Mind2Web) or `mobile` (META-GUI) | Code |
| `domain` | str | META-GUI category; `""` on Mind2Web rows | Source |
| `annotator_note` | str | annotator note (non-empty on 7 rows) | Human |
| `instruction` | str | the task given to the agent. Mind2Web: the recorded instruction; META-GUI: one instruction composed from the non-clarification turns | Source (Mind2Web) / Gemma (META-GUI) |
| `instruction_recorded` | str | the recorded instruction: Mind2Web `confirmed_task`; META-GUI the gold row's `instruction` (a user turn) | Source |
| `instruction_source` | str | `the recorded instruction, verbatim` or `composed from the non-clarification turns` | Code |
| `start_url` | str | the page the browser opens | Owner |
| `open_items` | list | as in the gold file: the list the simulated user and the ask judge use. Equal to the gold list on the 76 gold rows; on 4 of the 8 other rows one item written by the annotator (no `window`/`detail` keys) | Gemma / Human |
| `items_from_decision_points` | list | as in the gold file; never merge into `open_items` | Code over Human |
| `has_action_ground_truth` | bool | true for Mind2Web rows (the recorded web actions can be compared); false for META-GUI rows (recorded on Android: ask only) | Code |
| `steps` | list | Mind2Web rows: the recorded steps (`step_index`, `action_repr` = Mind2Web action text, `operation` = Mind2Web operation JSON, `image_path`, `point`, `gold_action`); `[]` on META-GUI rows | Source + Code |
| `turns` | list | META-GUI rows: the dialogue, as in `gold_eval_set.jsonl`; `[]` on Mind2Web rows | Source + Human + Code |
| `transfer` | obj / null | META-GUI rows: how well the Android task carries over to the website: `verdict` (`transfers` 14 / `partial` 25) and `judged_site`; `null` on Mind2Web rows | Owner |
| `flags` | list of str | `Ask-only: no action ground truth, the recording is Android` (39), `Stops before the commitment the recording completed` (25), `Labelled Clear but carries open items` (12 Clear rows with decision-point items; the online results code leaves them out of the Clear false-alarm measure) | Code |
| `max_steps` | int | the turn cap: 3 x the recorded trajectory length (Android rows: the gold row's step count) | Code |

## Known quirks (kept as the paper runs used them)

- `n_steps` is a string in `gold_eval_set.jsonl` and an int in the targets file.
- `ambiguity_type` is `""` in the eval file and `null` in the targets file on the same 6 rows.
- Online `domain` is `""` on the 45 Mind2Web rows (the gold rows have a domain).
- `steps[].annotations[]` values are strings, also for numbers and booleans.
- The hashes in `pointers.jsonl` (`atom_sha256`, `osworld_config_sha256`) are the first 128 bits
  (32 hex digits) of a SHA-256, to keep the file under 1 MB.
- No evaluation code reads these quirky fields; they are left as they are so that the files match
  the ones the paper's runs used.
