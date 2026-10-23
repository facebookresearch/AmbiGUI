# AmbiGUI benchmark — pointer-only release

This directory contains no third-party data. It holds the gold benchmark (offline evaluation,
298 rows) and the online benchmark (84 rows, curated from the gold rows plus 8 annotated Mind2Web
rows), as pointers and our own labels:

- `manifest/sources.yaml` — the source datasets (Mind2Web, META-GUI, OSWorld), pinned versions, licences (with the URL each was read from) and the release terms.
- `manifest/pointers.jsonl` — one line per `sample_id`: source-native ids, the step map, and hashes of the source values the build uses. OSWorld rows name the OSWorld task file, commit and config hash.
- `manifest/label_provenance.yaml` — for every field: who or what produced it, the evidence, the decision (APPROVED / NOT_APPROVED / UNKNOWN) and what ships. Provenance lives here, not in the data rows.
- `manifest/removed_labels.json` — the labels and fields that do not ship. `annotations/` already carries these removals, and `build.py` applies them again.
- `manifest/field_classes.yaml` — every field of the three gold files classified POINTER / OURS / DERIVED-FROM-SOURCE / MODEL-GENERATED.
- `manifest/CHECKSUMS.sha256` — sha256 of the four files `build.py` writes (default `--pool-root` and `--package-root`).
- `annotations/` — our labels, with every source-derived value replaced by a template that points at the source.
- `build.py` — rebuilds `gold_eval_set.jsonl`, `gold_benchmark_with_targets.jsonl`, `open_items.windows.jsonl` and `ambigui_online_benchmark.jsonl` from the above and your own copy of the sources.
- `TRACE_REPORT.md` — trace counts, label provenance, release terms, modified-text rows, residual risks.
- `DATA_CARD.md` — every field of the four files: type, meaning and who made it; row counts.

A row's `source` is `mind2web`, `meta-gui` or `osworld`.

## Two lists of open items: use `open_items` only

Gold and online rows have the same two lists:

- `open_items` — what the task left open and the answer the user would give. The simulated
  user answers from this list and the ask judge grades questions against it. On the gold rows
  these items were written by Gemma-4-31B; on 4 online rows that are not in the gold file
  (0baad732, 248803b0, 297361ad, 3068d843) an annotator wrote them.
- `items_from_decision_points` — items read off the annotators' own decision records.
  Supplementary; never merge them into `open_items`.

On the 76 online rows that are also gold rows, both lists equal the gold row's lists. One
Ambiguous row (406da915) has no `open_items`, only a decision-point item, so it cannot drive an
ask arm.

The online flag "Labelled Clear but carries open items" marks the 12 Clear rows on which the
annotators recorded decision points (their `items_from_decision_points` is non-empty; their
`open_items` is empty). The text is kept as the online results code expects it: that code
leaves these rows out of the Clear false-alarm measure.

Re-scoring the paper's recorded online runs with this file reproduces the paper's online
tables except for two denominators per policy: Ambiguous "asked" and "on-target" use 41 rows,
not 42 (406da915 has no curated open item; a policy that asked on it also loses 1 from the
Ambiguous "asked" numerator), and Clear "success no flags" uses 21 rows, not 20
(row 1564d78a carried only a model-panel flag, which is not shipped). See `TRACE_REPORT.md`
section 11.

The online file also carries `max_steps`, the per-task turn cap: 3 x the length of the
recorded trajectory (for rows recorded on Android, the gold row's step count). It equals the
cap of the paper's runs on all 84 rows.

## Ask windows: `open_items.windows.jsonl`

This file holds the proposed ask window of every open item (313 rows, one per item): the step
whose action fixes the value (`commit_step`, `null` if nothing fixes it), the last step at which
asking still helps (`latest_step`) and the evidence (`basis`). Join it to `open_items` on
`sample_id` and `name`. In the gold rows, `open_items[].window.source` says which window the row
carries: `proposed` (301 items, the same window as in this file) or `recorded` (12 items, written by hand by the owner and the
annotators, without commit step or basis; they differ from this file). The proposed windows are by
Gemma-4-31B. The offline
evaluation reads only the gold rows' windows. Details: `DATA_CARD.md`.

## What ships

Only the final labels and the open items ship, made by people in the human annotation study,
by the owner, or by an approved model (Gemma-4-31B wrote the open items, their proposed ask
windows, `canonical_instruction` and the instructions of the online rows carried over from
META-GUI). Not shipped: the initial judgements, the model-panel fields of the online file, the
`origin` of each open item, a few fields that hold the same value on every row and that no code
reads, and 9 `canonical_instruction` values that were only model reasoning (`null`). See
`TRACE_REPORT.md` section 10.

## OSWorld rows

The 33 OSWorld rows are OSWorld tasks. Each points at the OSWorld task file at a pinned commit;
its instruction comes from there, and our labels ship as annotations. These rows use OSWorld's
own environment and grounding, so the release ships no recorded trajectory for them: `steps`
is `[]`, `n_steps` is `null`, and there are no click coordinates. One instruction (vlc
`a5bbbcd5`) was reworded upstream after annotation; the release uses the OSWorld text at the
pinned commit.

What this means for the evaluation code (checked on the release code at the time of writing):

- **Offline** (teacher-forced over recorded steps): an OSWorld row yields no step to score. The
  loader skips it without an error and without a score; it is not counted as 0. A run over the
  whole gold file scores the 265 Mind2Web and META-GUI rows. A run over OSWorld rows alone stops
  with `no scorable steps`.
- **Online**: the online benchmark has no OSWorld rows. The online loader needs a `start_url`,
  so an OSWorld row given to it is refused (`has no start_url`). Run OSWorld rows in the OSWorld
  environment instead, with the OSWorld task file named in the pointer.

## Rebuild

1. Get the sources at the versions in `manifest/sources.yaml`:
   - Mind2Web: the test splits (`data/test_*.parquet`) of the Hugging Face dataset whose id is in `manifest/sources.yaml`, at the pinned revision.
   - META-GUI: `test.tar` of `X-LANCE/META-GUI` at the pinned revision (Hugging Face), extracted.
   - OSWorld: `git clone https://github.com/xlang-ai/OSWorld` and check out the pinned commit.
2. Python 3.12: `pip install -r requirements/benchmark.lock` (pyarrow 26.0.0; `build.py` needs nothing else).
3. Run:

```
python benchmark/build.py --mind2web-dir <Mind2Web dataset root> --metagui-dir <META-GUI extracted> \
    --osworld-dir <OSWorld> --out <dir>
(cd <dir> && sed '/^#/d; /^$/d' <repo>/benchmark/manifest/CHECKSUMS.sha256 | sha256sum -c -)
```

The checksum manifest has the repository's required copyright header; filter its comment
lines before passing the four checksum entries to `sha256sum`.

`--mind2web-dir` is the dataset root: the folder that holds `data/`, not the `data/` folder itself.

Every source value is checked against the hash in `pointers.jsonl` (the first 128 bits of its SHA-256). A different source version stops the run (exit code 1) and lists every row that differs, with its id and reason (for OSWorld rows: the task id, the config hash found and the one expected). With `--allow-missing-source` the other rows are written and the listed rows are left out (exit code 2). The OSWorld pointers were checked against a real `git clone` of OSWorld at the pinned commit: all 33 rows rebuild and the checksums match. Screenshots are not part of the rebuild: take them from each source using the step map in `pointers.jsonl` (see TRACE_REPORT.md §7). Screenshots can be added easily from Mind2Web (Hugging Face dataset id `osunlp/Multimodal-Mind2Web`), because it uses the same annotation and action ids as the pointers.
