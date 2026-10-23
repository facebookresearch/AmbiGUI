# Matched Qwen3.8 BASE vs SFT4 online evaluation: protocol

This page fixes the protocol of one paper comparison: the same base model before and after
supervised fine-tuning (SFT4), evaluated in the live browser under identical settings. The
comparison reruns the AmbiGUI online evaluation itself (not the training-time rollout
worker). Both checkpoints are evaluated fresh under one frozen protocol; no older baseline
result is reused.

## Checkpoints

- BASE: `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16,
  architecture `Qwen3_5ForConditionalGeneration`.
- SFT4: the same base plus our accepted SFT4 LoRA adapter. A LoRA adapter is a small set of
  extra weights trained on top of frozen base weights. Neither the adapter nor the training
  code is released with this repository, so this comparison cannot be reproduced from it; the
  protocol below applies unchanged to any LoRA fine-tune of the same base.
- Accepted adapter tensor digest (SHA-256 of `adapter_model.safetensors`):
  `cb3a3f916eaeabd162269223a04e523903467383b3e3e6a6ce8445ef7f5b7d57`.
- Serving the adapter: the adapter is merged into the base weights on CPU (for example with
  PEFT's `merge_and_unload`), and the merged checkpoint is served like any other model. After the
  merge, `harness/policy_serve.sh` (with `MERGED_FROM_BASE` and `ADAPTER` set) copies the base
  tokenizer, processor and chat template over the merged copies byte for byte, and
  `harness/verify_merge.py` requires identical vision weights, at least one actually changed
  language weight, and finite values in every tensor.
- Restoring the `mtp.*` weights: a PEFT merge of this base saves 15 fewer tensors, the
  multi-token-prediction weights (`mtp.fc`, `mtp.layers.0.*`, `mtp.norm`, `mtp.pre_fc_norm_*`).
  The adapter does not touch them. Copy these 15 tensors unchanged, byte for byte, from the base
  checkpoint into the merged checkpoint (and its weight index) before running
  `harness/verify_merge.py`; otherwise it stops with "parameter set changed".

## Identical protocol for both policies

- Code: this repository's online evaluation (harness and `src/`).
- Benchmark: the full84 task file, SHA-256
  `4efd151060567ab55a85a272fbbae26934b2d05cd1e7f66d27ec6e5ad74ab84f`.
- Prompt: the baseline system prompt, with no added encouragement to ask and no few-shot
  demonstrations.
- Observation: `som_state` (screenshot with element marks plus page state); Chromium 125,
  Playwright 1.44, BrowserGym 0.14.3.
- At most 4 images per request; 8192 new tokens per turn; 65536-token context; temperature 0;
  seed 0; 600-second deadline per episode; the per-row step caps of the task file, unchanged.
- vLLM 0.23, one GPU per policy (tensor parallel 1), served remotely to the harness, eager mode
  (`--enforce-eager`), at most 8 sequences at once (`--max-num-seqs 8`), GPU memory utilisation
  0.90, server seed 1234, BF16; the original processor, tokenizer and chat template bytes.
- Simulated user (the "answerer", which replies to the agent's questions):
  **Llama-3.2-90B-Vision-Instruct**, through an OpenAI-compatible API, with the
  question/statement gate and the nothing-open path. This differs on purpose from the
  simulated user used in training, **Qwen3-VL-235B**; the difference is explicit and held
  constant for both evaluated policies.
- Task judge, screenshot describer and ask judge: **Qwen3-VL-235B-A22B-Instruct**, reasoning
  off, the same endpoint and settings for both policies.

## Integrity and analysis

Both policies keep complete request and response journals of every policy call
(`RECORD_POLICY_CALLS=1`), including any reasoning field, the exact text, and the image bytes
stored by content hash. Full task screenshots, trajectories, questions and replies, and the
phase-2 grading coverage are saved with each run. The base, adapter and merged checkpoint
hashes travel with the run in the serve identity files.

Report Ambiguous and Clear tasks separately, under the harness, README (excluding "Stops"
rows), strict no-flags and action-ground-truth success views (see `docs/online.md`).
Reachability and missing outcomes stay explicit per policy and task; pair the two policies on
the tasks both could reach. Never turn missing question grades into zeros. The benchmark
includes 24 Android-origin Ambiguous tasks without action ground truth, so success over all
84 tasks is not the primary matched execution result.

The available provenance shows no exact task-ID or screenshot-path overlap with the 672-task
training split, no duplicate instructions after normalisation, and disjoint train and test
splits; Android tasks are absent from the training sources. Cross-dataset equivalences that
are not documented are not assumed.
