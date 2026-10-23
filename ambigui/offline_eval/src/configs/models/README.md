# Model catalogue

One file per model the offline evaluation can score, named for the model's key. The
evaluation runner resolves its `--model` argument here; a key with no file raises, listing
the keys that exist, rather than falling back to a default.

## Settings

| setting | meaning |
|---|---|
| `key` | how a run names this model. Must equal the filename without its extension. |
| `hf_path` | which weights these settings describe: the Hugging Face model id, or, for a checkpoint you trained yourself, its directory. Recorded in each run so a report can be attributed later. The pinned Hugging Face revision (commit hash) for each model is in `configs/offline/models.yaml` at the repository root; serve exactly that revision. |
| `vision` | whether the weights take images. `true`: the model can be run with `--screens on` or `--screens off` (every run passes one; there is no default). `false`: a text-only checkpoint; the runner refuses `--screens on` at start-up. The `false` entries are `llama-3.3-70b-instruct`, `llama-3.3-70b-instruct-awq`, `minimax-m2.5` and `qwen-agentworld-35b-a3b`. |
| `max_image_dim` | longest side each screenshot is scaled to before it is sent; `null` sends it at its original size. |
| `max_tokens` | the default room for the whole reply (a run can replace it with `--max-tokens`; the open-weight run-set files do so with each recorded pass's budget). A model that writes its reasoning before the call needs room for both; sized for the call alone, the reply stops mid-thought and the step records as unparseable. `null` sends no limit, so the provider's default applies. |
| `description` | why these settings are what they are. Required: a budget with no stated reason can only be copied, not reviewed. |

## Adding a model

Write one file named for the key:

```yaml
key: my-model
hf_path: org/My-Model
vision: true
max_image_dim: 768
max_tokens: 512
description: |
  Why these settings are what they are, in enough detail to review rather than copy.
```

Then add its Hugging Face id and pinned revision to `configs/offline/models.yaml`, so the
serve that answers for it loads exactly those weights.

## What the loader refuses

Each refusal names the file. A setting that is unknown, missing, of the wrong type, given
twice, or not positive where it must be; a file that is not valid YAML, is not a mapping,
cannot be read, is not named for the key it declares, or does not end in `.yaml`; and a
catalogue directory that is empty or absent.
