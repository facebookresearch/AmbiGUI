# Third-party notices

AmbiGUI is licensed under CC BY-NC 4.0 (see [LICENSE](LICENSE)). This file lists the third-party
code that is in this repository, the third-party code that the repository can call at run time, and
the direct Python dependencies of each install extra. Each of these keeps its own licence.

Third party content pulled from other locations are subject to its own licenses and you may have other legal obligations or restrictions that govern your use of that content.

## 1. Third-party code included in this repository

No third-party source code is included in this repository.

## 2. Third-party code called at run time but not included

### 2.1 Models and datasets

The configs name models on the Hugging Face Hub (for example Qwen, Gemma, Llama, InternVL, GLM,
UI-TARS) and hosted model APIs. You download or call them yourself. Each model is governed by its
own licence or terms of use, and some of these are not open-source licences. The benchmark source
datasets (Mind2Web, META-GUI and OSWorld) are not included and are governed
by their own terms.

### 2.2 Browser and serving code

- Chromium 125, downloaded by Playwright 1.44.0 at install time for the online evaluation and the
  ICL baseline (`python -m playwright install chromium`). It is not included in this repository
  and is governed by the Chromium licence and the licences of its components.
- The ICL headroom guard reads source files of an installed vLLM 0.23.0 (Apache-2.0) and uses
  transformers 5.14.1 (Apache-2.0); see DEPENDENCIES.md and docs/icl.md. Neither is included in
  this repository.

## 3. Direct dependencies

The direct dependencies are the packages named in the extras of `pyproject.toml`. The version is the
pinned version of that extra: from `pyproject.toml`, or from `requirements/offline.lock` for the
`[offline]` extra, whose `pyproject.toml` entries are minimum versions. The licence is read from the
package metadata (installed package or PyPI). The full pinned set of every extra, with transitive
packages, is in [DEPENDENCIES.md](DEPENDENCIES.md).


### `[offline]`

| package | pinned version | licence | URL |
|---|---|---|---|
| requests | 2.34.2 | Apache-2.0 | https://github.com/psf/requests |
| PyYAML | 6.0.3 | MIT | https://github.com/yaml/pyyaml |
| pillow | 13.0.0 | MIT-CMU | https://github.com/python-pillow/Pillow |
| openai | 3.35.1 | Apache-2.0 | https://github.com/openai/openai-python |

### `[online]`

| package | pinned version | licence | URL |
|---|---|---|---|
| browsergym-core | 0.14.3 | Apache-2.0 | https://github.com/ServiceNow/BrowserGym |
| playwright | 1.44.0 | Apache-2.0 | https://github.com/microsoft/playwright-python |
| greenlet | 3.0.3 | MIT AND PSF-2.0 (LICENSE and LICENSE.PSF) | https://github.com/python-greenlet/greenlet |
| pyee | 11.1.0 | MIT | https://pypi.org/project/pyee/ |
| openai | 3.35.1 | Apache-2.0 | https://github.com/openai/openai-python |
| requests | 2.34.2 | Apache-2.0 | https://github.com/psf/requests |
| pillow | 13.0.0 | MIT-CMU | https://github.com/python-pillow/Pillow |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://github.com/numpy/numpy |
| huggingface_hub | 2.2.0 | Apache-2.0 | https://github.com/huggingface/huggingface_hub |

### `[online-serve]`

| package | pinned version | licence | URL |
|---|---|---|---|
| vllm | 0.23.0 | Apache-2.0 | https://github.com/vllm-project/vllm |

### `[icl]`

| package | pinned version | licence | URL |
|---|---|---|---|
| browsergym-core | 0.14.3 | Apache-2.0 | https://github.com/ServiceNow/BrowserGym |
| playwright | 1.44.0 | Apache-2.0 | https://github.com/microsoft/playwright-python |
| greenlet | 3.0.3 | MIT AND PSF-2.0 (LICENSE and LICENSE.PSF) | https://github.com/python-greenlet/greenlet |
| pyee | 11.1.0 | MIT | https://pypi.org/project/pyee/ |
| gymnasium | 1.4.0 | MIT | https://github.com/Farama-Foundation/Gymnasium |
| openai | 3.35.1 | Apache-2.0 | https://github.com/openai/openai-python |
| requests | 2.34.2 | Apache-2.0 | https://github.com/psf/requests |
| pillow | 13.0.0 | MIT-CMU | https://github.com/python-pillow/Pillow |
| huggingface_hub | 1.25.1 | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| transformers | 5.14.1 | Apache-2.0 | https://github.com/huggingface/transformers |
| tokenizers | 0.22.2 | Apache-2.0 | https://github.com/huggingface/tokenizers |
| jinja2 | 3.1.6 | BSD-3-Clause | https://github.com/pallets/jinja |

### `[icl-serve]`

| package | pinned version | licence | URL |
|---|---|---|---|
| vllm | 0.23.0 | Apache-2.0 | https://github.com/vllm-project/vllm |

### `[benchmark]`

| package | pinned version | licence | URL |
|---|---|---|---|
| pyarrow | 26.0.0 | Apache-2.0 | https://github.com/apache/arrow |

## 4. Packages in the lock files whose licence is not permissive or not stated

These are transitive dependencies (none is a direct dependency). They are listed so that nothing is
hidden. They are installed from PyPI by the user; none of them is included in this repository.

| package | version | lock file(s) | licence (from package metadata) |
|---|---|---|---|
| certifi | 2026.10.19 | offline, online, icl | MPL-2.0 |
| tqdm | 4.70.1 | online, icl | MPL-2.0 AND MIT |
