# Dependency manifest

This file lists, for each install extra of `pyproject.toml`, every pinned package of its lock file
under `requirements/`. Direct dependencies (named in the extra) are marked **direct**; all others are
transitive. Licences are read from the package metadata (installed package or PyPI); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for URLs and licence texts.

Install one extra per environment. The extras pin different versions of some shared packages
(for example `openai`, `pillow`, `huggingface-hub`, `numpy`).

| extra | install | lock file | packages |
|---|---|---|---|
| `[offline]` | `pip install -r requirements/offline.lock && pip install --no-deps -e .` | `requirements/offline.lock` | 20 |
| `[online]` | `pip install -r requirements/online.lock && pip install --no-deps -e .` | `requirements/online.lock` | 63 |
| `[online-serve]` | `pip install -e ".[online-serve]"` | none (see below) | 1 direct |
| `[icl]` | `pip install -r requirements/icl.lock && pip install --no-deps -e .` | `requirements/icl.lock` | 71 |
| `[icl-serve]` | `pip install -e ".[icl-serve]"` | none (see below) | 1 direct |
| `[benchmark]` | `pip install -r requirements/benchmark.lock` | `requirements/benchmark.lock` | 1 |

The browser for `[online]` and `[icl]` is not a Python package:
`PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.playwright.dev python -m playwright install --with-deps chromium`
(Chromium 125 for Playwright 1.44.0; Playwright's built-in download host no longer serves this build).


## `[offline]` — `requirements/offline.lock`

| package | version | direct | licence |
|---|---|---|---|
| annotated-types | 0.8.0 |  | MIT |
| anyio | 4.15.1 |  | MIT |
| certifi | 2026.10.19 |  | MPL-2.0 |
| charset-normalizer | 3.5.2 |  | MIT |
| h11 | 0.16.0 |  | MIT |
| httpcore2 | 2.16.0 |  | BSD-3-Clause |
| httpx2 | 2.16.0 |  | BSD-3-Clause |
| idna | 3.20 |  | BSD-3-Clause |
| jiter | 0.17.0 |  | MIT |
| openai | 3.35.1 | direct | Apache-2.0 |
| pillow | 13.0.0 | direct | MIT-CMU |
| pydantic | 2.14.1 |  | MIT |
| pydantic_core | 2.50.1 |  | MIT |
| PyYAML | 6.0.3 | direct | MIT |
| requests | 2.34.2 | direct | Apache-2.0 |
| sniffio | 1.3.1 |  | Apache Software License; MIT License |
| truststore | 0.10.5 |  | MIT |
| typing-inspection | 0.4.4 |  | MIT |
| typing_extensions | 4.16.0 |  | PSF-2.0 |
| urllib3 | 2.8.0 |  | MIT |

## `[online]` — `requirements/online.lock`

| package | version | direct | licence |
|---|---|---|---|
| annotated-doc | 0.0.5 |  | MIT |
| annotated-types | 0.8.0 |  | MIT |
| anyio | 4.15.1 |  | MIT |
| attrs | 26.1.0 |  | MIT |
| beautifulsoup4 | 4.15.0 |  | MIT License |
| browsergym-core | 0.14.3 | direct | Apache-2.0 |
| certifi | 2026.10.19 |  | MPL-2.0 |
| cffi | 2.1.1 |  | MIT-0 |
| charset-normalizer | 3.5.2 |  | MIT |
| click | 8.5.0 |  | BSD-3-Clause |
| cloudpickle | 3.1.2 |  | BSD-3-Clause |
| cryptography | 50.0.2 |  | Apache-2.0 OR BSD-3-Clause |
| Farama-Notifications | 0.0.6 |  | MIT License |
| filelock | 4.2.4 |  | MIT |
| fsspec | 2026.7.0 |  | BSD-3-Clause |
| greenlet | 3.0.3 | direct | MIT AND PSF-2.0 (LICENSE and LICENSE.PSF) |
| gymnasium | 1.4.0 |  | MIT |
| h11 | 0.16.0 |  | MIT |
| hf-xet | 1.7.0 |  | Apache-2.0 |
| httpcore2 | 2.16.0 |  | BSD-3-Clause |
| httpx2 | 2.16.0 |  | BSD-3-Clause |
| huggingface_hub | 2.2.0 | direct | Apache-2.0 |
| idna | 3.20 |  | BSD-3-Clause |
| jiter | 0.17.0 |  | MIT |
| jsonschema | 4.26.0 |  | MIT |
| jsonschema-specifications | 2025.9.1 |  | MIT |
| lxml | 5.4.0 |  | BSD-3-Clause |
| markdown-it-py | 4.2.0 |  | MIT License |
| mcp | 2.3.0 |  | MIT |
| mcp-types | 2.3.0 |  | MIT |
| mdurl | 0.1.2 |  | MIT License |
| numpy | 2.5.4 | direct | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| openai | 3.35.1 | direct | Apache-2.0 |
| opentelemetry-api | 1.45.1 |  | Apache-2.0 |
| packaging | 26.3 |  | Apache-2.0 OR BSD-2-Clause |
| pillow | 13.0.0 | direct | MIT-CMU |
| playwright | 1.44.0 | direct | Apache-2.0 |
| pycparser | 3.11 |  | BSD-3-Clause |
| pydantic | 2.14.1 |  | MIT |
| pydantic_core | 2.50.1 |  | MIT |
| pyee | 11.1.0 | direct | MIT |
| Pygments | 2.21.0 |  | BSD-2-Clause |
| PyJWT | 2.15.1 |  | MIT |
| pyparsing | 3.3.3 |  | MIT |
| python-dotenv | 1.2.4 |  | BSD-3-Clause |
| python-multipart | 0.0.32 |  | Apache-2.0 |
| PyYAML | 6.0.3 |  | MIT |
| referencing | 0.37.0 |  | MIT |
| requests | 2.34.2 | direct | Apache-2.0 |
| rich | 15.0.0 |  | MIT |
| rpds-py | 2026.9.1 |  | MIT |
| shellingham | 1.5.4 |  | ISC License (ISCL) |
| sniffio | 1.3.1 |  | Apache Software License; MIT License |
| soupsieve | 3.0.3 |  | MIT |
| sse-starlette | 3.5.0 |  | BSD-3-Clause |
| starlette | 1.9.0 |  | BSD-3-Clause |
| tqdm | 4.70.1 |  | MPL-2.0 AND MIT |
| truststore | 0.10.5 |  | MIT |
| typer | 0.27.3 |  | MIT |
| typing-inspection | 0.4.4 |  | MIT |
| typing_extensions | 4.16.0 |  | PSF-2.0 |
| urllib3 | 2.8.0 |  | MIT |
| uvicorn | 0.54.0 |  | BSD-3-Clause |

## `[online-serve]`

| package | version | direct | licence |
|---|---|---|---|
| vllm | 0.23.0 | direct | Apache-2.0 |

This extra has no lock file of its own. It adds vLLM 0.23.0, the version of the paper runs. It is often
easiest to install vLLM in its own environment (see docs/online.md).


## `[icl]` — `requirements/icl.lock`

| package | version | direct | licence |
|---|---|---|---|
| annotated-doc | 0.0.5 |  | MIT |
| annotated-types | 0.8.0 |  | MIT |
| anyio | 4.15.1 |  | MIT |
| attrs | 26.1.0 |  | MIT |
| beautifulsoup4 | 4.15.0 |  | MIT License |
| browsergym-core | 0.14.3 | direct | Apache-2.0 |
| certifi | 2026.10.19 |  | MPL-2.0 |
| cffi | 2.1.1 |  | MIT-0 |
| charset-normalizer | 3.5.2 |  | MIT |
| click | 8.5.0 |  | BSD-3-Clause |
| cloudpickle | 3.1.2 |  | BSD-3-Clause |
| cryptography | 50.0.2 |  | Apache-2.0 OR BSD-3-Clause |
| Farama-Notifications | 0.0.6 |  | MIT License |
| filelock | 4.2.4 |  | MIT |
| fsspec | 2026.9.0 |  | BSD-3-Clause |
| greenlet | 3.0.3 | direct | MIT AND PSF-2.0 (LICENSE and LICENSE.PSF) |
| gymnasium | 1.4.0 | direct | MIT |
| h11 | 0.16.0 |  | MIT |
| hf-xet | 1.7.0 |  | Apache-2.0 |
| httpcore | 1.0.9 |  | BSD-3-Clause |
| httpcore2 | 2.16.0 |  | BSD-3-Clause |
| httpx | 0.28.1 |  | BSD-3-Clause |
| httpx2 | 2.16.0 |  | BSD-3-Clause |
| huggingface_hub | 1.25.1 | direct | Apache-2.0 |
| idna | 3.20 |  | BSD-3-Clause |
| Jinja2 | 3.1.6 | direct | BSD-3-Clause |
| jiter | 0.17.0 |  | MIT |
| jsonschema | 4.26.0 |  | MIT |
| jsonschema-specifications | 2025.9.1 |  | MIT |
| lxml | 5.4.0 |  | BSD-3-Clause |
| markdown-it-py | 4.2.0 |  | MIT License |
| MarkupSafe | 3.0.4 |  | BSD-3-Clause |
| mcp | 2.3.0 |  | MIT |
| mcp-types | 2.3.0 |  | MIT |
| mdurl | 0.1.2 |  | MIT License |
| numpy | 2.5.4 |  | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| openai | 3.35.1 | direct | Apache-2.0 |
| opentelemetry-api | 1.45.1 |  | Apache-2.0 |
| packaging | 26.3 |  | Apache-2.0 OR BSD-2-Clause |
| pillow | 13.0.0 | direct | MIT-CMU |
| playwright | 1.44.0 | direct | Apache-2.0 |
| pycparser | 3.12 |  | BSD-3-Clause |
| pydantic | 2.14.1 |  | MIT |
| pydantic_core | 2.50.1 |  | MIT |
| pyee | 11.1.0 | direct | MIT |
| Pygments | 2.21.0 |  | BSD-2-Clause |
| PyJWT | 2.15.1 |  | MIT |
| pyparsing | 3.3.3 |  | MIT |
| python-dotenv | 1.2.4 |  | BSD-3-Clause |
| python-multipart | 0.0.32 |  | Apache-2.0 |
| PyYAML | 6.0.3 |  | MIT |
| referencing | 0.37.0 |  | MIT |
| regex | 2026.9.29 |  | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | direct | Apache-2.0 |
| rich | 15.0.0 |  | MIT |
| rpds-py | 2026.9.1 |  | MIT |
| safetensors | 0.8.0 |  | Apache-2.0 |
| shellingham | 1.5.4 |  | ISC License (ISCL) |
| sniffio | 1.3.1 |  | Apache Software License; MIT License |
| soupsieve | 3.0.4 |  | MIT |
| sse-starlette | 3.5.0 |  | BSD-3-Clause |
| starlette | 1.9.0 |  | BSD-3-Clause |
| tokenizers | 0.22.2 | direct | Apache-2.0 |
| tqdm | 4.70.1 |  | MPL-2.0 AND MIT |
| transformers | 5.14.1 | direct | Apache-2.0 |
| truststore | 0.10.5 |  | MIT |
| typer | 0.27.3 |  | MIT |
| typing-inspection | 0.4.4 |  | MIT |
| typing_extensions | 4.16.0 |  | PSF-2.0 |
| urllib3 | 2.8.0 |  | MIT |
| uvicorn | 0.54.0 |  | BSD-3-Clause |

## `[icl-serve]`

| package | version | direct | licence |
|---|---|---|---|
| vllm | 0.23.0 | direct | Apache-2.0 |

This extra has no lock file of its own. It adds vLLM 0.23.0, the version that served the frozen
policy in the reported ICL runs. The ICL arm's headroom guard also reads four source files of an
installed vLLM 0.23.0 (without importing it; see docs/icl.md). It is often easiest to install vLLM in its own
environment and point `VLLM_PY` at that Python (see docs/icl.md).

## `[benchmark]` — `requirements/benchmark.lock`

| package | version | direct | licence |
|---|---|---|---|
| pyarrow | 26.0.0 | direct | Apache-2.0 |

`benchmark/build.py` needs only this package besides the Python 3.12 standard library: pyarrow reads
the Mind2Web parquet files. The script runs from the repository; it does not need `pip install -e .`.
