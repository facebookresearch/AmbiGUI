# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Frozen-weight ICL image: the ICL part, its pinned Python packages, the browser (Playwright 1.44.0
# with its Chromium 125, plus the system libraries Chromium needs) and the source files of vLLM
# 0.23.0 that the ICL arm's headroom guard reads (installed without dependencies, never imported).
# The models are not in the image: the simulated user is reached at ANSWERER_ENDPOINT, the judge at
# JUDGE_URL, and the frozen policy either at POLICY_URL or served by vLLM in a GPU container built
# on top of this one (pip install ".[icl-serve]"). See docs/icl.md.
#
#   docker build -f docker/icl.Dockerfile -t ambigui-icl .   # from the repository root
#   docker run --rm -it -e JUDGE_URL=... -e ANSWERER_ENDPOINT=... -e POLICY_URL=... \
#     -e AMBIGUI_TASKS=/data/tasks.jsonl -e ICL_CONDITION_DIR=/data/icl_condition \
#     -v $PWD/data:/data -v $PWD/runs:/work/runs ambigui-icl \
#     bash ambigui/icl/harness/run_online.sh matched icl4
FROM python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.playwright.dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements/icl.lock requirements/icl.lock
RUN python -m pip install -r requirements/icl.lock \
    && python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-deps --target /opt/vllm-src vllm==0.23.0
ENV ICL_VLLM_SITE_PACKAGES=/opt/vllm-src

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY ambigui ambigui
COPY configs configs
RUN python -m pip install --no-deps -e .

# A non-root user: Chromium's sandbox refuses to run as root.
RUN useradd --create-home --uid 1000 ambigui && mkdir -p /work/runs && chown -R ambigui /work /ms-playwright
USER ambigui

CMD ["bash"]
