# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Online evaluation image: the harness, its pinned Python packages and the browser
# (Playwright 1.44.0 with its Chromium 125, plus the system libraries Chromium needs).
# The models (policy, simulated user, judge) are not in the image: they are reached at
# OpenAI-compatible URLs given by environment variables (see docs/online.md).
#
#   docker build -f docker/online.Dockerfile -t ambigui-online .   # from the repository root
#   docker run --rm -it -e AMBIGUI_TASKS=/data/full84.jsonl -e ANSWERER_ENDPOINT=... \
#     -e ANSWERER_MODEL=... -e JUDGE_URL=... -e JUDGE_MODEL=... -e POLICY_URL=... \
#     -v $PWD/data:/data -v $PWD/runs:/work/runs ambigui-online \
#     bash ambigui/online_eval/harness/run_online.sh demo <policy>
FROM python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.playwright.dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements/online.lock requirements/online.lock
RUN python -m pip install -r requirements/online.lock \
    && python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY ambigui ambigui
COPY configs configs
RUN python -m pip install --no-deps -e .

# A non-root user: Chromium's sandbox refuses to run as root.
RUN useradd --create-home --uid 1000 ambigui && mkdir -p /work/runs && chown -R ambigui /work /ms-playwright
USER ambigui

CMD ["bash"]
