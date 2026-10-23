# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# The offline evaluation of AmbiGUI, built from this repository alone.
# Build from the repository root (the build context is the repository root).
#
# The image holds the evaluation code and its pinned Python packages. It does not hold any
# model: the policy, the simulated user and the judge are reached over HTTP, at the
# OpenAI-compatible endpoints the run-set file names (for example vLLM serves started from
# configs/offline/models.yaml). The benchmark file and screenshots are mounted at run time.
#
#   docker build --build-arg AMBIGUI_GIT_COMMIT="$(git rev-parse HEAD)" -f docker/offline.Dockerfile -t ambigui-offline .
#   docker run --rm -v "$PWD/runs:/opt/ambigui/runs" ambigui-offline \
#       score --config configs/offline/run_sets/frontier_llama90.yaml --model <key> --screens on
#
# See docs/offline.md.

FROM python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/ambigui

# The image holds no .git folder, so the commit recorded in run_manifest.json is passed in:
#   docker build --build-arg AMBIGUI_GIT_COMMIT="$(git rev-parse HEAD)" -f docker/offline.Dockerfile -t ambigui-offline .
ARG AMBIGUI_GIT_COMMIT=""
ENV AMBIGUI_GIT_COMMIT=${AMBIGUI_GIT_COMMIT}

COPY requirements/offline.lock requirements/offline.lock
RUN pip install -r requirements/offline.lock

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY ambigui ./ambigui
COPY configs ./configs
COPY docs/offline.md ./docs/offline.md
RUN pip install --no-deps -e .

# The three stage programs can also be run directly, e.g.
#   python -m offline_evaluation.run.score_ask_protocol <run-dir>
ENV PYTHONPATH=/opt/ambigui/ambigui/offline_eval/src

ENTRYPOINT ["python", "-m", "ambigui.offline_eval.launch"]
CMD ["--help"]
