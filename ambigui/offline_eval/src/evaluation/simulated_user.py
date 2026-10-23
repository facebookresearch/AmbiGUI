# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The user the policy talks to.

A policy can ask the person who set the task a clarifying question, but in a recorded
trajectory there is no person to answer. A *simulated user* is a language model that
plays that person: it reads the policy's question and replies the way the person who
recorded the task would. In this project it is also called the *answerer*.

The answerer itself lives in ``scripts/answerer.py``. This module only loads it, so the
offline runner (:mod:`offline_evaluation.run.run_ask_protocol`) can reach it without
depending on how ``sys.path`` is ordered.
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Callable


def load_answerer(endpoint: str | None, model: str | None) -> Callable[..., str | None]:
    """The served user simulator, loaded by absolute path.

    ``from scripts.answerer import Answerer`` can be shadowed by an unrelated ``scripts``
    package installed in the same environment, which raises ``ModuleNotFoundError``.
    Loading the file by path bypasses ``sys.path`` ordering.
    """
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "answerer.py"
    spec = importlib.util.spec_from_file_location("_stage_answerer", path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreadable file
        raise ImportError(f"no answerer at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    answerer: Callable[..., str | None] = module.Answerer(
        endpoint=endpoint, model=model
    ).answer
    return answerer
