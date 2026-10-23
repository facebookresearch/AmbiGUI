#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Run the unchanged phase-2 scorer with a queue-tolerant live-check timeout.

With a rate-limited judge endpoint, a 30-second preflight can fail even while
real requests succeed. Only the timeout is changed (fixed at 900 seconds).
"""
from __future__ import annotations

import functools
import runpy
import sys

import evaluation.serve_preflight as preflight


def main() -> None:
    preflight.assert_endpoint_live = functools.partial(
        preflight.assert_endpoint_live,
        timeout=900.0,
    )
    sys.argv = ["scripts/score_online_ask_quality.py", *sys.argv[1:]]
    runpy.run_path("scripts/score_online_ask_quality.py", run_name="__main__")


if __name__ == "__main__":
    main()
