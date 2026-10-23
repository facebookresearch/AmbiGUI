#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Run scripts/eval_cascp_online.py unchanged, with three recorded environment patches
(and, only when ASK_PROMPT_VARIANT is set, the ask-paragraph variant from prompt_variant.py).

1. Route label. The harness has an optional fallback proxy for pages that block the
   direct attempt (BRIGHTDATA_PROXY_* -> evaluation.web_transport). The record labels a
   rollout that used it with ``web_transport.FALLBACK``. Set PROXY_ROUTE_LABEL to name
   your proxy in the records; unset, the module's own label is kept. The label is
   record metadata only and is never scored.
2. Start-page load. BrowserGym's OpenEndedTask.setup loads the start page with a
   hard-coded 10 s `page.goto` that waits for the `load` event. Heavy start pages
   (weather.com, finance.yahoo.com, bestbuy.com, ign.com, tvguide.com) often never
   fire `load` -- not in 10 s, not in 45 s -- because some third-party subresource never
   completes, and the rollout would be recorded `env_unreachable` before the policy
   sees a page. Here the start page is awaited to `domcontentloaded` (60 s, the paper
   runs' setting), then `load` is given up to 10 s more, best effort. Nothing else
   changes.
"""

import os
import runpy
import sys

import browsergym.core.task as bg_task
import evaluation.web_transport as wt

wt.FALLBACK = os.environ.get("PROXY_ROUTE_LABEL") or wt.FALLBACK

_GOTO_MS = 60000
_LOAD_GRACE_MS = 10000


def _setup(self, page):  # noqa: ANN001, ANN202 - same contract as OpenEndedTask.setup
    page.goto(self.start_url, timeout=_GOTO_MS, wait_until="domcontentloaded")
    try:
        page.wait_for_load_state("load", timeout=_LOAD_GRACE_MS)
    except Exception:  # noqa: BLE001 - a page whose load never fires is still usable
        pass
    return self.goal, {}


bg_task.OpenEndedTask.setup = _setup

# 3. Judge live-check timeout. Each shard process first sends one tiny completion to the
#    judge (evaluation.serve_preflight.assert_endpoint_live, hard-coded 30 s read timeout).
#    A rate-limited judge endpoint (for example a hosted one that allows about one
#    request per second) queues these checks when ten shards start together, and the
#    later ones exceed 30 s: the shard then dies with ServeDead before its task starts. The check is kept; only its read timeout is raised, to
#    900 s, so it waits its turn instead.
import functools  # noqa: E402

import evaluation.serve_preflight as serve_preflight  # noqa: E402

serve_preflight.assert_endpoint_live = functools.partial(
    serve_preflight.assert_endpoint_live,
    timeout=900.0,
)

import prompt_variant  # noqa: E402 - harness/prompt_variant.py; a no-op unless ASK_PROMPT_VARIANT is set

prompt_variant.apply()

import policy_call_recording  # lossless journals; returns the original API response unchanged

policy_call_recording.apply()

sys.argv = ["scripts/eval_cascp_online.py", *sys.argv[1:]]
runpy.run_path("scripts/eval_cascp_online.py", run_name="__main__")
