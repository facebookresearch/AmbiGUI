# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The API key to send to an OpenAI-compatible endpoint.

Every model this evaluation talks to (the policy, the task judge, the screenshot
describer, the simulated user and the ask judge) is reached through an
OpenAI-compatible ``/v1`` URL. A local vLLM server needs no key, and the code has
always sent the placeholder ``"EMPTY"``. A hosted endpoint needs a real key. This
module picks that key from the environment, and nothing else: it never changes a
request's content.

How the key is chosen, for a given base URL:

1. If the URL equals one of the role URLs below, that role's key variable is used.
2. Otherwise ``AMBIGUI_API_KEY`` is used.
3. Otherwise the placeholder ``"EMPTY"`` is used (the previous behaviour).

=================  =========================  ===================
role               URL variable               key variable
=================  =========================  ===================
policy             ``POLICY_URL``             ``POLICY_API_KEY``
task judge         ``JUDGE_URL``              ``JUDGE_API_KEY``
describer          ``DESCRIBER_URL``          ``DESCRIBER_API_KEY``
simulated user     ``ANSWERER_ENDPOINT``      ``ANSWERER_API_KEY``
ask judge          ``ASK_JUDGE_URL``          ``ASK_JUDGE_API_KEY``
=================  =========================  ===================
"""

from __future__ import annotations

import os
from collections.abc import Mapping

#: The placeholder sent when no key is configured. vLLM accepts any value.
NO_KEY = "EMPTY"

#: (URL variable, key variable) for each role, in lookup order.
ROLE_VARS: tuple[tuple[str, str], ...] = (
    ("POLICY_URL", "POLICY_API_KEY"),
    ("JUDGE_URL", "JUDGE_API_KEY"),
    ("DESCRIBER_URL", "DESCRIBER_API_KEY"),
    ("ANSWERER_ENDPOINT", "ANSWERER_API_KEY"),
    ("ASK_JUDGE_URL", "ASK_JUDGE_API_KEY"),
)


def _norm(url: str) -> str:
    return (url or "").strip().rstrip("/")


def api_key_for(url: str | None, env: Mapping[str, str] | None = None) -> str:
    """The key for ``url``: its role's key, else ``AMBIGUI_API_KEY``, else ``"EMPTY"``."""
    source = os.environ if env is None else env
    target = _norm(url or "")
    if target:
        for url_var, key_var in ROLE_VARS:
            if _norm(source.get(url_var, "")) == target and source.get(key_var):
                return source[key_var]
    return source.get("AMBIGUI_API_KEY") or NO_KEY


def auth_headers(url: str | None, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """An ``Authorization`` header for plain HTTP calls, or ``{}`` when no key is set.

    Empty when no key is configured, so a call to a local server is exactly the call
    it was before.
    """
    key = api_key_for(url, env)
    return {} if key == NO_KEY else {"Authorization": f"Bearer {key}"}
