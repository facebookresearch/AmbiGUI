# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The one way this package talks to a serve.

Both the policy call and the judge call post the same shape to the same kind of endpoint, so
they share one poster rather than carrying a copy each. Every caller can inject its own, which
is how the tests run without an HTTP library and without a serve.
"""

from __future__ import annotations

import os
from typing import Any

#: The environment variable that holds the key for each role's endpoint. Each role can
#: point at a different provider, so each has its own key. Unset or empty means "send no
#: key", which is what a local vLLM serve expects.
POLICY_API_KEY_ENV = "AMBIGUI_POLICY_API_KEY"
ANSWERER_API_KEY_ENV = "AMBIGUI_ANSWERER_API_KEY"
JUDGE_API_KEY_ENV = "AMBIGUI_JUDGE_API_KEY"


def api_key_from_env(name: str) -> str | None:
    """The key in environment variable ``name``, or ``None`` when it is unset or empty."""
    return os.environ.get(name) or None


def default_post(
    url: str, payload: dict[str, Any], timeout: float, api_key: str | None = None
) -> Any:
    """POST ``payload`` as JSON and return the decoded body, raising on a non-2xx.

    ``api_key`` is sent as ``Authorization: Bearer <key>`` when it is given. A local vLLM
    serve needs no key, so the default sends no header at all, exactly as before. A hosted
    OpenAI-compatible endpoint needs one; the caller reads it from an environment variable
    (see ``docs/offline.md``), never from a file in the repository.

    ``trust_env`` is off, so the environment's ``http_proxy`` is ignored. Some compute
    clusters export a proxy whose ``no_proxy`` list is written as a CIDR block. A serve
    addressed by hostname does not match a CIDR, so every call is routed through a proxy
    that cannot reach it, times out, and returns an empty body. That reads exactly like a
    dead serve. ``evaluation/serve_preflight.py`` turns it off too, so a preflight and the
    real calls always take the same route.

    ``trust_env`` also governs ``REQUESTS_CA_BUNDLE`` and ``.netrc``, so both are ignored
    here too. A credential is passed explicitly through ``api_key`` instead.

    ``requests`` is imported here rather than at module scope so that marking an
    already-finished run needs no HTTP library installed.
    """
    import requests  # noqa: PLC0415

    with requests.Session() as session:
        session.trust_env = False
        if api_key:
            response = session.post(
                url,
                json=payload,
                timeout=timeout,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        else:
            response = session.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
