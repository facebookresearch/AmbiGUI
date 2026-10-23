# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The chat client the simulated user (the answerer) calls its model through.

- ``MODEL_CONFIG`` -- per-model inference settings, keyed by the **served model name**.
  The name the answerer is served under therefore chooses its token budget: a name not in
  the table gets ``_default``. To reproduce a published run, serve the answerer under the
  exact name that run recorded (see ``docs/offline.md``).
- ``call_llm`` -- one text chat completion, with bounded retries.

The endpoint is any OpenAI-compatible chat API, given as a base URL ending in ``/v1``: a
local vLLM serve of the Hugging Face weights (``vllm serve <hf-id> --revision <commit>
--served-model-name <name>``) or a hosted API. The ``openai`` client is imported when a
call is made.

**Same route as the preflight.** The answerer's start-up check
(:func:`evaluation.serve_preflight.assert_endpoint_live`) and the policy and judge calls
use ``requests`` with ``trust_env`` off, so they ignore ``HTTP_PROXY``, ``HTTPS_PROXY``,
``NO_PROXY`` and the other environment settings. The answerer's real calls are made the
same way here: the ``openai`` client gets an HTTP client with ``trust_env`` off. Without
this the check and the real calls could take different routes, and one could pass while
the other fails.

Environment variables:

- ``AMBIGUI_ANSWERER_API_KEY`` -- the key for a hosted endpoint. Unset or empty means
  no ``Authorization`` header is sent, which is what a local vLLM serve expects.
- ``AMBIGUI_ANSWERER_HTTP_TIMEOUT_S`` -- seconds allowed for one HTTP call. Default: ``60``.
"""

from __future__ import annotations

import os
from typing import Any


# Per-model inference config. Keys are the vLLM ``served-model-name`` (the ``model=``
# argument in chat calls). ``_default`` is the fallback for an unconfigured model.
MODEL_CONFIG: dict[str, dict[str, Any]] = {
    # Instruction-tuned models with no reasoning trace.
    "_default": {"max_tokens": 1500},
    # GLM-4.6V (vision+text, reasoning) emits a <think> trace before its answer, so it
    # needs a large budget (TP=4, MAX_MODEL_LEN=32768).
    "glm_4_6v": {"max_tokens": 8192},
}


def get_model_config(model_name: str) -> dict[str, Any]:
    """Return the per-model inference config, falling back to ``_default``."""
    return MODEL_CONFIG.get(model_name, MODEL_CONFIG["_default"])


#: The environment variable holding the answerer endpoint's key. It has the same name as
#: :data:`offline_evaluation.http.ANSWERER_API_KEY_ENV`, which the runner's preflight reads.
ANSWERER_API_KEY_ENV = "AMBIGUI_ANSWERER_API_KEY"


#: The environment variable holding the seconds allowed for one answerer HTTP call.
HTTP_TIMEOUT_ENV = "AMBIGUI_ANSWERER_HTTP_TIMEOUT_S"

#: The key the ``openai`` client is built with when no key is set. The client needs a
#: non-empty value; the ``Authorization`` header it would carry is removed before the
#: request is sent (see :func:`_drop_authorization`), so this value never leaves the
#: process.
_NO_KEY = "EMPTY"


def _api_key() -> str | None:
    """The answerer endpoint's key, or ``None`` when it is unset or empty."""
    return os.environ.get(ANSWERER_API_KEY_ENV) or None


def _drop_authorization(request: Any) -> None:
    """Remove the ``Authorization`` header from an outgoing request (no key is set)."""
    request.headers.pop("Authorization", None)


def _make_client(serve_url: str):  # type: ignore[no-untyped-def]
    import openai

    key = _api_key()
    hooks = {} if key else {"request": [_drop_authorization]}
    # trust_env=False: the same proxy handling as the preflight (see the module docstring).
    http_client = openai.DefaultHttpxClient(trust_env=False, event_hooks=hooks)
    # Bound each HTTP call, so one slow or hung serve cannot stall the caller.
    return openai.OpenAI(
        base_url=serve_url,
        api_key=key or _NO_KEY,
        timeout=float(os.getenv(HTTP_TIMEOUT_ENV, "60")),
        max_retries=0,
        http_client=http_client,
    )


def call_llm(prompt: str, serve_url: str, model_name: str) -> str:
    """Call the served model through OpenAI-compatible chat completions (text only).

    ``model_name`` is the served-model name (the ``model=`` field, and the ``MODEL_CONFIG``
    key that sets the token budget). Greedy (temperature 0). Two retries on a transient
    error, with exponential backoff. Returns the raw response text.
    """
    max_tokens = get_model_config(model_name)["max_tokens"]
    retries = 2
    with _make_client(serve_url) as client:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - retried, then re-raised below
                last_exc = e
                if attempt < retries:
                    import time

                    time.sleep(2 ** (attempt + 1))
                    continue
                raise
    raise RuntimeError(f"call_llm({model_name}) exhausted retries: {last_exc}")
