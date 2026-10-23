# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The chat client the simulated user (the answerer) and the ask judge call their model through.

- ``MODEL_CONFIG`` -- per-model inference settings, keyed by the **served model name**. A
  name not in the table gets ``_default``.
- ``call_llm`` -- one text chat completion, with bounded retries.

The endpoint is any OpenAI-compatible chat API, given as a base URL ending in ``/v1``: a
local vLLM serve or a hosted API. Its key comes from the environment
(``evaluation/endpoint_auth.py``); a local vLLM serve takes the placeholder ``EMPTY``. The
``openai`` client is imported when a call is made.

One HTTP call may take up to 900 seconds (the paper runs' setting).
"""

from __future__ import annotations

from typing import Any


# Per-model inference config. Keys are the vLLM ``served-model-name`` (the ``model=``
# argument in chat calls). ``_default`` is the fallback for an unconfigured model.
MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "_default": {
        "max_tokens": 1500,
        "temperature": 0.0,
        "notes": "safe defaults for instruction-tuned VLMs with no reasoning trace",
    },
}


def get_model_config(model_name: str) -> dict[str, Any]:
    """Return the per-model inference config, falling back to ``_default``."""
    return MODEL_CONFIG.get(model_name, MODEL_CONFIG["_default"])


def _make_client(serve_url: str):  # type: ignore[no-untyped-def]
    from openai import OpenAI

    # The key is one the environment configures for this URL
    # (evaluation/endpoint_auth.py), else the placeholder ``EMPTY``.
    from evaluation.endpoint_auth import api_key_for

    # Bound each HTTP call, so one slow or hung serve cannot stall the caller.
    return OpenAI(
        base_url=serve_url,
        api_key=api_key_for(serve_url),
        timeout=900.0,
        max_retries=0,
    )


def call_llm(
    prompt: str,
    serve_url: str,
    model_name: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    retries: int = 2,
) -> str:
    """Call the served model through OpenAI-compatible chat completions (text only).

    Args:
        prompt: the user-message text.
        serve_url: base URL of the OpenAI-compatible endpoint, ending in ``/v1``.
        model_name: served-model name (the ``model=`` field; also the ``MODEL_CONFIG`` key).
        max_tokens: response length ceiling; ``None`` uses the model's ``MODEL_CONFIG`` value.
        temperature: sampling temperature; ``None`` uses the model's ``MODEL_CONFIG`` value.
        retries: number of retries on transient errors (so ``retries + 1`` total attempts),
            with exponential backoff.

    Returns the raw response text.
    """
    cfg = get_model_config(model_name)
    if max_tokens is None:
        max_tokens = cfg["max_tokens"]
    if temperature is None:
        temperature = cfg["temperature"]

    client = _make_client(serve_url)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
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
