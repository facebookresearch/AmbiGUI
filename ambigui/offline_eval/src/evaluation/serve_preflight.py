# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Stop a run immediately when the model server it needs is not usable.

A failed policy call in this package is recorded as an empty reply and scored as a wrong
step. That is the right behaviour for one item and the wrong behaviour for a whole run:
the program finishes, exits successfully, and writes a complete-looking file in which
every reply is empty. The numbers then read as a bad model rather than a broken setup.

There are three ways a server is unusable and all three look identical from inside the
program:

* it is not running, or the job holding it ended;
* it moved to a different machine, so the address is stale;
* it is running but publishing the model under a different name, so every request is
  rejected as "model not found".

So: make one real request before doing any work, and fail on it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any


class ServeDead(RuntimeError):
    """The serve is unreachable, served under another name, or answering with nothing."""


#: The probe's token budget. The probe asks for one word, so eight tokens were enough for
#: a model that answers directly, and far too few for one that reasons first: it spends
#: the whole budget inside an unclosed ``<think>``, the serve returns a well-formed
#: response with no text in either field, and the rule below -- a 200 holding nothing
#: means dead -- fails a serve that is working. The budget is raised instead of the rule
#: being relaxed, because the rule is what catches a serve whose weights never loaded,
#: and a probe that accepts an empty reply catches nothing. The cost is one short
#: generation per serve per run, paid once before any work.
PROBE_MAX_TOKENS = 1024


def _default_post(
    url: str, payload: dict[str, Any], timeout: float, api_key: str | None = None
) -> Any:
    """POST via ``requests``, imported here rather than at module scope.

    A module-level import would make every program that imports this package need an
    HTTP library, including the ones that only score finished runs.

    ``api_key``, when given, is sent as ``Authorization: Bearer <key>``; with no key no
    header is sent, which is what a local vLLM serve expects.
    """
    import requests  # noqa: PLC0415

    # A serve on a private network must never go through a proxy. With HTTP_PROXY set,
    # requests routes a private address through the proxy, every probe raises
    # ProxyError, and the result is "all serves dead" -- the exact misleading verdict this
    # check exists to prevent. trust_env=False ignores the environment. The policy and
    # judge calls (offline_evaluation/http.py) and the simulated user's calls
    # (cascp/common/llm.py) do the same, so the probe and the real calls always take the
    # same route.
    with requests.Session() as session:
        session.trust_env = False
        if api_key:
            return session.post(
                url,
                json=payload,
                timeout=timeout,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        return session.post(url, json=payload, timeout=timeout)


def assert_endpoint_live(
    endpoint: str,
    model: str,
    *,
    timeout: float = 30.0,
    post: Callable[[str, dict[str, Any], float], Any] = _default_post,
    api_key: str | None = None,
) -> None:
    """Send one real completion to ``endpoint`` as ``model``, or raise :class:`ServeDead`.

    A non-200 is reported with its body, because the
    body is what distinguishes "serve is down" from "serve is up under another name".

    ``post`` is injectable so a caller can probe without an HTTP library at all.
    ``api_key`` is the endpoint's key (see :mod:`offline_evaluation.http`); it is bound
    onto the default poster only, so with no key the probe is exactly the keyless one.
    """
    if api_key and post is _default_post:
        post = functools.partial(_default_post, api_key=api_key)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "temperature": 0.0,
        "max_tokens": PROBE_MAX_TOKENS,
    }
    try:
        resp = post(f"{endpoint}/chat/completions", payload, timeout)
    except Exception as exc:  # noqa: BLE001 - any failure here must end the run
        raise ServeDead(
            f"serve not usable: model={model!r} endpoint={endpoint} "
            f"-> {type(exc).__name__}: {str(exc)[:200]}"
        ) from exc
    if resp.status_code != 200:
        # A dead port answers with HTML or nothing, not JSON, so the body is read as text.
        raise ServeDead(
            f"serve returned HTTP {resp.status_code}: model={model!r} "
            f"endpoint={endpoint} -> {resp.text[:200]}"
        )
    if not _reply_text(resp).strip():
        # A 200 is not an answer. A serve whose weights failed to load, or whose context
        # is already full, returns a well-formed response holding nothing, and every row
        # of the run then records the empty string the client turns that into.
        raise ServeDead(
            f"serve answered with nothing: model={model!r} endpoint={endpoint}"
        )


def _reply_text(resp: Any) -> str:
    """The assistant's text in a chat-completions body, or ``""`` if there is none."""
    try:
        message = resp.json()["choices"][0]["message"]
        # A reasoning model that spent its whole budget thinking puts its reply in
        # `reasoning_content`, and reading `content` alone would call that serve dead.
        return str(message.get("content") or message.get("reasoning_content") or "")
    except Exception:  # noqa: BLE001 - an unreadable body is an empty reply
        return ""
