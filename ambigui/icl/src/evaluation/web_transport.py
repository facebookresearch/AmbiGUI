# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""How the browser reaches the web, and the fallback for when it cannot.

Two separate things decide whether a rollout ever sees its start page.

**The inherited proxy, which is cleared.** Jobs here start with an ``HTTPS_PROXY``
set to a host that does not accept connections. Chromium honours it, so every page
load times out, and a timeout is indistinguishable to this harness from a site
refusing us -- it files ``dns_or_connect_fail`` and sets the rollout aside. That is
most of what the arms were discarding: pages recorded as unreachable load normally
once the variable is gone. See :func:`clear_inherited_proxy_env`.

**The fallback proxy, for pages that really are blocked.** Bot walls and interstitials
remain after the above, and a residential proxy is the standard answer. This module is
the seam for one. Two things about how it is used here:

**It is a fallback, not a transport.** Every rollout is attempted directly first. The
proxy is used only when the direct attempt produced a blocker, so a run pays the proxy
for the pages that need it and nothing else, and a comparison against a run without
one stays meaningful for the pages that were reachable either way.

**It is off unless configured, and it refuses a half-configuration.** An unset proxy
means "no fallback", which is the behaviour every run so far has had. A *partly* set
one -- a host with no password, say -- is not a weaker fallback, it is a silent
failure to connect that would be recorded as the site blocking us. That is exactly the
shape of fault this project keeps finding, so it raises instead.

Nothing here holds a credential. The values come from the environment.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Final

#: The environment variables that configure the fallback. ``SERVER`` alone is enough
#: for an unauthenticated proxy; a username and a password must be given together.
SERVER_VAR = "BRIGHTDATA_PROXY_SERVER"
USERNAME_VAR = "BRIGHTDATA_PROXY_USERNAME"
PASSWORD_VAR = "BRIGHTDATA_PROXY_PASSWORD"

#: What the record calls each route, so a finished run can say how many of its
#: rollouts needed the fallback and how many were reachable without it.
DIRECT = "direct"
FALLBACK = "brightdata"


class ProxyHalfConfigured(ValueError):
    """Some of the proxy settings are present and some are not.

    Treated as an error rather than as "no proxy". A proxy that is reached for and
    not there fails as a connection error, which this harness records as the site
    blocking us -- so the misconfiguration would appear in the results as a worse
    web, and the fallback would look like it had been tried and not helped.
    """


@dataclass(frozen=True)
class Proxy:
    """Playwright's proxy settings, in the shape it accepts."""

    server: str
    username: str = ""
    password: str = ""

    def as_playwright(self) -> dict[str, str]:
        settings = {"server": self.server}
        if self.username:
            settings["username"] = self.username
            settings["password"] = self.password
        return settings


def configured(env: dict[str, str] | None = None) -> Proxy | None:
    """The configured fallback proxy, or ``None`` when there is none.

    Raises :exc:`ProxyHalfConfigured` when the settings are partly present.
    """
    source = os.environ if env is None else env
    server = (source.get(SERVER_VAR) or "").strip()
    username = (source.get(USERNAME_VAR) or "").strip()
    password = (source.get(PASSWORD_VAR) or "").strip()

    if not server:
        if username or password:
            raise ProxyHalfConfigured(
                f"{USERNAME_VAR} or {PASSWORD_VAR} is set but {SERVER_VAR} is not. "
                f"A credential with nowhere to send it is not a fallback."
            )
        return None
    if bool(username) != bool(password):
        raise ProxyHalfConfigured(
            f"{SERVER_VAR} is set but exactly one of {USERNAME_VAR} and "
            f"{PASSWORD_VAR} is. An authenticated proxy given half its credentials "
            f"refuses the connection, and this harness records a refused connection "
            f"as the site blocking us."
        )
    return Proxy(server=server, username=username, password=password)


#: The proxy settings a batch job inherits from the cluster's own environment.
#: Chromium reads these at launch on Linux, so whatever they name is what every page
#: load goes through, whether or not anything is listening there.
INHERITED_PROXY_VARS: Final[tuple[str, ...]] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

_announced = False


def clear_inherited_proxy_env(
    env: MutableMapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Drop inherited proxy settings so the browser reaches the web directly.

    Returns the names that were removed, in :data:`INHERITED_PROXY_VARS` order.

    This exists because of a specific, measured failure. Jobs on this cluster inherit
    an ``HTTPS_PROXY`` pointing at a host that does not accept connections; chromium
    honours it, so every page load ends in ``ERR_TIMED_OUT``. The harness has no way
    to tell that apart from a site refusing us, so it records the refusal: in one
    training arm 3,168 of 4,576 rollouts were set aside as ``dns_or_connect_fail``
    and the run trained on what was left. The same pages load normally once these
    are unset.

    It is cleared here rather than in the launch script because a launch script is
    one a future run can be started without. The fallback proxy in this module is
    unaffected -- that one is passed to the browser explicitly, not through the
    environment, and is reached for only when a page is genuinely blocked.
    """
    global _announced
    source = os.environ if env is None else env
    removed = tuple(
        name for name in INHERITED_PROXY_VARS if source.pop(name, None) is not None
    )
    if removed and not _announced:
        # Said once per process: a run that silently changed how it reaches the web
        # is one whose results cannot be compared against a run that did not.
        _announced = True
        print(
            f"[web_transport] browsing directly; dropped inherited {', '.join(removed)}",
            flush=True,
        )
    return removed
