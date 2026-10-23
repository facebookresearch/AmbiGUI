# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""The user agent the browser announces to the sites it visits.

Chromium launched headless announces itself as ``HeadlessChrome``, and a measurable
share of sites refuse that string. Over 60 benchmark start pages, five of the seven
outright blocks cleared when the marker was removed -- three HTTP 403 walls and two
CloudFront errors -- with the same two survivors on a repeat run. Raising the
navigation timeout and matching the browser's locale, timezone and viewport to the
region we egress from were measured the same way and moved nothing.

The marker is the only thing removed. The version and every other field are read
from the binary making the request, so the header stays an accurate description of
the client rather than a claim to be a browser we are not running.
"""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache

#: What Chromium on this platform reports once the headless marker is gone. Only the
#: version varies between builds; the rest of the string is fixed for Linux x86_64.
_TEMPLATE = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/{version} Safari/537.36"
)
#: ``--version`` prints a name then a dotted version, e.g.
#: "Google Chrome for Testing 151.0.7922.34".
_VERSION = re.compile(r"\d+(?:\.\d+)+")
#: Asking a binary its version is immediate; this only bounds a hung executable.
_VERSION_TIMEOUT_S = 30


def chromium_version(executable: str) -> str:
    """The dotted version ``executable`` reports, or ``""`` if it cannot be asked."""
    if not executable:
        return ""
    try:
        printed = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_TIMEOUT_S,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    found = _VERSION.search(printed)
    return found.group(0) if found else ""


def chromium_executable() -> str:
    """Path to the Chromium Playwright would launch, or ``""`` if it cannot be found.

    Reads the path without launching a browser: entering the Playwright context
    starts its driver and nothing heavier.
    """
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415

        with sync_playwright() as pw:
            return str(pw.chromium.executable_path)
    except Exception:  # noqa: BLE001 - no Playwright, no override; the default stands
        return ""


@lru_cache(maxsize=None)
def user_agent() -> str:
    """The running Chromium's user agent without the headless marker.

    Returns ``""`` when the version cannot be read, which leaves the browser's own
    default in place: announcing a version we did not confirm would be a claim about
    the client rather than a description of it.
    """
    version = chromium_version(chromium_executable())
    if not version:
        return ""
    return _TEMPLATE.format(version=version)
