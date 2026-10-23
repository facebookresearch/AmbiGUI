# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Actions a live run must not perform, whatever the policy asks for.

These run against real websites with real side effects. Three classes of action are
withheld: committing ones (buying, paying, logging in), credential and payment
fields, and navigation off the site the task started on. The call is counted and
reported to the policy as withheld, not silently dropped — a policy told nothing
happened repairs the wrong thing, and a run that drops actions quietly cannot be
distinguished from one that had nothing to do.

The checks are pure functions of the action and the page, so they can be tested
without a browser. They moved here out of the project's own browser harness, which
BrowserGym replaced; the phrase lists are unchanged, because they are the ones the
withheld-commit counts already published were measured with.

Naming is by BrowserGym verb. ``click`` and ``dblclick`` commit; ``fill`` both
commits and can expose a secret; ``goto`` navigates.
"""

from __future__ import annotations

from typing import Final

from urllib.parse import urlparse

#: Verbs that can commit the user to something by pressing a control.
COMMITTING: frozenset[str] = frozenset({"click", "dblclick"})

#: Verbs that put text into a field.
FILLING: frozenset[str] = frozenset({"fill"})

_COMMIT_PHRASES: tuple[str, ...] = (
    "place order",
    "place your order",
    "buy now",
    "purchase",
    "complete purchase",
    "checkout",
    "check out",
    "proceed to checkout",
    "proceed to payment",
    "continue to payment",
    "pay now",
    "make payment",
    "submit payment",
    "confirm and pay",
    "confirm order",
    "submit order",
    "complete order",
    "confirm booking",
    "confirm reservation",
    "book now",
    "reserve now",
    "complete booking",
    "place bid",
    "sign in",
    "log in",
    "login",
    "sign up",
)


#: Broader state-changing actions, withheld ONLY on tasks the gold panel labelled
#: major or critical. They collide with benign navigation — "apply filter", "submit
#: search" — so they are gated on stakes rather than blocked outright. Multi-word
#: for the same reason.
_SIDE_EFFECT_PHRASES: tuple[str, ...] = (
    "submit order",
    "submit payment",
    "submit review",
    "submit application",
    "submit request",
    "submit rating",
    "post review",
    "post comment",
    "post now",
    "publish",
    "send message",
    "send email",
    "send request",
    "send invite",
    "delete",
    "remove account",
    "cancel order",
    "cancel reservation",
    "cancel subscription",
    "unsubscribe",
    "subscribe",
    "apply now",
    "apply for",
    "confirm order",
    "confirm reservation",
    "confirm payment",
    "add payment",
    "add card",
    "agree and continue",
    "accept and continue",
)


#: Never typed into a live field, whatever the label on the control says.
_SENSITIVE_FIELD_TOKENS: tuple[str, ...] = (
    "password",
    "passwd",
    "card number",
    "cardnumber",
    "credit card",
    "cvv",
    "cvc",
    "security code",
    "card code",
    "expiry",
    "expiration",
    "ssn",
    "social security",
    "routing number",
    "account number",
)


def is_high_stakes_commit(verb: str, name: str, severity: str | None = None) -> bool:
    """Whether pressing ``name`` would commit the user to something.

    ``name`` is the accessible name of the element the action addresses, looked up
    from its bid. Purchase, payment and login phrases are always withheld; the
    broader side-effect phrases only on a major or critical task.
    """
    if verb not in COMMITTING:
        return False
    target = name.lower()
    if any(p in target for p in _COMMIT_PHRASES):
        return True
    return str(severity).lower() in ("major", "critical") and any(
        p in target for p in _SIDE_EFFECT_PHRASES
    )


def is_sensitive_field(name: str) -> bool:
    """Whether ``name`` is a credential or payment field that must never be filled."""
    lowered = name.lower()
    return any(token in lowered for token in _SENSITIVE_FIELD_TOKENS)


def domain_of(url: str) -> str:
    """The bare host of ``url``: lowercased, without ``www.``, port or userinfo.

    Empty when the URL cannot be parsed, which callers read as "no opinion" rather
    than as a mismatch.
    """
    net = urlparse(url if "//" in url else "//" + url).netloc.lower()
    net = net.split("@")[-1].split(":")[0]
    return net[4:] if net.startswith("www.") else net


def is_offdomain_goto(url: str, start_domain: str) -> bool:
    """Whether a navigation leaves the site the task started on.

    Subdomains either way are allowed. This keeps a live agent on the task's own
    site instead of out on the open web.
    """
    if not start_domain:
        return False
    target = domain_of(url)
    if not target:
        return False
    return not (
        target == start_domain
        or target.endswith("." + start_domain)
        or start_domain.endswith("." + target)
    )


#: The rule that withheld an action. Not interchangeable: a commit is the agent
#: reaching the end of the task on a live site and the episode ends there, while a
#: credential field or an off-domain hop is the agent going somewhere it may not and
#: the episode carries on.
WITHHELD_COMMIT: Final[str] = "commit"
WITHHELD_FIELD: Final[str] = "sensitive_field"
WITHHELD_OFFDOMAIN: Final[str] = "offdomain"


def withheld_kind(
    verb: str,
    *,
    name: str = "",
    url: str = "",
    start_domain: str = "",
    severity: str | None = None,
    commit_name: str = "",
) -> str:
    """Which rule withholds this action, or ``""`` if none does.

    ``commit_name`` is the control the task row names as its commit, when it has
    one. Matched exactly it beats :func:`is_high_stakes_commit`, which is a name
    heuristic gated on stakes: the heuristic exists to catch commits nothing told
    us about, and a row that names its own control has told us outright.
    """
    named = bool(commit_name) and name.strip().casefold() == (
        commit_name.strip().casefold()
    )
    # A named control still has to clear the stakes bar. Letting the name alone
    # decide meant a row naming a generic label -- "Visit", "Actions" -- ended
    # the episode on benign navigation and credited the agent with reaching a
    # commit it never reached. Minor-severity tasks commit nothing worth
    # withholding, by definition of the label.
    if verb in COMMITTING and named and (severity or "").lower() != "minor":
        return WITHHELD_COMMIT
    if is_high_stakes_commit(verb, name, severity):
        return WITHHELD_COMMIT
    if verb in FILLING and is_sensitive_field(name):
        return WITHHELD_FIELD
    if verb == "goto" and is_offdomain_goto(url, start_domain):
        return WITHHELD_OFFDOMAIN
    return ""


def withheld_reason(
    verb: str,
    *,
    name: str = "",
    url: str = "",
    start_domain: str = "",
    severity: str | None = None,
    commit_name: str = "",
) -> str:
    """Why this action must not be performed, or ``""`` if it may be.

    One entry point so the environment has a single place to ask, and so the reason
    it reports to the policy is the reason that was checked. The wording is built
    from :func:`withheld_kind`'s verdict rather than from a second pass over the
    rules, so the sentence and the decision cannot disagree.
    """
    kind = withheld_kind(
        verb,
        name=name,
        url=url,
        start_domain=start_domain,
        severity=severity,
        commit_name=commit_name,
    )
    if kind == WITHHELD_COMMIT:
        return f"withheld: {name!r} would commit a real action on a live site"
    if kind == WITHHELD_FIELD:
        return f"withheld: {name!r} is a credential or payment field"
    if kind == WITHHELD_OFFDOMAIN:
        return (
            f"withheld: {url!r} leaves {start_domain!r}, the site this task started on"
        )
    return ""
