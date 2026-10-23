# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rendering a rollout's actions so a judge can read them.

Separated from the judge that first needed it: the current task-success judge
(:mod:`evaluation.lookup_judge`) formats its action history with these, and the
judge they were written for is now in ``deprecated/``.

The markers matter more than the formatting. An action the page refused looks,
in a bare list of attempts, exactly like one that worked -- so a run of
silently failed clicks would read as a success.
"""

from __future__ import annotations

from evaluation.action_guards import WITHHELD_COMMIT as _WITHHELD_COMMIT

from collections.abc import Sequence
from typing import Any

#: Marker the environment puts on an action it refused to execute.
#: The tool name the environment records when the policy calls the finish verb.
#: Written here rather than imported from the environment so that rendering a
#: trace does not pull in a browser.
FINISH_TOOL = "FINISH"

WITHHELD = "withheld_high_stakes"

#: Which rule withheld it. A commit is the agent reaching the end of the task and
#: the harness declining to press the button; a credential field or an off-domain
#: hop is the agent being stopped from going somewhere, and did not happen. Telling
#: the judge to credit the second as performed would inflate task success.
WITHHELD_KIND = "withheld_kind"

#: Re-exported so the renderer and the guard cannot disagree on the spelling.
WITHHELD_COMMIT = _WITHHELD_COMMIT

#: Marker the environment puts on an action whose target it could not resolve.
UNRESOLVED = "unresolved_target"

#: Statuses that mean the action did not take effect.
FAILED_PREFIXES = ("target_not_found", "playwright_error", "error")


def render_actions(actions: Sequence[dict[str, Any]]) -> str:
    """The trace as a numbered list a reader can follow."""
    lines = []
    for i, action in enumerate(actions, 1):
        tool = str(action.get("tool", "")).lower()
        args = action.get("args") or {}
        if tool == "ask":
            lines.append(f'{i}. asked the user: "{args.get("question", "")}"')
        elif tool == "stop":
            lines.append(f"{i}. declared the task finished")
        elif tool == FINISH_TOOL.lower():
            # The summary the agent wrote is deliberately not rendered. It is the
            # agent's description of what it believes it achieved, and a reader
            # given it tends to treat it as a report of what happened. Success is
            # the judge's verdict on the screens, so the claim is shown as a claim
            # and its content is withheld.
            lines.append(
                f"{i}. claimed the task was finished "
                f"[A CLAIM BY THE AGENT, NOT EVIDENCE. It ends the episode and "
                f"nothing more. Judge whether the task was done from the screens "
                f"and the actions that were actually performed.]"
            )
        else:
            detail = ", ".join(f"{k}={v!r}" for k, v in args.items())
            line = f"{i}. {tool} {detail}".rstrip()
            if action.get(WITHHELD):
                # Self-contained wording. The "see the note below" this used to say
                # refers to a note in the retired trace judge's prompt; the live
                # lookup judge has no such note, so the line carries its own
                # meaning. Branched on the rule, because only one of the three is
                # the agent having done the thing.
                # An absent kind is a record written before the kind existed, or
                # by the retired training environment, which marks every refusal
                # and names none. Those were overwhelmingly commits and were
                # judged as performed, so absent reads as it did then: changing it
                # would make a re-judge disagree with the run it came from.
                if action.get(WITHHELD_KIND, WITHHELD_COMMIT) == WITHHELD_COMMIT:
                    line += (
                        "   [THE AGENT DID THIS; THE HARNESS DID NOT EXECUTE IT. "
                        "This control commits a real action on a live site, so "
                        "the harness absorbed the press. Treat it as performed "
                        "by the agent.]"
                    )
                else:
                    line += (
                        "   [DID NOT HAPPEN: a safety rule refused it. Do not "
                        "credit it.]"
                    )
            elif action.get(UNRESOLVED):
                line += f"   [DID NOT HAPPEN: {action[UNRESOLVED]}]"
            else:
                status = str(action.get("status") or "")
                if status and status.startswith(FAILED_PREFIXES):
                    line += f"   [DID NOT HAPPEN: {status}]"
            after = str(action.get("url_after") or "")
            if after:
                line += f"\n     -> page: {after}"
                count = action.get("n_interactives_after")
                if count is not None:
                    line += f"  ({count} controls)"
            lines.append(line)
    return "\n".join(lines) or "(the agent took no actions)"
