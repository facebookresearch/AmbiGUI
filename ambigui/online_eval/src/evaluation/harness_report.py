# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Whose fault was it, and how much did the harness help?

Two numbers a run has never reported, both of which turned a broken run into a
plausible-looking result.

**Whose fault.** `end_reason` mixes agent behaviour with our own breakage. A
rollout that stopped because the policy looped and one that stopped because the
serve rejected a sixth screenshot both read as a failed task. In the first
baseline that confusion hid an image cap: 212 of 384 rollouts died at turn four
and the run reported 15% task success as though the policy had earned it.

**How much we helped.** The harness repairs malformed tool calls -- it renames
`navigate` to `goto`, falls back from the structured call to parsing the text,
and tries parsers in turn until one bites. Every repair makes a policy look more
capable than it is. The offline decision eval already reports `strict_parse_rate`
for exactly this reason, and its own documentation says the number does not carry
over to the multi-step setting. This is that number for the online setting.

Neither is a judgement about the policy. Both are facts about the run that have
to travel with the score.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

#: The policy did something, and the thing it did ended the rollout.
AGENT: Final = "agent"
#: We broke. A serve rejected the request, a parser could not read a reply, the
#: element the agent named could not be matched to the page.
HARNESS: Final = "harness"
#: The world broke. A site was down, behind a bot wall, or refused us.
WORLD: Final = "world"

#: How each recorded ending is attributed. The keys are exactly the members of
#: :data:`evaluation.online_records.END_REASONS`, and a test asserts that, so adding an
#: end reason without classifying it fails rather than falling through.
#:
#: Anything absent is still attributed to the harness: an unclassified ending is our
#: bookkeeping gap, not the agent's fault, and assuming otherwise flatters the harness.
#:
#: Two vocabularies used to be mixed in here. Action-level statuses --
#: ``target_not_found``, ``withheld_high_stakes``, ``unknown_tool``,
#: ``type_failed_verify`` -- are not endings; they are per-call outcomes and are counted
#: by :func:`repair_of`. ``judge_error`` is a scoring failure, not a rollout ending.
#: Both were removed rather than left looking authoritative.
FAULT_OF: Final[dict[str, str]] = {
    "stop": AGENT,
    # The policy said it had finished. Its doing, exactly as `stop` is -- and no more
    # a verdict than `stop` is: this says who ended the rollout, not whether the task
    # was done. The judge answers that, and routinely disagrees.
    "finished": AGENT,
    # The agent pressed the control that commits the task and the harness absorbed
    # the press. Its doing, like `finished`, and no more a verdict than `finished`
    # is: this says the agent got to the commit, not that what it did before the
    # commit was right. The judge decides that.
    "commit_withheld": AGENT,
    "repeat_break": AGENT,
    "max_steps": AGENT,
    # Charged to US, not the agent. This fires when every parser we have failed to read
    # a call out of the generation, and we cannot tell a policy that emitted prose from
    # a policy whose format we failed to parse. The project has already shipped the
    # second: a launcher defaulting to the wrong tool-call format made every call
    # undecodable for a whole run. Charging that to the policy would have hidden it.
    "no_tool_call": HARNESS,
    # Our budget, not the policy's doing: it was still writing when the serve
    # stopped it.
    "truncated": HARNESS,
    "generation_error": HARNESS,
    "harness_error": HARNESS,
    # The concurrency deadline hit before the task returned. Our bookkeeping, not a
    # failure of the policy -- made explicit rather than left to the default.
    "unfinished": HARNESS,
    "env_unreachable": WORLD,
    # The page loaded and was unusable: a bot wall, a CloudFront denial, an
    # interstitial. Not the policy's doing and not ours -- we were never allowed in.
    "env_blocked": WORLD,
}


def fault_for(end_reason: str) -> str:
    """Who ended this rollout."""
    return FAULT_OF.get(str(end_reason or "").strip(), HARNESS)


#: Keys the environment sets on an action the safety guards absorbed or refused.
#: The action renderer reads the same keys, so both name the same thing.
WITHHELD: Final = "withheld_high_stakes"
#: ``result`` prefixes the environment writes for an action that did not take effect.
_REJECTED_PREFIXES: Final = ("error", "target_not_found", "playwright_error")
UNRESOLVED: Final = "unresolved_target"


def repair_of(action: Mapping[str, Any]) -> str | None:
    """What the harness had to do for one recorded action, or None if nothing.

    An action that never happened is not a clean call. Counting it as one would
    put the harness's own failures into the numerator of its format-fidelity
    rate, which is the one number that rate exists to keep honest.
    """
    if action.get(WITHHELD):
        return WITHHELD
    if action.get(UNRESOLVED):
        return UNRESOLVED
    status = str(action.get("status") or "").strip().lower()
    if status:
        return None if status == "ok" else status
    # The environment writes each action's outcome under ``result``. A call the
    # page or BrowserGym rejected is not a clean call. Counted here only: this
    # statistic is separate from what the task judges receive.
    result = action.get("result")
    if isinstance(result, str) and result.strip().lower().startswith(_REJECTED_PREFIXES):
        return "rejected"
    return None


@dataclass
class HarnessReport:
    """What the harness did to and for the policy over one run."""

    #: tool calls that arrived already well-formed and needed no repair
    clean_calls: int = 0
    #: tool calls the harness had to repair, by what the repair was
    repairs: Counter[str] = field(default_factory=Counter)
    #: actions whose named target could not be matched to anything on the page
    unresolved_targets: int = 0
    #: how each rollout ended
    endings: Counter[str] = field(default_factory=Counter)
    #: trajectory records the environment could not write. A rollout survives a
    #: bookkeeping failure, but the loss has to be visible: a judge that rots later can
    #: only be re-run against records that exist.
    trajectory_write_failures: list[str] = field(default_factory=list)
    #: Rows labelled Ambiguous that arrived with no open items. The user simulator
    #: has nothing to answer from and does not invent a value: such a question gets
    #: no answer. A Clear row with no items is correct and is not listed here.
    open_items_missing_on_ambiguous: list[str] = field(default_factory=list)

    def record_call(self, repair: str | None) -> None:
        """One tool call. ``repair`` names what we had to do, or None if nothing."""
        if repair:
            self.repairs[repair] += 1
            if repair == UNRESOLVED:
                self.unresolved_targets += 1
        else:
            self.clean_calls += 1

    def record_actions(self, actions: Iterable[Mapping[str, Any]]) -> None:
        """Every tool call of one rollout, as the environment recorded it.

        The counters above were defined, tested and never filled: the run built
        its report from the per-task summary rows, which carry no actions, so a
        rollout that dispatched five calls reported ``tool_calls: 0`` and the
        summary said none had been made. Feeding the actions in is what makes
        the format-fidelity line mean anything.
        """
        for action in actions:
            self.record_call(repair_of(action))

    def record_ending(self, end_reason: str) -> None:
        self.endings[str(end_reason or "unknown")] += 1

    @property
    def total_calls(self) -> int:
        return self.clean_calls + sum(self.repairs.values())

    @property
    def strict_call_rate(self) -> float | None:
        """Share of tool calls that needed no help. ``None`` if there were none.

        None is not zero. A run that emitted no tool calls at all has not shown
        us a 0% format rate; it has shown us nothing, and the two should not be
        averaged together.
        """
        if not self.total_calls:
            return None
        return self.clean_calls / self.total_calls

    def faults(self) -> Counter[str]:
        out: Counter[str] = Counter()
        for reason, n in self.endings.items():
            out[fault_for(reason)] += n
        return out

    @property
    def ours(self) -> float | None:
        """Share of rollouts that ended because of us rather than the policy."""
        total = sum(self.endings.values())
        if not total:
            return None
        return (self.faults()[HARNESS] + self.faults()[WORLD]) / total

    def as_dict(self) -> dict[str, object]:
        return {
            "tool_calls": self.total_calls,
            "clean_calls": self.clean_calls,
            "repaired_calls": sum(self.repairs.values()),
            "strict_call_rate": self.strict_call_rate,
            "repairs_by_kind": dict(self.repairs),
            "unresolved_targets": self.unresolved_targets,
            "endings": dict(self.endings),
            "faults": dict(self.faults()),
            "not_the_policys_fault": self.ours,
            "trajectory_write_failures": len(self.trajectory_write_failures),
            "open_items_missing_on_ambiguous": len(
                self.open_items_missing_on_ambiguous
            ),
        }

    def summary(self) -> str:
        """The paragraph a reader needs before believing the score."""
        lines = []
        rate = self.strict_call_rate
        if rate is None:
            lines.append(
                "  tool calls: none were made, so format fidelity is not measured"
            )
        else:
            lines.append(
                f"  tool calls: {self.clean_calls}/{self.total_calls} arrived usable "
                f"({100 * rate:.0f}%); the harness repaired {sum(self.repairs.values())}"
            )
            for kind, n in self.repairs.most_common():
                lines.append(f"      {n:5d}  {kind}")
        if self.unresolved_targets:
            lines.append(
                f"  the agent named {self.unresolved_targets} element(s) we could not "
                "find on the page"
            )
        if self.trajectory_write_failures:
            lines.append(
                f"  {len(self.trajectory_write_failures)} rollout(s) left no trajectory "
                "record; a re-judge cannot reach them"
            )
            for failure in self.trajectory_write_failures[:3]:
                lines.append(f"      {failure}")
        if self.open_items_missing_on_ambiguous:
            lines.append(
                f"  {len(self.open_items_missing_on_ambiguous)} Ambiguous row(s) reached "
                "the simulator with no curated open items; a clarifying question on "
                "them gets no answer (the policy reads '(no answer)')"
            )
        ours = self.ours
        if ours is not None:
            faults = self.faults()
            lines.append(
                f"  rollouts ended by: the policy {faults[AGENT]}, "
                f"the harness {faults[HARNESS]}, the world {faults[WORLD]}"
            )
            if ours > 0.2:
                lines.append(
                    f"  WARNING: {100 * ours:.0f}% of rollouts ended for reasons that "
                    "are not the policy's. Treat the score as a lower bound on it."
                )
        return "\n".join(lines)
