# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A rollout of an Online-Mind2Web task in a BrowserGym environment.

This replaces the project's own browser harness. That harness offered seven actions
addressed by the visible text of a target and resolved the text against a list of
elements it had shown the policy. The resolution could not be made sound: page text
is not unique (five fields on one site were all called ``on``) and the rendered text
is not the accessible name (``SAIL TO`` on screen is ``Sail To`` to the browser), so
every fix widened the matching — substrings, case folding, collapsing duplicates —
until the harness was aiming the policy's actions for it and reporting success for
actions that landed somewhere else. A harness that compensates for imprecise tool
calls cannot measure whether the policy makes precise ones.

BrowserGym removes the class of fault rather than another instance of it. Elements
are addressed by ``bid``, a handle stamped on every node of the accessibility tree
the policy is shown. A bid is either on the page or it is not; there is nothing to
match and nothing to guess. Its action set is also wide enough to express what the
benchmark's own reference trajectories do, which the seven-action set was not —
those trajectories contain hover steps.

What BrowserGym does not supply, and this module adds, is everything specific to the
experiment rather than to driving a browser:

* **A user to ask.** ``send_msg_to_user`` is BrowserGym's channel to the person who
  set the task, and in this experiment that person is simulated from the details the
  task left open. The message is executed by BrowserGym, answered here, and the
  answer is put back on BrowserGym's own chat channel, so asking is an ordinary
  action and not a side door.
* **A turn cap and a wall clock**, both per task.
* **The reward**, its trajectory record, and the safety guards on committing actions
  and leaving the task's own site.

Nothing here resolves, repairs or retries an action. An action BrowserGym rejects is
reported to the policy as BrowserGym rejected it.
"""

from __future__ import annotations

import concurrent.futures as _cf
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from evaluation import simulated_user
from evaluation.action_guards import (
    WITHHELD_COMMIT,
    domain_of,
    withheld_kind,
    withheld_reason,
)
from evaluation.action_timeout import raise_action_timeout
from evaluation import web_transport
from evaluation.browser_identity import user_agent
from evaluation.env_blockers import PageProbe, classify_page, status_is_usable
from evaluation.action_trace import WITHHELD, WITHHELD_KIND
from evaluation.harness_report import HarnessReport
from evaluation.page_canonical import (
    PageState,
    changes,
    describe,
    history_line,
    page_state,
)
from evaluation.observation_mode import (
    FRAME_MAX_DIM,
    SOM_STATE,
    ObservationMode,
    downscale,
    render_frame,
    render_tree,
)
from evaluation.missing_parameters import (
    Parameter,
    open_items_from_row,
    parameters_from_row,
    row_left_nothing_open,
)
from evaluation.finish_action import FINISH_VERB
from evaluation.online_records import (
    END_COMMIT_WITHHELD,
    END_ENV_BLOCKED,
    END_ENV_UNREACHABLE,
    END_FINISHED,
    END_HARNESS_ERROR,
    END_MAX_STEPS,
    END_UNFINISHED,
    PreflightFailure,
)
from evaluation.rollout_deadline import (
    RolloutDeadlines,
    RolloutExpired,
    child_pids,
    kill_process_tree,
)
from evaluation.tool_contract import action_set
from evaluation.trajectory_store import episode_dir, new_episode_id
from training.cascp_reward import STEP_CAP_KEY, valid_step_cap

#: The verb at the head of a line of BrowserGym action code. Only the verb is read —
#: to tell an action aimed at the user from one aimed at the page, and to spend the
#: turn cap. The arguments are BrowserGym's to parse.
_VERB = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")

#: One addressable element in a rendered accessibility tree: ``[191] link 'Home'``.
_BID_LINE = re.compile(r"^\s*\[\w+\]\s", re.M)

#: Addressed to the user rather than to the page: a question, which is answered and the
#: episode continues, and "this is done" (:data:`FINISH_VERB`). ``report_infeasible``
#: never reaches the environment: the evaluation loop ends the rollout as ``stop``
#: before dispatching it.
TO_USER: frozenset[str] = frozenset({"send_msg_to_user", FINISH_VERB})

#: The one thread in this process that touches a browser.
#:
#: Not one per environment. BrowserGym keeps a single process-global Playwright
#: (``browsergym.core._get_global_playwright``), started on whichever thread first
#: asks for it, and Playwright's synchronous API binds every object to the thread
#: that made it. A second environment on a second thread therefore cannot start:
#: *cannot switch to a different thread (which happens to have exited)*. A worker
#: builds one environment per rollout, so per-environment threads would let the
#: first rollout run and fail every one after it — scoring zero, with nothing in
#: the metrics to distinguish that from a policy that cannot do the task.
#:
#: It is also why the thread is never ended: the global Playwright outlives any one
#: environment, and stopping its thread would strand it.
_BROWSER: _cf.ThreadPoolExecutor | None = None

#: How long one browser call may take before its worker is presumed wedged.
#:
#: There is one worker per process, so an unbounded wait is not a slow rollout -- it
#: is the whole process stopped. Six training arms sat at ``running: 1, finished: 7``
#: for hours on exactly this: one call that never returned, and seven rollouts queued
#: behind it that never started. Generous, because a real page load on a slow site is
#: allowed to take a while; the point is that *forever* is not a duration.
BROWSER_CALL_TIMEOUT_S = 180.0


class NoStartFrame(RuntimeError):
    """Raised when the start page was never photographed.

    Turn zero of an online task has no observation without it. A caller that carried
    on would show the policy either nothing or some other page, so this stops rather
    than substitutes. :attr:`training.bg_web_env.BrowserGymWebEnv.preflight_failure`
    reports the same condition without raising, and is what the rollout checks first.
    """

    def __init__(self, sample_id: str) -> None:
        super().__init__(f"no start frame was captured for {sample_id or '(unnamed)'}")
        #: The task the frame is missing for, so a caller logging this need not parse
        #: the message to name the row.
        self.sample_id = sample_id


class BrowserWedged(RuntimeError):
    """A browser call did not return, and its worker has been abandoned."""


class _Episode:
    """A token whose lifetime is exactly one :meth:`BrowserGymWebEnv.reset`.

    :class:`evaluation.rollout_deadline.RolloutDeadlines` keys on an object that
    lives as long as the episode. The environment is not that object: the trainer's
    tool builds, resets and tears one down per turn, while a pooled caller reuses
    one across rollouts. So a token is minted per reset and the previous one
    finished first, which makes the bound exactly one reset long however the caller
    brackets it.
    """


def browser_thread() -> _cf.ThreadPoolExecutor:
    """The process's browser thread, started on first use."""
    global _BROWSER
    if _BROWSER is None:
        _BROWSER = _cf.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bg-browser"
        )
    return _BROWSER


def retire_browser_thread() -> None:
    """Abandon the current browser worker; the next call starts a fresh one.

    The wedged thread is not stopped, because it cannot be: it is inside a blocking
    read in a C extension and Python has no way to interrupt it. It is dropped
    instead, and leaks along with the browser it was driving. One leaked thread per
    wedge is the price of the process continuing at all, and the alternative --
    waiting on it -- is what cost six arms a day of training.
    """
    global _BROWSER
    if _BROWSER is not None:
        _BROWSER.shutdown(wait=False, cancel_futures=True)
    _BROWSER = None
    forget_global_playwright()


def forget_global_playwright() -> None:
    """Drop BrowserGym's process-global Playwright along with its thread.

    ``browsergym.core._PLAYWRIGHT`` is created once per process, on whichever
    thread first asks for it, and Playwright's synchronous API backs it with an
    asyncio loop owned by *that* thread. Retiring the worker without clearing this
    leaves the global pointing at a dead loop, and every later reset dies inside
    BrowserGym's own setup at ``pw.selectors.set_test_id_attribute`` with
    ``RuntimeError: no running event loop`` -- before a browser is even requested.

    That is not hypothetical. Seven wedged calls retired the worker nineteen times
    across three training arms and produced **1,835** such failures, each recorded
    by the start-page probe as ``dns_or_connect_fail`` because the probe's
    catch-all cannot tell a crash in our setup from a site refusing us. The arms
    discarded 86-92% of every step for ten hours against sites that were up.

    The old Playwright and its driver process leak: they can only be stopped from
    the thread that made them, and that thread is gone. One leak per wedge is the
    price; the alternative is a worker that can never build an environment again.
    """
    try:
        import browsergym.core as bg  # noqa: PLC0415

        bg._set_global_playwright(None)
    except Exception:  # noqa: BLE001 - a harness without BrowserGym has none to drop
        pass


def quoted_argument(action: str) -> str:
    """The first quoted argument in ``action``, which is its bid or its url.

    ``quoted_argument("click('191')")`` is ``"191"``. Returns ``""`` when there is
    no quoted argument, which the callers read as "nothing to check".
    """
    found = re.search(r"""['"]([^'"]*)['"]""", action or "")
    return found.group(1) if found else ""


def verb_of(action: str) -> str:
    """The action name at the head of ``action``, or ``""`` if it has none.

    ``verb_of("click('191')")`` is ``"click"``. An action with no readable verb is
    not repaired here: it is passed to BrowserGym, which reports the parse failure
    to the policy against the page it was aimed at.
    """
    found = _VERB.match(action or "")
    return found.group(1) if found else ""


def _navigation_status(page: Any) -> int | None:
    """The HTTP status the browser got for the page it is on, or None.

    BrowserGym navigates to the start page inside ``reset()`` and does not keep the
    response, so the status has to be read back out of the browser afterwards. It is
    read from the page's own Navigation Timing entry rather than by re-requesting the
    URL: an HTTP client and Chromium do not always agree — one site answered 200 to a
    plain request while the browser was shown a CDN error page — and it is the
    browser's view that decides whether a policy can act.

    The status is what separates a bot wall from a slow page. Half the start pages in
    this store are blocked, so losing it would quietly turn a blocked rollout into a
    rollout the policy failed.
    """
    try:
        status = page.evaluate(
            "() => { const n = performance.getEntriesByType('navigation')[0];"
            " return n ? n.responseStatus : null; }"
        )
    except Exception:  # noqa: BLE001 - a page that will not run script is not fatal
        return None
    return int(status) if isinstance(status, (int, float)) and status else None


class MissingStepCap(ValueError):
    """Raised for a task row that carries no turn cap.

    The cap is a multiple of the turns the task's own reference trajectory took, so
    it is a property of the task and cannot be defaulted. A default would make
    running out of turns mean something different for every task.
    """


class BrowserGymWebEnv:
    """One BrowserGym rollout of a task: a live browser, the simulated user and the records.

    Args:
        answerer_endpoint / answerer_model: the serve that plays the user when the
            policy sends a message.
        deadlines: the run's per-rollout wall-clock bound, or None when untimed
            (tests and the offline evaluators).
        answer_fn: the simulated user's reply function; loaded from the answerer
            endpoint on the first question when None.

    The turn cap is per task and is read from the row in :meth:`reset`; see
    :class:`MissingStepCap`.
    """

    def __init__(
        self,
        *,
        answerer_endpoint: str | None = None,
        answerer_model: str | None = None,
        deadlines: RolloutDeadlines | None = None,
        answer_fn: Callable[..., str | None] | None = None,
    ) -> None:
        self.answerer_endpoint = answerer_endpoint
        self.answerer_model = answerer_model
        self.max_steps = 0
        self._deadlines = deadlines
        #: This rollout's token on the deadline clock; see :class:`_Episode`.
        self._episode: _Episode | None = None
        self._answer_fn = answer_fn
        self._env: Any = None
        self._obs: dict[str, Any] = {}
        self._sid = ""
        #: The key the reward log tags this rollout's events with, set by the
        #: trainer when it builds the environment for a rollout. Written into the
        #: trajectory record so a frame directory can be tied back to the reward
        #: that scored it. Left empty outside a trainer, where there is nothing to
        #: tie it to — every record written before this existed carries null, and
        #: those have to be joined by task instead.
        self.rollout_id = ""
        self._start_url = ""
        #: The HTTP status the start page answered with, used to tell a bot wall from
        #: a real page. None when the browser could not report one.
        self._start_status: int | None = None
        #: Why the browser did not come up, or "" if it did. A rollout that never
        #: started scores zero whatever the cause, so the cause has to be kept.
        self._boot_error = ""
        #: Route and per-rollout counters; reset() sets them again for each rollout.
        self.transport: str | None = None
        self.fallback_used = False
        self._direct_opened_blocked = False
        self.n_unanswerable_no_curated_intent = 0
        #: The page as the policy was last told about it. Under a state mode the
        #: next turn is told the difference from this, not the whole page again.
        self._page_seen: PageState | None = None
        self._episode_id: str | None = None
        self._ambiguity: str | None = None
        self._start_domain = ""
        self._severity: str | None = None
        #: The control this task's row names as its commit, when it names one.
        self._commit_name: str = ""
        self._commit_withheld = False
        self._cut_off = False
        #: How this rollout ended, or ``None`` while it is still running. Written by
        #: :meth:`_record_ending` and carried into the episode's own record.
        self._end_reason: str | None = None
        self._judged = False
        self._task_success = 0.0
        self._judge_reason = ""
        self._browser_pids: list[int] = []
        self._reset_done = False
        self._stopped = False
        #: Frames this rollout has already written a sidecar for.
        self._sidecars: set[str] = set()
        self._instruction = ""
        self._parameters: tuple[Parameter, ...] = ()
        self._open_items: list[dict[str, Any]] = []
        #: Set when the row left nothing open at all, which is not the same as a
        #: row whose items went missing. Only the first may be answered "no
        #: preference"; the second has to stay a counted guess.
        self._nothing_open = False
        self._actions: list[dict[str, Any]] = []
        self._shots: list[str] = []
        self._shot_dir: Path | None = None
        self._preflight: PreflightFailure | None = None
        self._n_steps = 0
        self._ask_count = 0
        self._report = HarnessReport()

    # ── the browser thread ───────────────────────────────────────────────────
    @staticmethod
    def _run(fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on the process's browser thread.

        BrowserGym drives Playwright's synchronous API, which refuses to run in a
        thread with a running asyncio loop — and the trainer's rollout loop is
        exactly that. A single-worker executor gives a loop-free thread and keeps
        every page object bound to the thread that made it. See :data:`_BROWSER`
        for why there is one of these per process rather than one per environment.

        Bounded, because the worker is shared: a call that never returns holds it
        against every other rollout in the process. On timeout the worker is retired
        and :class:`BrowserWedged` is raised, which the caller turns into the end of
        this rollout rather than the end of the training step.
        """
        try:
            return browser_thread().submit(fn).result(timeout=BROWSER_CALL_TIMEOUT_S)
        except _cf.TimeoutError as exc:
            retire_browser_thread()
            raise BrowserWedged(
                f"a browser call did not return within {BROWSER_CALL_TIMEOUT_S}s; "
                f"its worker has been abandoned and a fresh one will be started"
            ) from exc

    # ── lifecycle ────────────────────────────────────────────────────────────
    @staticmethod
    def _step_cap_of(row: dict[str, Any]) -> int:
        """This task's hard turn cap, from the row, or raise :class:`MissingStepCap`.

        Read through the predicate the *builder* uses, not a second copy of the
        rule written here. What counts as a capped row has to be one definition:
        two slightly different ones in two files is how a build passes a row the
        rollout then refuses.

        Never defaulted. The cap is a property of the task, so no fallback would
        mean anything — it would be a single global cap wearing a per-task name,
        binding tightly on a long task and not at all on a short one.
        """
        cap = valid_step_cap(row)
        if cap is None:
            raise MissingStepCap(
                f"row {row.get('task_id') or row.get('sample_id')!r} has no usable "
                f"{STEP_CAP_KEY!r} (got {row.get(STEP_CAP_KEY)!r}); every row of a "
                f"stamped build carries one and there is no default. Rebuild with "
                f"a {STEP_CAP_KEY!r} on every row."
            )
        return cap

    def reset(self, **row: Any) -> None:
        """Open a BrowserGym environment on the task's start page.

        Returns None. Whether the policy can be given a first turn here is reported
        by :attr:`preflight_failure`, which every caller must check before acting:
        an online task's opening observation is the live start page, so a page that
        did not open leaves nothing to show.
        """
        self.close()
        self._sid = str(row.get("task_id") or row.get("sample_id") or "")
        self._instruction = str(row.get("instruction") or row.get("task") or "")
        self._open_items = open_items_from_row(row)
        self._nothing_open = row_left_nothing_open(row)
        self._parameters = parameters_from_row(row)
        self._ambiguity = str(row.get("ambiguity") or "") or None
        self._cut_off = False
        self._end_reason = None
        self._commit_withheld = False
        self._judged = False
        self._task_success = 0.0
        self._judge_reason = ""
        self._actions, self._shots = [], []
        self._n_steps, self._ask_count = 0, 0
        #: Every (question, answer) this rollout has had; see :meth:`_qa_block`.
        self._qa: list[tuple[str, str]] = []
        self._stopped, self._reset_done = False, False
        self._sidecars = set()
        #: Which route this rollout reached the page by, and whether the
        #: fallback was needed. Recorded so a finished run can say how many
        #: of its rollouts the direct network could not reach at all.
        self.transport = web_transport.DIRECT
        self.fallback_used = False
        #: The direct route opened the start page but it was blocked there. Kept so
        #: that a page blocked directly and not opened by the fallback proxy is
        #: still set aside as blocked, as it is when no proxy is configured.
        self._direct_opened_blocked = False
        #: Questions on an Ambiguous row with no curated open item: the simulated
        #: user has no intended value and gives no answer.
        self.n_unanswerable_no_curated_intent = 0
        self._report = HarnessReport()
        # The episode's deadline starts here: this reset is what begins the rollout.
        self._begin_episode()
        start_url = str(row.get("start_url") or row.get("candidate_url") or "")
        self._start_url = start_url
        self._start_status = None
        self._boot_error = ""
        # Not cleared: the trainer sets it per rollout, before or after reset, and
        # clearing here would race with that and drop the key on half the records.
        self._start_domain = domain_of(start_url) if start_url else ""
        self._severity = (str(row.get("severity") or "").lower()) or None
        self._commit_name = str(row.get("commit_action") or "").strip()
        # The run-fatal check sits here, before a page load is spent on the
        # rollout: a row with no turn cap.
        self._page_seen = None
        self.max_steps = self._step_cap_of(row)

        persist = os.environ.get("CASCP_SHOT_DIR")
        self._episode_id = new_episode_id()
        if persist:
            # Per episode, not per task. Keying on the task id alone made every
            # rollout of a task write the same frames: 28,678 rollouts shared 352
            # directories, so the records looked complete while the frames they
            # named belonged to whichever rollout finished last.
            self._shot_dir = episode_dir(
                Path(persist), self._sid or "task", self._episode_id
            )
            self._shot_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._shot_dir = Path(tempfile.mkdtemp(prefix="bg_web_"))

        try:
            self._reset_done = self._attempt(start_url)
            if self._reset_done or self._boot_error:
                # Only when the direct attempt produced a blocker, and only once.
                # Most of what the arms discard is the site refusing the request --
                # 856 of 864 set-aside rollouts in one step never loaded at all --
                # and that is not the policy failing. A page blocked on both routes
                # is blocked; retrying further would cost browser time on exactly
                # the rollouts that will be set aside anyway.
                self._reset_done = self._fall_back_if_blocked(start_url)
            if self._reset_done:
                # The page as it was before the policy touched it. This is the
                # policy's turn-zero observation, the judge's first frame of the
                # trajectory, and the only screenshot of this task that is of this
                # task: the offline demonstration a row was built from photographed
                # a different page on a different day.
                self._snap("initial")
        except Exception as exc:  # noqa: BLE001 - an unreachable start page scores 0
            # Recorded, not only swallowed. A rollout that never started scores zero
            # either way, but "the site refused us" and "the browser would not launch"
            # need different work, and without this they are the same empty result.
            self._boot_error = f"{type(exc).__name__}: {exc}"
            print(f"[bg_web_env] reset failed: {self._boot_error}", flush=True)
            self._reset_done = False
        self._preflight = self._preflight_now()
        if self._preflight is not None:
            # Named on stdout and given an ending here, once, so a task nobody could
            # start is identifiable afterwards instead of being dropped on the floor.
            # This is the rollout's own record; what a run reports is built by its
            # caller from the result it returns.
            self._record_ending(self._preflight.end_reason)
            print(str(self._preflight), flush=True)
        return None

    def _preflight_now(self) -> PreflightFailure | None:
        """Why the policy cannot be given a first turn here, or None if it can.

        A reset that did not finish is reported as unreachable whether the browser
        refused to launch or the page came back unusable. The two are not told apart
        here because this cannot tell them apart: ``_boot`` returns False without
        setting an error for some failures, so inferring a bot wall from an empty
        error string would mislabel pages that never loaded at all.

        A page that did open is then checked against :func:`status_is_usable`, so a
        page the site refused is not handed to the policy. Three outcomes, three
        faults: a page that never loaded is the world's, a page we failed to
        photograph is ours, and a page we were refused is the world's again but a
        different thing -- it is set aside rather than counted as a task the policy
        failed.
        """
        if not self._reset_done and self._direct_opened_blocked:
            # Blocked on the direct route, and the fallback proxy did not open it:
            # blocked, set aside, exactly as without a proxy.
            return PreflightFailure(
                sample_id=self._sid or "(row with no id)",
                start_url=self._start_url,
                end_reason=END_ENV_BLOCKED,
                reason="the start page was blocked on the direct route and the "
                f"fallback proxy did not open it ({self._boot_error or 'no error text'})",
            )
        if not self._reset_done:
            return PreflightFailure(
                sample_id=self._sid or "(row with no id)",
                start_url=self._start_url,
                end_reason=END_ENV_UNREACHABLE,
                reason=self._boot_error or "the start page did not open",
            )
        if not self._shots:
            return PreflightFailure(
                sample_id=self._sid or "(row with no id)",
                start_url=self._start_url,
                # Ours, not the site's: the page opened and we failed to photograph it.
                end_reason=END_HARNESS_ERROR,
                reason="the start page opened but no first frame was captured",
            )
        # The page opened and was photographed. Whether the site actually served it
        # is a separate question, and until now nothing asked it here: classify_page
        # drove only the proxy retry, so a start page that was still a wall after the
        # retry was handed to the policy anyway. A rollout on a bot wall is scored as
        # though it were the task, which charges the world's refusal to the policy.
        #
        # Only the status rule is applied here, not the whole verdict. The other rules
        # read the rendered page, and the richest of them -- no addressable elements --
        # cannot tell "the page offers nothing" from "the tree could not be read",
        # which would turn an unreadable tree into a refused task. Those rules keep
        # their existing jobs: choosing the proxy retry, and partitioning a store.
        status = self.start_page_probe().status
        if not status_is_usable(status):
            return PreflightFailure(
                sample_id=self._sid or "(row with no id)",
                start_url=self._start_url,
                # The world's doing, not ours and not the policy's.
                end_reason=END_ENV_BLOCKED,
                reason=f"the start page was served HTTP {status}",
            )
        return None

    @property
    def preflight_failure(self) -> PreflightFailure | None:
        """Why this rollout never started, or None if it can start. Set by reset."""
        return self._preflight

    def start_frame(self) -> str:
        """The live start page, photographed after navigation and before any action.

        This is what turn zero shows the policy. Raises :class:`NoStartFrame` when
        there is none; callers check :attr:`preflight_failure` first, so reaching
        here without a frame is a broken caller rather than a bad task.
        """
        if not self._shots:
            raise NoStartFrame(self._sid)
        return self._shots[0]

    def close(self) -> None:
        """End the episode's browser, if one is open.

        The browser thread is deliberately **not** ended here. This environment is
        pooled and reset once per rollout — construct once, reset often — and
        Playwright binds every object it makes to the thread that made it. Shutting
        the executor down between rollouts means the next ``reset`` runs on a new
        thread and Playwright refuses it: *cannot switch to a different thread
        (which happens to have exited)*. The rollout then scores zero for a reason
        that has nothing to do with the policy, and it would do so for every rollout
        after the first.

        The thread is shared with every other environment in this process and is
        never ended here; see :data:`_BROWSER`.

        Every ordinary ending reaches here, so this is where the rollout comes off
        the clock. Without that the entry sits there until the deadline passes and
        the reaper counts a rollout that finished cleanly as expired.
        """
        self._end_episode()
        if self._env is not None:
            env, self._env = self._env, None
            try:
                self._run(env.close)
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass

    @property
    def _page(self) -> Any:
        """The live Playwright page BrowserGym is driving, or None.

        Exposed because the start-page blocker probe reads the rendered page
        directly. Nothing takes actions through it: actions go through BrowserGym so
        that bids stay the only way to address an element.
        """
        if self._env is None:
            return None
        unwrapped: Any = self._env.unwrapped
        return unwrapped.page

    # ── reaching the page ────────────────────────────────────────────────────
    def _boot(self, start_url: str, proxy: dict[str, str] | None = None) -> bool:
        """Open a BrowserGym environment on ``start_url``, optionally via a proxy."""
        # Before the browser exists, not after: chromium reads the proxy environment
        # when it launches. An inherited one that nothing answers makes every page
        # time out, which this harness can only record as the site refusing us.
        web_transport.clear_inherited_proxy_env()

        def run() -> bool:
            import gymnasium as gym  # noqa: PLC0415

            before = set(child_pids(os.getpid()))
            extra: dict[str, Any] = {}
            if proxy:
                # Playwright takes the proxy at launch; BrowserGym passes these
                # through to it untouched.
                extra["pw_chromium_kwargs"] = {"proxy": proxy}
            agent = user_agent()
            if agent:
                # Read on the browser thread, where every other browser call is made.
                extra["pw_context_kwargs"] = {"user_agent": agent}
            # Patched on the browser thread and before the first page exists, so no
            # action can be dispatched against an unpatched locator.
            raise_action_timeout()
            self._env = gym.make(
                "browsergym/openended",
                task_kwargs={"start_url": start_url},
                headless=True,
                # The user is simulated here, on our side of the chat channel, so the
                # environment must not block waiting for a real one to type.
                wait_for_user_message=False,
                action_mapping=action_set().to_python_code,
                **extra,
            )
            self._obs, _info = self._env.reset()
            # Read after the page is up, and never allowed to fail the reset: the
            # status sharpens the blocked/not-blocked call but a rollout can run
            # without it.
            self._start_status = _navigation_status(self._page)
            self._browser_pids = sorted(set(child_pids(os.getpid())) - before)
            return True

        return bool(self._run(run))

    def _attempt(self, start_url: str, proxy: dict[str, str] | None = None) -> bool:
        """One attempt at the start page. A failure is recorded, not raised."""
        try:
            self._boot_error = ""
            return self._boot(start_url, proxy)
        except Exception as exc:  # noqa: BLE001 - an unreachable page is a blocker
            self._boot_error = f"{type(exc).__name__}: {exc}"
            return False

    def _start_blocked(self) -> bool:
        """Whether the page we landed on is one no policy could act on."""
        if self._boot_error:
            return True
        try:
            return bool(classify_page(self.start_page_probe()).blocked)
        except Exception:  # noqa: BLE001 - an unreadable page is a blocked one
            return True

    def _fall_back_if_blocked(self, start_url: str) -> bool:
        """Retry through the configured proxy, once, if the direct attempt failed.

        Returns whether an environment is now open. A proxy that is not configured
        leaves the direct result exactly as it was, which is what every run before
        this one did.
        """
        if not self._start_blocked():
            return True
        proxy = web_transport.configured()
        if proxy is None:
            return bool(self._reset_done)
        self._first_verdict_direct = "blocked"
        self._direct_opened_blocked = bool(self._reset_done) and not self._boot_error
        self.close()
        self.transport = web_transport.FALLBACK
        self.fallback_used = True
        # A page that opens through the proxy is returned open even when it is still
        # blocked there: the caller's start-page check then sets it aside as
        # env_blocked, exactly as on the direct route. Returning "not open" here
        # filed such pages as env_unreachable (a failure kept in the denominator).
        return self._attempt(start_url, proxy.as_playwright())

    def start_page_probe(self) -> PageProbe:
        """What the browser actually sees at the start page.

        Only a browser verdict counts. A screenshot check is not enough — an error
        page has text and a logo too — so this reports the status, the title, the
        visible text and how many elements are addressable. A page with no
        addressable elements is one no policy can act on, whatever it renders.
        """

        def read() -> dict[str, Any]:
            page = self._page
            return {
                "final_url": page.url,
                "title": page.title(),
                "text": (page.inner_text("body") or "")[:4000],
                # Addressable elements, counted the way the policy sees them: one per
                # bid in the tree it is shown.
                "n_interactive": len(_BID_LINE.findall(self._tree_text())),
            }

        try:
            seen, error = self._run(read), None
        except Exception as exc:  # noqa: BLE001 - a page we cannot read is blocked
            seen, error = {}, f"{type(exc).__name__}: {exc}"
        return PageProbe(
            url=self._start_url,
            final_url=str(seen.get("final_url") or ""),
            status=self._start_status,
            title=str(seen.get("title") or ""),
            text=str(seen.get("text") or ""),
            n_interactive=int(seen.get("n_interactive") or 0),
            error=error,
        )

    # ── the one tool ─────────────────────────────────────────────────────────
    def browse(self, action: str) -> Any:
        """Take one BrowserGym action on the live page.

        Args:
            action: A single line of BrowserGym action code, such as ``click('191')``.
                Elements are named by the bid shown in square brackets in the
                accessibility tree.

        Returns:
            The resulting page as a screenshot and its accessibility tree, with a
            status line. An action BrowserGym rejected is reported as it rejected it.
        """
        if not self._reset_done or self._stopped:
            return "(environment not active)"
        # Before the turn cap, because a rollout stuck inside one turn never reaches
        # the cap: with a four-turn limit it is not looping, it is wedged.
        if self._expired():
            self._expire()
            return "(environment not active)"
        if self._n_steps >= self.max_steps:
            self._cut_off_now()
            return "(environment not active)"
        self._n_steps += 1
        verb = verb_of(action)
        try:
            if verb in TO_USER:
                return self._to_user(verb, action)
            return self._on_page(verb, action)
        except BrowserWedged:
            # The rollout is over either way; letting this reach the trainer would
            # take the whole training step with it.
            self._expire()
            return "(environment not active)"

    def _on_page(self, verb: str, action: str) -> Any:
        """Execute a page action and observe the result, unless it must be withheld.

        The accessible name of the element the action addresses is resolved from its
        bid once, used for the safety check, and written into the record so the run
        can be audited afterwards.
        """
        # Looked up once and recorded, not only checked. A record holding a bare bid
        # cannot be audited after the run: `click('191')` says nothing about what was
        # pressed, so a safety sweep reading these records would see no commit-like
        # action anywhere and report a clean run.
        name = self._name_for(action)
        url = quoted_argument(action) if verb == "goto" else ""
        kind = withheld_kind(
            verb,
            name=name,
            url=url,
            start_domain=self._start_domain,
            severity=self._severity,
            commit_name=self._commit_name,
        )
        if kind:
            # Only a commit sets this. It is read as "the agent reached the thing
            # that would have completed the task"; a blocked credential field or
            # an off-domain hop is neither, and setting it for those made the
            # field mean "some refusal happened somewhere".
            if kind == WITHHELD_COMMIT:
                self._commit_withheld = True
            refusal = withheld_reason(
                verb,
                name=name,
                url=url,
                start_domain=self._start_domain,
                severity=self._severity,
                commit_name=self._commit_name,
            )
            self._report.record_call("withheld")
            self._actions.append(
                {
                    "tool": verb,
                    "args": {"action": action, "name": name},
                    "result": refusal,
                    # The key the judge's action renderer reads. Without it the
                    # trace shows a click that simply had no effect, and the judge
                    # is told elsewhere not to credit an action that did not
                    # happen -- so a trajectory that reached the goal reads as one
                    # that failed at it.
                    WITHHELD: refusal,
                    # Only a commit is the agent having done the thing. A
                    # credential field or an off-domain hop did not happen, and
                    # the judge must not credit it.
                    WITHHELD_KIND: kind,
                }
            )
            if kind == WITHHELD_COMMIT:
                # The agent got to the thing that would have completed the task.
                # Pressing it is not ours to do, and carrying on is not the agent
                # having more to do: it has nowhere left to go, and every turn
                # after this one was spent re-pressing a button that never
                # responds, ending the episode at the step cap.
                #
                # `_stopped` as well as the ending, exactly as `report_task_complete`
                # does. Recording an ending without it left
                # the rollout running: the next turn overwrote `_end_reason`, and
                # the ending was counted once per press instead of once per
                # rollout.
                self._stopped = True
                self._record_ending(END_COMMIT_WITHHELD)
            return self._observe(refusal)

        def step() -> str:
            self._obs, _r, _term, _trunc, _info = self._env.step(action)
            return str(self._obs.get("last_action_error") or "")

        try:
            error = self._run(step)
        except BrowserWedged:
            # Not an action that failed: the process's only browser worker is gone.
            # Reporting it as a page error would hand the policy an observation and
            # invite another call onto a worker that no longer exists.
            raise
        except Exception as exc:  # noqa: BLE001 - a crashed step ends the rollout
            error = f"{type(exc).__name__}: {exc}"
        status = f"error: {error}" if error else "ok"
        self._report.record_call(None if not error else "rejected")
        self._actions.append(
            {
                "tool": verb or "unparsed",
                "args": {"action": action, "name": name},
                "result": status,
            }
        )
        self._snap(verb or "action")
        return self._observe(status)

    def _to_user(self, verb: str, action: str) -> Any:
        """Execute a message to the user, and answer it as that user.

        BrowserGym executes the action, which puts the message on its chat channel.
        The reply goes back on the same channel, so the transcript the environment
        holds is the conversation that actually happened — and so a later turn can
        read it there rather than from anything this harness keeps on the side.
        """

        def step() -> str:
            self._obs, _r, _term, _trunc, _info = self._env.step(action)
            return str(self._obs.get("last_action_error") or "")

        try:
            error = self._run(step)
        except BrowserWedged:
            raise  # see _on_page: a lost worker is not a failed action
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        if error:
            self._actions.append(
                {"tool": verb, "args": {"action": action}, "result": f"error: {error}"}
            )
            return self._observe(f"error: {error}")

        message = self._last_user_message()
        if verb == FINISH_VERB:
            # Not an ask, and not a stop. `_ask_count` is untouched -- the ask reward
            # reads the ASK entries in `_actions` and nothing is appended under that
            # name here -- and `_cut_off` stays False, which is what makes the rollout
            # eligible for a task-success score at all.
            #
            # The claim is recorded, not believed. `success_after_cutoff` no longer
            # zeroes this rollout, and the judge then decides whether the task was
            # actually done; a verb that granted its own success would be a reward
            # hack rather than a measurement.
            self._actions.append(
                {"tool": "FINISH", "args": {"summary": message}, "result": "finished"}
            )
            self._stopped = True
            self._record_ending(END_FINISHED)
            self._snap("finish")
            return "finished"

        self._ask_count += 1
        ask = {"tool": "ASK", "args": {"question": message}}
        self._actions.append(ask)
        self._snap("ask")
        reply = self._reply_to(message)
        # Kept beside the question. Without it the record shows the agent asking and
        # then acting on something the record does not contain, and the reply is the
        # served user simulator's own output -- the thing most worth inspecting,
        # because it can answer with a value the task never specified.
        ask["reply"] = reply

        def deliver() -> None:
            # The reply goes on BrowserGym's own chat channel, then the observation
            # is taken again. Without the second read the environment still holds
            # the observation from before the reply, so the transcript the next turn
            # sees would not contain the answer it is supposed to act on.
            self._env.unwrapped.chat.add_message("user", reply)
            self._obs = self._env.unwrapped._get_obs()

        self._qa.append((message, reply))
        self._run(deliver)
        return self._observe(f"the user replied: {reply}")

    def _reply_to(self, question: str) -> str:
        """The simulated user's reply to ``question``.

        Under a reward that grades asks, the reply is built from the parameters the
        ask judge credited, so what the policy is told and what it is paid for are
        the same verdict — consulted once, not twice.
        """
        if self._answer_fn is None:
            try:
                self._answer_fn = simulated_user.load_answerer(
                    self.answerer_endpoint, self.answerer_model
                )
            except Exception:  # noqa: BLE001 - reported, never fatal
                return "(no answer)"
        reply = simulated_user.answer(
            self._answer_fn,
            question,
            self._instruction,
            self._open_items,
            self._nothing_open,
        )
        if reply is None and not self._nothing_open and not any(
            str(item.get("answer") or "").strip() for item in self._open_items
            if isinstance(item, dict)
        ):
            # No curated intended value to answer from (an Ambiguous row with no
            # open item): the answerer refuses rather than invent one. Counted.
            self.n_unanswerable_no_curated_intent += 1
        return reply or "(no answer)"

    def _name_for(self, action: str) -> str:
        """The accessible name of the element ``action`` addresses, or ``""``.

        Looked up by bid, which is an exact key, not matched against anything the
        policy wrote. It is needed only so the safety guards can read what a control
        says before the environment presses it.
        """
        bid = quoted_argument(action)
        if not bid:
            return ""
        found = re.search(rf"\[{re.escape(bid)}\]\s+\S+\s+'([^']*)'", self._tree_text())
        return found.group(1) if found else ""

    def _tree_text(self) -> str:
        """The current accessibility tree as the policy is shown it."""
        from browsergym.utils.obs import flatten_axtree_to_str  # noqa: PLC0415

        try:
            return str(flatten_axtree_to_str(self._obs["axtree_object"]))
        except Exception:  # noqa: BLE001 - a page that will not snapshot still steps
            return ""

    def _last_user_message(self) -> str:
        """The text of the message the policy just sent, off BrowserGym's chat."""
        for entry in reversed(list(self._obs.get("chat_messages") or [])):
            if entry.get("role") in ("assistant", "agent"):
                return str(entry.get("message") or "")
        return ""

    def _cut_off_now(self) -> None:
        """End the rollout because it ran out of turns."""
        self._stopped = True
        self._cut_off = True
        self._record_ending(END_MAX_STEPS)

    def _record_ending(self, reason: str) -> None:
        """Note how this rollout ended, for the run's totals and for its own record.

        Both, because they answer different questions. The report's counter says how
        a run divided up; the field says how *this* rollout ended, and without it a
        single episode's ending cannot be read back off anything the training path
        writes -- which is how a harness with no way to record a completed task went
        unnoticed through several runs.
        """
        self._end_reason = reason
        self._report.record_ending(reason)

    # ── the wall-clock bound ─────────────────────────────────────────────────
    def _begin_episode(self) -> None:
        """Put this rollout on the clock, taking any previous one off it."""
        if self._deadlines is None:
            return
        if self._episode is not None:
            self._deadlines.finish(self._episode)
        self._episode = _Episode()
        self._deadlines.start(self._episode)
        self._deadlines.on_expire(self._episode, self._expire)

    def _end_episode(self) -> None:
        """Take this rollout off the clock because it ended of its own accord.

        Without this the entry stays on the clock until the deadline passes, and the
        reaper then counts a rollout that finished cleanly as expired -- a second
        ending recorded against it, and a hang in the log that never happened.
        """
        if self._deadlines is None or self._episode is None:
            return
        self._deadlines.finish(self._episode)
        self._episode = None

    def _expired(self) -> bool:
        """Whether this rollout has outrun its deadline, checked at a turn boundary."""
        if self._deadlines is None or self._episode is None:
            return False
        try:
            self._deadlines.check(self._episode)
        except RolloutExpired:
            return True
        return False

    def _expire(self) -> None:
        """End this rollout because it outran its wall-clock deadline.

        Recorded as ``unfinished``, which the fault table attributes to the harness.
        That is the right reading: a rollout that hits the turn cap would not stop
        and is the agent's doing, while one that hits the deadline was stuck inside
        something, which is ours.

        Called either from the turn boundary or from the reaper's thread. From the
        reaper the browser is ended by **signalling its driver process**, not by
        calling ``close()``: Playwright's synchronous API is bound to the thread that
        created it and raises when touched from any other, so a close from here would
        be swallowed and the pending call would stay blocked. Killing the process the
        browser thread is waiting to read from is what makes that read raise.
        """
        if self._cut_off:
            return
        self._stopped = True
        self._cut_off = True
        self._record_ending(END_UNFINISHED)
        for pid in self._browser_pids:
            kill_process_tree(pid)

    # ── what the policy sees ─────────────────────────────────────────────────
    def _qa_block(self) -> str:
        """Everything the user has told this rollout, restated.

        Carried on every turn after the first answer, not only on the turn it
        arrived. An answer is not part of the page, so the canonicaliser has
        nothing to diff it against and would otherwise state it once and never
        again. That is safe only while the conversation is perfectly append-only;
        the moment anything truncates it, the policy loses the detail it spent a
        turn obtaining and resumes guessing at precisely that detail, with nothing
        in the record to say so. It is a few dozen tokens.
        """
        lines = [f"  you asked: {q}\n  the user told you: {a}" for q, a in self._qa]
        return "What the user has told you so far:\n" + "\n".join(lines)

    def _observe(self, status: str) -> Any:
        """The page as a screenshot and a bid-labelled accessibility tree.

        The tree is what makes an action addressable: every line carries the bid an
        action names. Without it the screenshot shows the policy a page it has no
        way to point at.
        """
        mode = SOM_STATE
        body = self._what_changed(mode)
        text = f"status: {status}\nurl: {self._obs.get('url', '')}\n\n{body}"
        if self._qa:
            text += "\n\n" + self._qa_block()
        marked = render_frame(
            self._obs.get("screenshot"),
            self._obs.get("extra_element_properties") or {},
        )
        # Bounded here, and only here. The frame written to disk keeps the size
        # the browser rendered, because the task-success judge reads those and a
        # person reviewing a run should see what the page looked like; it is the
        # policy's copy that has to fit in a context the episode shares with nine
        # other turns.
        image = downscale(self._as_image(marked), FRAME_MAX_DIM)
        self._record_observation(text)
        if image is None:
            return text
        return [{"type": "image", "image": image}, {"type": "text", "text": text}]

    def ended_itself(self) -> bool:
        """Whether the environment ended this rollout on its own.

        The caller's loop decides when to stop from the verb it dispatched, which
        works for the two terminal verbs and not for a commit: that arrives as an
        ordinary ``click`` and is only recognised once the guard has resolved what
        it addresses. Without this the loop ran on past the commit to the step cap.
        """
        return self._stopped

    @property
    def end_reason(self) -> str | None:
        """The ending the environment recorded, if it recorded one."""
        return self._end_reason

    def _record_observation(self, text: str) -> None:
        """Write what the policy was shown beside the frame it was shown with.

        The frame on disk is the raw page: marking happens on the policy's copy
        only. So a reviewer reading the record sees a screenshot and an action
        naming a bid, with nothing connecting the two. The geometry that resolves
        it is in ``extra_element_properties``, which exists here and nowhere
        later, so every placed element is written out with its box, and a reader
        of the record can resolve a bid to a region of the frame.

        The text goes with it because it is the other half of the same turn, and
        a trajectory read without it is a sequence of actions against pictures,
        with the reason for each one missing.

        Never raises. This is a record of a turn, not the turn: the rollout has
        already paid for its browser time and a sidecar is not worth losing it
        over.
        """
        if not self._shots:
            return
        beside = Path(self._shots[-1]).with_suffix(".obs.json")
        # Written once per frame. `_observe` also runs on paths that take no new
        # frame first -- a withheld action, a failed message to the user, the
        # observation after a reply -- and keying on the latest frame let those
        # overwrite the sidecar of the frame before them, so the record claimed the
        # policy had been shown a refusal beside the opening screenshot.
        #
        # Tracked on this rollout rather than by asking the filesystem, so the
        # check costs no stat() and cannot be confused by anything already on
        # disk. (Each rollout writes under its own episode directory, so a stale
        # file from an earlier run is not actually reachable -- this is about
        # the three paths within one rollout that observe without snapping.)
        if str(beside) in self._sidecars:
            return
        try:
            # Inside the try, not above it. `list(prop["bbox"])` raises on a
            # truthy non-iterable box and `.items()` raises if the browser hands
            # back something that is not a mapping, and either would travel out of
            # `_observe` into the evaluator's blanket handler and cost the rollout
            # -- the exact price this method promises never to impose.
            elements = {
                str(bid): {"bbox": list(prop["bbox"])}
                for bid, prop in (
                    self._obs.get("extra_element_properties") or {}
                ).items()
                # An element the browser could not place is skipped rather than
                # written with a null box, for the reason render_frame leaves it
                # unmarked: a missing box costs one annotation, not the record.
                if isinstance(prop, dict) and prop.get("bbox")
            }
            beside.write_text(
                json.dumps(
                    {
                        "url": self._obs.get("url", ""),
                        "text": text,
                        "elements": elements,
                    }
                )
            )
            # Marked only after the write lands. Marking first made a transient
            # failure permanent for that frame.
            self._sidecars.add(str(beside))
        except Exception:  # noqa: BLE001 - a sidecar must not cost a turn
            return

    def _what_changed(self, mode: ObservationMode) -> str:
        """The page as a difference from the last one the policy was shown.

        Only what is genuinely new is stated. Everything else was said on an
        earlier turn and the conversation is append-only, so it is still in front
        of the model — repeating it costs the episode turns it could have spent
        acting. What *has* to be restated is anything whose handle changed or
        disappeared, because a handle the policy is still holding does not fail
        when the page moves: it resolves to whatever now carries that number.
        """
        current = page_state(
            str(self._obs.get("url") or ""),
            self._page_title(),
            self._tree_for_policy(mode),
        )
        block = describe(changes(self._page_seen, current))
        self._page_seen = current
        if self._actions:
            block = f"Your last action: {history_line(self._actions[-1])}\n\n{block}"
        return block

    def _page_title(self) -> str:
        """The page's title, or empty when the browser cannot report one."""
        try:
            page = self._page
            return str(self._run(page.title)) if page is not None else ""
        except Exception:  # noqa: BLE001 - a title is a nicety, not a turn
            return ""

    def _tree_for_policy(self, mode: ObservationMode) -> str:
        """The tree as this run's observation mode shows it.

        Separate from :meth:`_tree_text`, which stays the full tree: the blocked
        start-page check counts addressable elements, and that count has to mean
        the same thing in every arm of an ablation.
        """
        try:
            return render_tree(
                self._obs["axtree_object"],
                self._obs.get("extra_element_properties") or {},
                mode,
            )
        except Exception:  # noqa: BLE001 - a page that will not snapshot still steps
            return ""

    def _screenshot(self) -> Any:
        """The frame as the browser rendered it, unmarked.

        This is what is written to disk and what the task-success judge reads. It
        does not change with the observation mode: only the policy's view does, so
        a judgement stays comparable across the arms of an ablation and nobody is
        scored against a picture covered in annotations.
        """
        return self._as_image(self._obs.get("screenshot"))

    def _as_image(self, array: Any) -> Any:
        """A frame array as a PIL image, or None if there is not one."""
        if array is None:
            return None
        try:
            from PIL import Image  # noqa: PLC0415

            return Image.fromarray(array)
        except Exception:  # noqa: BLE001 - an unusable frame is not a failed step
            return None

    def _snap(self, tag: str) -> str | None:
        """Write the current frame beside the rollout's others, and return its path.

        The path is returned because the trainer shows the latest frame alongside an
        error turn, so that a failed action is still reported against a picture of
        the page it failed on.
        """
        if self._shot_dir is None:
            return None
        image = self._screenshot()
        if image is None:
            return None
        path = self._shot_dir / f"step_{len(self._shots):02d}_{tag}.png"
        try:
            image.save(path)
        except Exception:  # noqa: BLE001 - a frame that will not write is not fatal
            return None
        self._shots.append(str(path))
        return str(path)
