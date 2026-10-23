#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Phase 1 of the online evaluation: run a served policy on live websites and score each task.

For each task the policy gets the instruction and a screenshot of the live start page, and
answers with one tool call per turn. The call is run in a real browser (BrowserGym); the
next screenshot goes back to the policy. A message to the user is answered by the simulated
user. When the rollout ends, the task judge (``--judge-kind lookup``, see
``evaluation/score_rollout.py``) decides success. Results are reported apart for Ambiguous
and Clear tasks. The questions the policy asked are written to ``ask_tasks.json`` for
phase 2 (``scripts/score_online_ask_quality.py``).

``harness/run_online.sh`` runs this file (through ``harness/run_eval.py``) with these flags:

    python scripts/eval_cascp_online.py --served-url http://host:8000/v1 --served-model <name> \
        --eval-rollouts <tasks.jsonl> \
        --judge-url <judge /v1> --judge-model <name> --judge-kind lookup \
        --describer-url <describer /v1> --describer-model <name> \
        --answerer-endpoint <answerer /v1> --answerer-model <name> \
        --max-tokens 8192 --shard 1/84 \
        --out <shard>/scores.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import re
from pathlib import Path
from typing import Any

from training.bg_web_env import verb_of
from evaluation.online_ask_quality import (
    AskedTask,
    counts as ask_counts,
    write_tasks as write_ask_tasks,
)
from evaluation.missing_parameters import parameters_from_row
from evaluation.browse_tool import (
    UnknownTool,
    action_code,
    parse_qwen3_coder_call,
)
from evaluation.env_blockers import start_page_verdict
from evaluation.finish_action import FINISH_VERB
from evaluation.harness_report import HarnessReport
from evaluation.image_budget import IMAGE_BUDGET, prune_images
from evaluation.observation_mode import FRAME_MAX_DIM, downscale
from evaluation.online_records import (
    END_ENV_BLOCKED,
    END_FINISHED,
    END_GENERATION_ERROR,
    END_HARNESS_ERROR,
    END_MAX_STEPS,
    END_NO_TOOL_CALL,
    END_REPEAT_BREAK,
    END_TRUNCATED,
    END_STOP,
    RolloutResult,
    scorable_for_success,
    task_key,
    task_record,
)
from evaluation.rollout_deadline import RolloutDeadlines
from evaluation.run_guards import (
    RunUnhealthy,
    ScoringHealth,
    assert_scoring_healthy,
)
from evaluation.score_rollout import JudgeMisconfigured
from evaluation.serve_preflight import assert_endpoint_live
from evaluation.tool_contract import tool_json_schema
from evaluation.trajectory_store import (
    build_trajectory,
    directory_for_shots,
    missing_trajectories,
    persist_rollout,
)
from evaluation.turn_metrics import format_turns, summarise_turns, turns_block
from training.cascp_common import CASCP_SYSTEM, task_rows_from_store

#: Tasks whose trajectory could not be persisted. Collected rather than raised so one
#: bookkeeping failure cannot kill a two-hour sweep, but surfaced in the summary: the
#: previous silent version discarded every non-empty record for weeks while the store
#: looked populated.
TRAJECTORY_WRITE_FAILURES: list[str] = []

#: Wall-clock bound on one rollout, in seconds (every paper run used 600). A rollout
#: with no bound can hang for hours.
EPISODE_DEADLINE_S = 600.0

# gemma-4 renders a tool call as `call:<name>{k:<|"|>v<|"|>, ...}` (the <|"|> quote tokens may be
# stripped in the API's content). Parse from content when vLLM has no server-side tool parser.
_CALL_RE = re.compile(r"call:(\w+)\s*\{(.*)\}", re.DOTALL)
_KV_RE = re.compile(
    r"(\w+)\s*:\s*(?:<\|\"\|>)?(.*?)(?:<\|\"\|>)?(?:,\s*(?=\w+\s*:)|$)", re.DOTALL
)

def parse_tool_from_text(text: str) -> tuple[str | None, dict[str, Any]]:
    """Extract (name, args) from a Gemma-rendered tool-call string `call:name{...}`; (None, {}) if none."""
    if not text:
        return None, {}
    m = _CALL_RE.search(text)
    if not m:
        return None, {}
    name, body = m.group(1), m.group(2)
    args = {k: v.strip() for k, v in _KV_RE.findall(body)}
    return name, args


def parse_tool_json(text: str) -> tuple[str | None, dict[str, Any]]:
    """Extract (name, args) from an OpenAI/Hermes-style JSON tool call, e.g. Qwen/GLM/InternVL/Llama:
    `<tool_call>{"name":"click","arguments":{"target":"NEWS"}}</tool_call>` or a bare
    `{"name":...,"arguments":...}` (possibly preceded by role/junk text). Scans for the first JSON
    object carrying a "name" key; arg values are stringified (the env tools take string args).
    """
    if not text:
        return None, {}
    t = text.replace("<tool_call>", " ").replace("</tool_call>", " ")
    dec = json.JSONDecoder()
    i = 0
    while True:
        c = t.find("{", i)
        if c < 0:
            return None, {}
        try:
            obj, end = dec.raw_decode(t[c:])
        except json.JSONDecodeError:
            i = c + 1
            continue
        if isinstance(obj, dict) and "name" in obj:
            raw = obj.get("arguments") or obj.get("parameters") or {}
            args = (
                {
                    k: (v if isinstance(v, str) else json.dumps(v))
                    for k, v in raw.items()
                }
                if isinstance(raw, dict)
                else {}
            )
            return str(obj["name"]), args
        i = c + end


def shard_of(rows: list[Any], spec: str) -> list[Any]:
    """The ``i/n`` slice of ``rows``, taken round-robin.

    Round-robin and not contiguous: the task set is ordered, and a contiguous slice
    would hand one shard every task from one site. A shard that drew all of a
    blocked domain would report a success rate for that domain rather than for the
    policy.

    Raises:
        ValueError: on a spec that is not ``i/n`` with ``1 <= i <= n``. A silently
            ignored shard means every shard runs every task and the results are
            counted several times.
    """
    try:
        index, total = (int(part) for part in spec.split("/", 1))
    except ValueError:
        raise ValueError(f"--shard {spec!r} is not of the form i/n") from None
    if total < 1 or not 1 <= index <= total:
        raise ValueError(f"--shard {spec!r} needs 1 <= i <= n and n >= 1")
    return rows[index - 1 :: total]


def parse_tool(text: str) -> tuple[str | None, dict[str, Any]]:
    """Read a tool call from reply text: a Qwen3-Coder XML call first, then a Gemma-style
    ``call:name{...}``, then a JSON object ``{"name": ..., "arguments": ...}``. ``(None, {})``
    when none of the three is found."""
    for candidate in (
        parse_qwen3_coder_call,
        parse_tool_from_text,
        parse_tool_json,
    ):
        name, args = candidate(text)
        if name:
            return name, args
    return None, {}


def _extract_call(msg) -> tuple[str | None, dict[str, Any]]:  # noqa: ANN001
    """Read the call from the content text. The request sends ``tool_choice="none"``, so
    the serve never returns parsed ``tool_calls``."""
    return parse_tool(msg.content or "")


def _data_url(pil) -> str:  # noqa: ANN001
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _tool_response_content(obs: Any) -> list[dict]:
    """Environment tool return -> OpenAI multimodal tool-message content (image_url + text)."""
    if isinstance(obs, str):
        return [{"type": "text", "text": obs}]
    out = []
    for part in obs:
        if part.get("type") == "image":
            out.append(
                {"type": "image_url", "image_url": {"url": _data_url(part["image"])}}
            )
        else:
            out.append({"type": "text", "text": part.get("text", "")})
    return out


#: The v2 verdict reason when an Ambiguous task carries no curated open item.
NO_CURATED_INTENT = "no curated intent"


def _no_curated_intent(row, curated_row) -> bool:  # noqa: ANN001
    """An Ambiguous task whose curated row lists no open item: v2 has nothing to check.

    The same rule as ``score_rollout_v2.curated_intent``: an empty list is
    known-empty intent only on a row labelled Clear. A missing, mismatched or
    malformed curated row is a source error, reported by the v2 judge itself.
    """
    if not isinstance(curated_row, dict):
        return False
    same_task = (str(curated_row.get("task_id") or curated_row.get("sample_id") or "")
                 == str(row.get("task_id") or row.get("sample_id") or "")
                 and curated_row.get("instruction") == row.get("instruction"))
    items = curated_row.get("open_items")
    return same_task and isinstance(items, list) and not items and curated_row.get("ambiguity") != "Clear"


def run_task(
    env,
    client,
    model,
    tools,
    row,
    judge_url,
    judge_model,
    judge_kind,
    max_tokens=1024,
    describer_url=None,
    describer_model=None,
    *,
    judge_version="judge_v1",
    curated_row=None,
    judge_config=None,
    judge_journal_root=None,
    judge_transport=None,
):  # noqa: ANN001
    """Drive one eval task through the tool-call loop; return a :class:`RolloutResult`.

    ``n_dispatched`` counts tool calls that actually reached an env method, so a caller
    can tell an agent that chose to do nothing from a harness that could not dispatch
    anything. ``n_turns`` counts policy generations the harness consumed, which is the
    larger number: a reply with no readable tool call, or one naming a tool the harness
    cannot place, cost a turn and dispatched nothing.

    ``end_reason`` says why the loop stopped. ``max_steps`` and ``repeat_break`` mean the
    harness cut the rollout off rather than the policy ending it, which right-censors the
    turn count -- see :mod:`evaluation.turn_metrics`.

    Judge v2 changes only the judge input. The rollout, the rows that are set aside
    and every ending that is not judged (unreachable or blocked start page, harness
    error, no screenshot) are exactly as under v1: no judge call and
    ``success=False``. Only a rollout that v1 would put to the judge goes to the v2
    judge, and only there can the verdict be ``None`` (unresolved)."""
    from datetime import datetime, timezone  # noqa: PLC0415

    from evaluation.score_rollout import score_rollout  # noqa: PLC0415

    n_dispatched = [0]
    judge_error: str | None = None
    judge_rationale: str | None = None
    judge_journal_path: str | None = None
    if judge_version != "judge_v1":
        # Run settings, not task data: a wrong version, config or journal root is
        # wrong for every task, so it stops the run (as JudgeMisconfigured does for
        # v1). The curated row is checked only when the judge is called.
        from evaluation.judge_v2_adapter import validate_config
        from evaluation.score_rollout_v2 import VERSION
        if judge_version != VERSION or judge_journal_root is None:
            raise JudgeMisconfigured(f"unknown judge version {judge_version!r} or no journal root")
        try:
            validate_config(judge_config, kind=judge_kind, judge_url=judge_url,
                judge_model=judge_model, describer_url=describer_url, describer_model=describer_model)
        except Exception as exc:
            raise JudgeMisconfigured(f"judge v2 config: {type(exc).__name__}: {exc}") from exc

    # When this rollout ran, in UTC. Judge v2 reads the date as the task's time
    # evidence, so "today" and "tomorrow" refer to the day the policy acted.
    started = datetime.now(timezone.utc).replace(microsecond=0)
    time_evidence = {"timestamp": started.isoformat().replace("+00:00", "Z"),
                     "date": started.date().isoformat(), "timezone": "UTC"}
    env.reset(**row)
    transport = getattr(env, "transport", None)
    preflight = env.preflight_failure
    if preflight is not None:
        # A full result, not a short tuple: the caller unpacks every field, and a
        # short return here was once read as a harness crash on exactly the rows the
        # clean/blocked cut exists to separate. The environment has already named
        # this task and its reason on stdout, so nothing is printed again here.
        # Same outcome under v1 and v2: not judged, success False.
        return RolloutResult(n_turns=0, end_reason=preflight.end_reason,
            success=False, judge_version=judge_version,
            transport=transport, task_time_evidence=time_evidence)

    # The page loaded; can a policy act on it? A bot wall, a CloudFront denial or an
    # interstitial looks like a page and is not one. Left undetected, the rollout burns
    # its whole step budget on an error page, asks questions that score irrelevant
    # because none can name a task parameter, and is recorded as the policy failing.
    verdict = start_page_verdict(env)
    if verdict.blocked:
        print(f"  [blocked] {verdict.kind.value}: {verdict.evidence[:90]}")
        return RolloutResult(n_turns=0, end_reason=END_ENV_BLOCKED,
            success=False, judge_version=judge_version,
            transport=transport, task_time_evidence=time_evidence)

    # Turn-1 observation: the live start page, photographed by the environment after
    # it navigated. Never a stored frame -- the offline demonstration a row was built
    # from photographed a different page on a different day.
    #
    # Bounded by the same rule every later turn is. ``start_frame`` is the file on
    # disk, and frames on disk are deliberately written at the resolution the browser
    # drew, for the judge and for anyone reviewing a run. The policy's copy is
    # smaller. Sending the file straight through crossed the two: measured on a live
    # task the policy saw 1280x720 on turn one and 768x432 on every turn after it,
    # against a trainer that only ever shows it 768x432.
    from PIL import Image  # noqa: PLC0415

    env._observe("initial observation")
    seed_url = _data_url(downscale(Image.open(env.start_frame()), FRAME_MAX_DIM))
    messages = [
        {"role": "system", "content": CASCP_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Instruction: {row['instruction']}"},
                {
                    "type": "image_url",
                    "image_url": {"url": seed_url},
                },
            ],
        },
    ]
    turns: list[dict[str, Any]] = []
    last_sig, repeat = None, 0
    # `max_steps` is the standing reason: running the range to exhaustion means the
    # harness cut the rollout off, and every other exit overwrites it on its way out.
    #
    # The bound is the env's, which it read off this row: a multiple of the length of
    # this task's own recorded solution. The evaluator used to carry its own flag
    # instead, so the number of turns a task got depended on which of four unrelated
    # limits the run happened to be under rather than on the task.
    n_turns, end_reason = 0, END_MAX_STEPS
    end_detail: str | None = None
    #: Set when the loop stops because of us rather than because of the policy. The
    #: rollout's record -- its asks, its actions, its frames -- lives on the
    #: environment, and letting the exception escape threw all of it away: the
    #: caller's blanket handler built a fresh default result, so a policy that had
    #: asked was recorded as ``n_ask: 0, asks: []``. Caught here instead, so the
    #: return below reports what actually happened before the failure.
    harness_failed = False
    try:
        for _ in range(env.max_steps):
            try:
                # The serve refuses a prompt carrying more images than it was started
                # with, and the loop above appends one screenshot per turn. Unpruned,
                # every episode ended on the turn that crossed the budget -- 47 of 126
                # in one baseline, recorded only as a generation error.
                resp = client.chat.completions.create(
                    model=model,
                    messages=prune_images(messages, IMAGE_BUDGET),
                    tools=tools,
                    # gemma-4 has no vLLM tool-call parser; "none" keeps the tool schema rendered in
                    # the prompt (via the chat template) but disables server-side parsing — we text-parse
                    # the `call:...` from content, matching how the policy was trained.
                    tool_choice="none",
                    max_tokens=max_tokens,
                    temperature=0.0,
                    seed=0,
                )
            except (
                Exception
            ) as e:  # noqa: BLE001 - a generation failure ends the rollout
                # Recorded on the row, not only printed under a debug flag. This was the
                # commonest ending of both baseline evaluations and the row said only
                # "generation_error"; what it actually was -- an HTTP 400, mostly the
                # context limit -- had to be recovered from the serve's own log.
                end_detail = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"  [dbg] generate raised {end_detail}")
                end_reason = END_GENERATION_ERROR
                break
            # Counted before the exit tests below: the generation that produced a stop, an
            # unreadable reply or a repeat still happened and still cost a turn.
            n_turns += 1
            msg = resp.choices[0].message
            turns.append(
                {
                    "turn": n_turns,
                    "content": msg.content or "",
                    "finish_reason": getattr(resp.choices[0], "finish_reason", None),
                }
            )
            name, args = _extract_call(msg)
            print(
                f"  [dbg] tool_calls={bool(getattr(msg, 'tool_calls', None))} "
                f"name={name} args={args} content={(msg.content or '')[:300]!r}"
            )
            if not name:
                # `length` means the serve stopped it at the token limit, so
                # there was nothing complete to parse. That is a budget we set,
                # and it needs different work from a policy that wrote prose.
                stopped_at = getattr(resp.choices[0], "finish_reason", None)
                end_reason = (
                    END_TRUNCATED if stopped_at == "length" else END_NO_TOOL_CALL
                )
                end_detail = f"finish_reason={stopped_at}"
                break
            # loop-breaker: a warm-start policy often repeats one action forever without emitting stop;
            # a stuck agent isn't completing the task, so end the rollout (saves 12 slow browser steps).
            sig = (name, tuple(sorted(args.items())))
            repeat = repeat + 1 if sig == last_sig else 0
            last_sig = sig
            if repeat >= 2:
                print(f"  [dbg] loop-break on repeated {name}")
                end_reason = END_REPEAT_BREAK
                break
            messages.append({"role": "assistant", "content": msg.content or ""})

            # Resolve the tool call to the BrowserGym action it names.
            #
            # An unrecognised call ends the STEP, not the rollout: the task's caller catches
            # every exception and scores the whole task success=False, n_ask=0, so letting
            # one malformed call out of here would discard a rollout's real asks and real
            # progress and record it as a policy that did nothing.
            try:
                code = action_code(name, args)
            except UnknownTool as e:
                messages.append({"role": "user", "content": f"Tool error: {e}"})
                continue
            verb = verb_of(code)
            if verb == "report_infeasible":
                end_reason = END_STOP
                break
            # The terminal success verb.
            if verb == FINISH_VERB:
                # Dispatched before the loop breaks, unlike `report_infeasible` above:
                # that way the environment records the FINISH action, its ending and its
                # final frame exactly as it does on the training path, and the judge here
                # scores the same trajectory it would score there. A verb handled before
                # dispatch on one path and after it on the other is how these two
                # harnesses diverge.
                n_dispatched[0] += 1
                env.browse(code)
                end_reason = END_FINISHED
                break
            n_dispatched[0] += 1
            obs = env.browse(code)
            # An ending the environment decided, not the verb. A commit is an
            # ordinary `click` until the guard sees what it addresses, so this loop
            # cannot know the rollout is over from the action text -- and without
            # reading it back, `commit_withheld` never reached a scores row and the
            # rollout carried on to the step cap.
            # Any ending the environment decided, not just the commit. Narrowing
            # this to the commit looked conservative and was not: the episode
            # deadline also stops the rollout -- it kills the browser process
            # tree -- and `unfinished` appears nowhere else in this file, so the
            # loop carried on generating against a dead environment for the rest
            # of its turns and the row was finally labelled `max_steps`.
            #
            # This does relabel those rows, from `max_steps` to `unfinished`.
            # That is the point: they are a bookkeeping timeout, not a policy
            # that used its budget, and the harness report attributes the two to
            # different causes.
            if env.ended_itself():
                end_reason = env.end_reason or end_reason
                break
            # The observation goes back as a plain user turn: gemma-4's chat template only
            # emits image placeholders for user turns, not tool turns (a tool message 500s
            # with "Failed to apply prompt replacement").
            messages.append({"role": "user", "content": _tool_response_content(obs)})
    except Exception as exc:  # noqa: BLE001 - the rollout's record must survive it
        end_reason = END_HARNESS_ERROR
        end_detail = f"{type(exc).__name__}: {str(exc)[:300]}"
        harness_failed = True
    success: bool | None = False
    if harness_failed:
        # Nothing is put to the judge. A trajectory that stops because of us is not
        # evidence about the policy, and a verdict on it would read as one.
        # Same outcome under v1 and v2.
        print(f"  [dbg] harness error, not judged: {end_detail}")
    elif judge_version != "judge_v1" and env._shots:
        if _no_curated_intent(row, curated_row):
            # An Ambiguous task with no curated open item has no intended values to
            # check, so the v2 verdict is unresolved. Not a judge error, no call.
            success, judge_rationale = None, NO_CURATED_INTENT
            print(f"  [dbg] judge v2 not called: {NO_CURATED_INTENT}")
        else:
            from evaluation.judge_v2_adapter import score_future_task
            verdict_v2 = score_future_task(row=row, shots=list(env._shots), actions=list(env._actions),
                curated_row=curated_row, config=judge_config, journal_root=Path(judge_journal_root),
                kind=judge_kind, judge_url=judge_url, judge_model=judge_model,
                describer_url=describer_url, describer_model=describer_model,
                transport=judge_transport,
                original_task_time_evidence=time_evidence,
                population_metadata={"flags": row.get("flags"), "end_reason": end_reason,
                                     "view_eligibility": row.get("view_eligibility"),
                                     "policy_configured_output_budget": max_tokens})
            success, judge_rationale, judge_error = verdict_v2.success, verdict_v2.why, verdict_v2.source_error
            judge_journal_path = verdict_v2.journal_path
            print(f"  [dbg] judge v2 success={success} error={judge_error}")
    elif env._shots:
        try:
            success, judge_rationale = score_rollout(
                judge_kind,
                judge_url,
                judge_model,
                row["instruction"],
                env._shots,
                env._actions,
                describer_url=describer_url,
                describer_model=describer_model,
            )
            print(
                f"  [dbg] {judge_kind} success={success} shots={len(env._shots)} "
                f"actions={len(env._actions)}"
            )
        except JudgeMisconfigured:
            # A judge that cannot score anything -- an unknown kind, a serve the kind
            # needs and does not have -- is wrong for every task in the run, not for
            # this one. Recording it per task turns it into 126 ordinary failures and
            # a complete-looking success=0.0000 table. Let it end the run instead.
            raise
        except Exception as e:  # noqa: BLE001
            success = False
            # Recorded, not just logged: a swallowed judge exception is indistinguishable
            # from a failed task, and seven checkpoints once reported 0.0000 because of it.
            judge_error = f"{type(e).__name__}: {str(e)[:120]}"
            print(
                f"  [dbg] {judge_kind} judge RAISED {type(e).__name__}: {str(e)[:300]}"
            )
    else:
        print("  [dbg] no shots captured -> success=False (env produced no trajectory)")
    return RolloutResult(
        success=success,
        judge_rationale=judge_rationale,
        judge_version=judge_version,
        judge_journal_path=judge_journal_path,
        n_ask=env._ask_count,
        n_dispatched=n_dispatched[0],
        judge_error=judge_error,
        n_turns=n_turns,
        end_reason=end_reason,
        end_detail=end_detail,
        shots=tuple(env._shots),
        actions=tuple(env._actions),
        turns=tuple(turns),
        transport=transport,
        task_time_evidence=time_evidence,
        n_unanswerable_no_curated_intent=int(getattr(env, "n_unanswerable_no_curated_intent", 0) or 0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    from evaluation.judge_v2_adapter import add_activation_arguments, activation_from_args
    add_activation_arguments(ap)
    ap.add_argument("--served-url", required=True)
    ap.add_argument("--served-model", required=True)
    ap.add_argument(
        "--eval-rollouts", type=Path, required=True, help="the benchmark task file (JSONL)"
    )
    ap.add_argument("--judge-url", required=True)
    ap.add_argument(
        "--judge-model",
        required=True,
        help="served name of the task-success judge. There is no default: a name "
        "that does not match the serve's --served-model-name fails every call.",
    )
    ap.add_argument(
        "--judge-kind",
        required=True,
        help="which judge scores the rollout; 'lookup' is the only one.",
    )
    ap.add_argument(
        "--describer-url",
        default=None,
        help="vision serve the 'lookup' judge asks about the screens",
    )
    ap.add_argument("--describer-model", default=None)
    ap.add_argument("--answerer-endpoint", required=True)
    ap.add_argument("--answerer-model", required=True)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="new tokens per policy turn",
    )
    ap.add_argument(
        "--shard",
        default="1/1",
        help="run one slice of the task set, as i/n. Parallelism inside one process "
        "is limited: BrowserGym keeps a single process-global Playwright bound to "
        "the thread that started it, so every browser operation in a process "
        "serialises however many worker threads there are. Sharding across "
        "processes is what actually scales.",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    judge_config, curated_rows = activation_from_args(args)

    from openai import OpenAI  # noqa: PLC0415

    from training.bg_web_env import BrowserGymWebEnv  # noqa: PLC0415

    # --eval-rollouts is the benchmark task file: JSON Lines, one task per line. Each
    # task's ambiguity label is its own row's.
    labels = {
        str(json.loads(x).get("sample_id", json.loads(x).get("task_id"))).split("__r")[0]: str(
            json.loads(x).get("ambiguity")
        )
        for x in args.eval_rollouts.read_text().splitlines()
        if x.strip()
    }
    # One loader for both judge versions, so the rollouts, the set-aside rows and the
    # success views are the same under v1 and v2. Judge v2 reads the curated intent
    # from --curated-sidecar, never from these rows.
    tasks = task_rows_from_store(args.eval_rollouts)
    tasks = shard_of(tasks, args.shard)
    tools = [tool_json_schema()]
    print("[harness] turn cap: per task, from each row's max_steps.")
    from evaluation.endpoint_auth import api_key_for  # noqa: PLC0415

    client = OpenAI(base_url=args.served_url, api_key=api_key_for(args.served_url))

    # One live browser environment, created for the first task and reused for the rest.
    # Tasks run one at a time; parallelism comes from running shards (--shard) as
    # separate processes.
    _env_cache: list[Any] = []

    def _env() -> Any:
        if not _env_cache:
            _env_cache.append(BrowserGymWebEnv(
                answerer_endpoint=args.answerer_endpoint,
                answerer_model=args.answerer_model,
                deadlines=RolloutDeadlines(deadline_s=EPISODE_DEADLINE_S),
            ))
        return _env_cache[0]

    # /v1/models is not a liveness check: a rotted multi-day serve answers it and then
    # returns nothing to a real request. Both of these send an actual completion.
    if args.judge_version == "judge_v1":
        assert_endpoint_live(endpoint=args.judge_url, model=args.judge_model)
    # V2 uses the actual versioned task route, never a generic credential-fallback probe.
    assert_endpoint_live(endpoint=args.answerer_endpoint, model=args.answerer_model)
    print("[harness] judge and answerer answered a live completion", flush=True)

    health = ScoringHealth()

    harness_rows: list[tuple[str, list[Any]]] = []
    #: One entry per task, for the ask-quality pass below. Collected during the run
    #: because it needs the row's open parameters, which the per-task record does not
    #: carry; scored afterwards because an evaluation has no reward to pay and so no
    #: reason to hold a rollout open waiting on a judge.
    asked_tasks: list[AskedTask] = []
    #: Questions the simulated user could not answer because the Ambiguous row has no
    #: curated open item (task id -> count).
    unanswerable: dict[str, int] = {}

    def _run_one(i: int, row: dict) -> dict:
        base = task_key(row)
        try:
            result = run_task(
                _env(),
                client,
                args.served_model,
                tools,
                row,
                args.judge_url,
                args.judge_model,
                args.judge_kind,
                max_tokens=args.max_tokens,
                describer_url=args.describer_url,
                describer_model=args.describer_model,
                judge_version=args.judge_version,
                judge_config=judge_config,
                curated_row=(curated_rows.get(row.get("task_id", row.get("sample_id"))) if curated_rows is not None else None),
                judge_journal_root=args.judge_journal_root,
            )
        except JudgeMisconfigured:
            # Not one task's failure: the run named a judge that cannot score
            # anything, so every remaining task would fail the same way and the
            # eval would finish and report a number. See run_task.
            raise
        except Exception as e:  # noqa: BLE001 - never let one task kill the eval
            # `harness_error`, not `generation_error`: this is our code breaking, and
            # filing it as a refused completion is the same mislabelling the
            # env_unreachable branch was added to stop.
            # Same outcome under v1 and v2: not judged, success False.
            result = RolloutResult(
                end_reason=END_HARNESS_ERROR,
                success=False,
                judge_version=args.judge_version,
                end_detail=f"{type(e).__name__}: {str(e)[:300]}",
            )
            print(f"[{i}] {base} ERROR {type(e).__name__}: {str(e)[:80]}")
        rec = task_record(task_id=base, ambiguity=labels.get(base), result=result)
        # Carried on the row so the summary can set it aside without re-reading
        # the benchmark. Absent on a set that does not carry the flag, which
        # reads as False and changes nothing.
        if args.judge_version != "judge_v1":
            rec["exact_task_id"] = row.get("task_id", row.get("sample_id"))
        rec["stops_before_commitment"] = not scorable_for_success(row)
        # The ending and the ACTIONS, not the summary row. The report's counters were
        # defined, tested and never filled because the run fed it per-task summary
        # rows, which carry no actions -- so a rollout that dispatched five calls
        # reported `tool_calls: 0`. Appending is atomic, and the report is folded once
        # after the run so concurrent tasks cannot race a Counter.
        harness_rows.append((result.end_reason, list(result.actions)))
        if result.n_unanswerable_no_curated_intent:
            unanswerable[base] = result.n_unanswerable_no_curated_intent
        asked_tasks.append(
            AskedTask(
                task_id=base,
                instruction=str(row["instruction"]),
                # Empty on a Clear row even when it carries open items: a
                # person judged the task clear, so no question was needed and
                # an ask on it is over-asking, not a hit against those items.
                # Twelve rows of the curated set are labelled exactly that way.
                parameters=(
                    parameters_from_row(row)
                    if str(row.get("ambiguity", "")) == "Ambiguous"
                    else ()
                ),
                questions=tuple(result.asks),
            )
        )
        if result.success is not None:
            # An unresolved v2 verdict is neither a success nor a judge exception;
            # it is counted beside the rates instead (n_unresolved).
            health.record(success=result.success, judge_error=result.judge_error)

        # Persist everything the judge consumed, so a judge failure can be recovered
        # without running the rollout again.
        shot_dir = os.environ.get("CASCP_SHOT_DIR")
        if shot_dir:
            failure = persist_rollout(
                directory_for_shots(list(result.shots), fallback=Path(shot_dir) / base),
                build_trajectory(
                    task_id=base,
                    instruction=row["instruction"],
                    shot_paths=list(result.shots),
                    actions=list(result.actions),
                    asks=list(result.asks),
                    success=result.success,
                    judge_error=result.judge_error,
                    judge_rationale=result.judge_rationale,
                    judge_version=result.judge_version,
                    meta={
                        "served_model": args.served_model,
                        "exact_task_id": row.get("task_id", row.get("sample_id")),
                        "judge_journal_path": result.judge_journal_path,
                        # When the rollout ran (UTC); judge v2 reads it as time evidence.
                        "original_task_time_evidence": result.task_time_evidence,
                        "transport": result.transport,
                        "n_ask": result.n_ask,
                        # The same key the training environment writes, so one reader
                        # can ask either store why an episode ended. The eval used to
                        # keep this only in its own results file, which meant a dump
                        # from this path could not be told from one that ended at the
                        # step cap.
                        "end_reason": result.end_reason,
                        # The policy's own words, per turn. The actions say what
                        # the environment received; only this says what the model
                        # wrote, and whether the serve stopped it mid-sentence.
                        "turns": [dict(t) for t in result.turns],
                    },
                ),
            )
            if failure:
                TRAJECTORY_WRITE_FAILURES.append(failure)
                print(f"[{i+1}] {base} trajectory not written: {failure}")
        print(
            f"[{i+1}/{len(tasks)}] {base} amb={labels.get(base)} "
            f"success={result.success} asks={result.n_ask} "
            f"acts={result.n_dispatched} turns={result.n_turns} "
            f"end={result.end_reason}"
        )
        return rec

    results = [_run_one(i, row) for i, row in enumerate(tasks)]

    def _rate(xs):  # noqa: ANN001
        """``rate: None`` when there is nothing to divide by.

        It used to be ``0.0``, which is what made every all-act cell publish
        ``success_if_asked: {"n": 0, "rate": 0.0}`` -- read as "asking never works" when
        the policy simply never asked.
        """
        xs = [x for x in xs if x is not None]
        n = len(xs)
        return {"n": n, "rate": (sum(xs) / n if n else None), "k": sum(xs)}

    v2 = args.judge_version != "judge_v1"

    def _success_rate(rows):  # noqa: ANN001
        """Success over resolved verdicts. Under v2, the unresolved (null) verdicts
        are left out of the rate and counted next to it."""
        out = _rate([r["success"] for r in rows])
        if v2:
            out["unresolved"] = sum(r["success"] is None for r in rows)
        return out

    def strat(rows):  # noqa: ANN001
        """Task accuracy, ask/act split, and accuracy CONDITIONED on asked vs acted."""
        asked = [r for r in rows if r["n_ask"] > 0]
        acted = [r for r in rows if r["n_ask"] == 0]
        return {
            "n": len(rows),
            "success": _success_rate(rows),
            "ask_rate": _rate([r["n_ask"] > 0 for r in rows]),
            "act_rate": _rate([r["n_ask"] == 0 for r in rows]),
            "success_if_asked": _success_rate(asked),
            "success_if_acted": _success_rate(acted),
        }

    # Blocked rows are a different population, not noise to average in. A page we were
    # never allowed to use is not a task the policy failed, and a score averaged over a
    # third of such rows is not a measurement of the policy. They are set aside here and
    # counted in `set_aside` below, not silently dropped.
    blocked_rows = [r for r in results if r.get("end_reason") == END_ENV_BLOCKED]
    # Rows whose recording completed a commitment the agent must not reach. The
    # curated benchmark says so on the row, and says plainly that task success is
    # not measurable on them: the end state is deliberately out of reach. Scored,
    # they would report the benchmark's design as the policy's failure.
    unreachable = [
        r
        for r in results
        if r.get("end_reason") != END_ENV_BLOCKED and r.get("stops_before_commitment")
    ]
    scored = [
        r
        for r in results
        if r.get("end_reason") != END_ENV_BLOCKED
        and not r.get("stops_before_commitment")
    ]
    amb_rows = [r for r in scored if r["ambiguity"] == "Ambiguous"]
    clr_rows = [r for r in scored if r["ambiguity"] == "Clear"]
    cuts = {"overall": scored, "ambiguous": amb_rows, "clear": clr_rows}
    summary = {
        "judge_version": args.judge_version,
        "n_unresolved": sum(r["success"] is None for r in results),
        # Simulated-user replies on Ambiguous rows with no curated open item (the
        # policy read "(no answer)"); see docs/online.md.
        "unanswerable_no_curated_intent": sum(unanswerable.values()),
        "unanswerable_no_curated_intent_task_ids": sorted(unanswerable),
        "n": len(scored),
        "n_attempted": len(results),
        "set_aside": {
            "env_blocked": len(blocked_rows),
            "stops_before_commitment": len(unreachable),
            "stops_before_commitment_task_ids": [r.get("task_id") for r in unreachable][
                :30
            ],
            "task_ids": [r.get("task_id") for r in blocked_rows][:20],
            "why": (
                "the start page could not be acted on (bot wall, CloudFront denial, "
                "interstitial, or nothing interactive). Excluded from every rate above."
            ),
        },
        # Kept for the readers of scores.json: the plain system prompt is the only one,
        # and no trace-mode component metric is collected.
        "sys_mode": "plain",
        "overall": strat(scored),
        "ambiguous": strat(amb_rows),
        "clear": strat(clr_rows),
        "components": {
            "n_traced": 0,
            "applicable": False,
            "reason": (
                "not collected: sys_mode='plain'. Component metrics need "
                "--sys-mode trace and a policy trained to emit a TRACE block. "
                "Not measured, as distinct from measured zero."
            ),
        },
        # No single ceiling to report: every task is capped by its own trajectory, so a
        # run-wide number here would be one task's cap printed against all of them.
        "turns": turns_block(cuts, max_steps=None),
        "results": results,  # every row, including the set-aside ones
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Written first so the evidence survives, then judged. A run whose scores came from
    # a broken judge still exits non-zero, so nothing downstream treats it as a result.
    summary["scoring_health"] = {
        "n_scored": health.n_scored,
        "n_success": health.n_success,
        "judge_errors": health.judge_errors,
        "judge_error_rate": round(health.judge_error_rate, 4),
    }
    # A trajectory store that silently lost every usable record still reported a clean
    # run, so the count travels with the scores rather than only to stdout.
    # Whose fault each rollout's ending was, and how much the harness had to repair.
    # Both travel with the score: a run where a fifth of the rollouts ended on us is
    # not a run that measured the policy.
    harness_report = HarnessReport()
    for end_reason, actions in harness_rows:
        harness_report.record_ending(end_reason)
        harness_report.record_actions(actions)
    # Ambiguous rows with no curated open items. The user simulator does not invent a
    # value for them: a question on such a row gets no answer. Listed here.
    from training.cascp_common import OPEN_ITEM_MISSING_ON_AMBIGUOUS  # noqa: PLC0415

    harness_report.open_items_missing_on_ambiguous = list(
        OPEN_ITEM_MISSING_ON_AMBIGUOUS
    )
    summary["harness_report"] = harness_report.as_dict()
    # Handed to a separate job rather than judged here. The rollouts hold a policy
    # serve and a browser; the judging holds neither, so making the ask judge a
    # dependency of this job would buy nothing and can only cost it. The counts are
    # written now because they are measured now; the quality is written by
    # scripts/score_online_ask_quality.py reading the file below.
    ask_tasks_path = args.out.with_name("ask_tasks.json")
    write_ask_tasks(ask_tasks_path, asked_tasks)
    summary["ask_quality"] = {
        "applicable": False,
        "reason": (
            f"scored by a separate pass: "
            f"python scripts/score_online_ask_quality.py --tasks {ask_tasks_path} "
            f"--out {args.out.with_name('ask_quality.json')}"
        ),
        "tasks_file": str(ask_tasks_path),
        **ask_counts(asked_tasks),
    }
    # A write that never raised but also never happened was invisible: the failure
    # count was 0 and the records simply were not there. `missing_trajectories` already
    # knew how to find them and only the re-judge tool ever called it.
    shot_root = os.environ.get("CASCP_SHOT_DIR")
    no_record = missing_trajectories(Path(shot_root)) if shot_root else []
    summary["trajectory_health"] = {
        "n_write_failures": len(TRAJECTORY_WRITE_FAILURES),
        "examples": TRAJECTORY_WRITE_FAILURES[:5],
        "n_with_frames_but_no_record": len(no_record),
        "no_record_examples": no_record[:5],
        "n_rollouts": len(results),
    }
    args.out.write_text(json.dumps(summary, indent=2))
    try:
        assert_scoring_healthy(health)
    except RunUnhealthy as exc:
        print(f"[harness] FATAL: {exc}", file=sys.stderr)
        raise SystemExit(4) from exc

    def fmt(s):  # noqa: ANN001
        def r(d):
            # An undefined rate prints as n/a rather than 0%, so a stratum nobody asked
            # on is not read as a stratum where asking never worked.
            extra = f" ({d['unresolved']} unresolved)" if d.get("unresolved") else ""
            if d["rate"] is None:
                return f"{d['k']}/{d['n']}=n/a{extra}"
            return f"{d['k']}/{d['n']}={d['rate']:.0%}{extra}"

        return (
            f"acc {r(s['success'])} | ask {r(s['ask_rate'])} act {r(s['act_rate'])} | "
            f"acc|asked {r(s['success_if_asked'])} acc|acted {r(s['success_if_acted'])}"
        )

    print("=== ONLINE EVAL ===")
    print("[harness]")
    print(harness_report.summary())
    if blocked_rows:
        print(
            f"  set aside: {len(blocked_rows)} of {len(results)} rows started on a page "
            f"a policy could not act on; every rate below excludes them"
        )
    print(f"  Ambiguous: {fmt(summary['ambiguous'])}")
    print(f"  Clear:     {fmt(summary['clear'])}")
    print("  [turns per task]")
    for cut, rows in cuts.items():
        print("    " + format_turns(cut, summarise_turns(rows, max_steps=None)))
    print(f"  [CASCP components] sys_mode={summary['sys_mode']}")
    print(f"    NOT MEASURED: {summary['components']['reason']}")

if __name__ == "__main__":
    main()
