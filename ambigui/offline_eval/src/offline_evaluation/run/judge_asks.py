# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 2 of the ask protocol: judge the questions a finished run recorded.

The inference loop used to judge each question the moment it was asked. That put a second
model inside a loop that already had the policy in it, and tied the run's survival to the
judge's: a judge serve dying at hour six threw away six hours of policy calls that cannot
be reproduced, because a served model decoding greedily is not bit-for-bit repeatable.

So judging moved out. The run writes ``asks.jsonl`` with the question, the user's reply
and the serve that gave it, and nothing else. This stage reads that file, asks the judge
what each question earned, and writes the verdicts back onto the same rows:

``is_request``
    whether the judge read the text as a request to the user for information.
``resolved``
    the open items it credited, each spendable once per task.
``relevant``
    whether it credited anything at all.

A judge outage here costs a re-judge and no GPU hours: the verdicts are computed in
memory and ``asks.jsonl`` is rewritten only once every row has one, through a sibling
temporary file that is renamed onto the target, so a crash can never leave a
half-length file in place of the only record of the questions. Re-running is safe --
both files this stage writes are rewritten from the recorded questions, never accumulated
onto. ``inference.jsonl`` is rewritten with its previous judge rows dropped rather than
appended to; a second judging of one question would otherwise leave two rows for it with
nothing saying which verdict is live. Rewriting in place rather than writing a separate
judge log is the smaller change: the run's log already holds one row per model call, and
this stage's rows are the only ones under the judge's role, so they can be told from the
run's own by that field alone.

Unknown task/step joins are rejected before any judge call. Known tasks with no open
items are valid over-asks and receive explicit no-credit status without a judge call.

The judge's **registry entry** is ``--judge-registry-key`` (required): the catalogue
entry under ``configs/models`` whose ``max_tokens`` the judge is posed with. See
:func:`judge_config`.

Usage::

    python -m offline_evaluation.run.judge_asks RUN_DIR \\
        --judge-endpoint http://host:8000/v1 --judge-model arbiter

A hosted judge's key is read from ``AMBIGUI_JUDGE_API_KEY``; unset, no key is sent.

Then :mod:`offline_evaluation.run.score_ask_protocol` on the same directory; it refuses a
run whose asks have not been through here.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from evaluation.missing_parameters import Parameter, parameters_from_details
from evaluation.serve_preflight import assert_endpoint_live

from offline_evaluation.ask_judge import AskVerdict, parse_verdict, request_prompt
from offline_evaluation.client import ModelConfig, ServeClient, get_model, models
from offline_evaluation.http import JUDGE_API_KEY_ENV, api_key_from_env
from offline_evaluation.inference_log import JUDGE, InferenceRow, require_complete_run
from offline_evaluation.metrics import ask_step_items
from offline_evaluation.jsonl import read_jsonl
from offline_evaluation.serve_pool import (
    ServePool,
    ServeUnavailable,
    describe,
    serve_targets,
)
from offline_evaluation.substrate import StepItem, read_substrate

#: Tries of the judge serve, and the wait between them. A serve briefly saturated usually
#: answers on the next try; one that is gone never will.
JUDGE_ATTEMPTS = 3
JUDGE_RETRY_PAUSE = 5.0
#: Seconds allowed for one judge call (every run set used 300).
JUDGE_TIMEOUT_S = 300.0

#: Reads one text against a task's open parameters and says what it asks about.
Matcher = Callable[[str, str, Sequence[Parameter]], AskVerdict]


def make_matcher(pool: ServePool) -> Matcher:
    """A matcher that asks the judge whether a text is a request, and what it asks for.

    The prompt and the reply reader are
    :func:`offline_evaluation.ask_judge.request_prompt` and
    :func:`offline_evaluation.ask_judge.parse_verdict`. The prompt first asks whether the
    text is a request at all, so a report that states the values earns nothing.

    A serve failure is not a verdict. :class:`offline_evaluation.serve_pool.ServePool`
    retries it and finally raises, so "the judge was down" can
    never be recorded as "the question asked about nothing". Only a reply the judge
    actually produced reaches :func:`parse_verdict`.

    Reached only with at least one open parameter: :func:`judge_rows` skips a row before
    it when the task has nothing left open.
    """

    def match(
        instruction: str, text: str, parameters: Sequence[Parameter]
    ) -> AskVerdict:
        prompt = request_prompt(
            instruction=instruction, text=text, parameters=parameters
        )
        given = pool.ask_recorded(prompt)
        return replace(
            parse_verdict(given.text, parameters),
            prompt=prompt,
            raw=given.text,
            serve=given.serve,
            retries=given.failures,
        )

    return match


def judge_rows(
    rows: Sequence[Mapping[str, Any]],
    items: Sequence[StepItem],
    *,
    matcher: Matcher,
) -> tuple[list[dict[str, Any]], list[InferenceRow]]:
    """Every ask row with its verdict on it, and the judge calls that produced them.

    The rows come back in the order they were given, so the file keeps the order the run
    wrote. They are *judged* in ``(step_index, ask_index)`` order within each task, which
    is the order they were asked: an open item is spendable once per task, and reading a
    later step first would spend it before the step that earned it.

    A task whose first step has no open items is not judged at all. ``is_request`` stays
    ``None`` there, which says no judge ran -- not the same as a judged refusal, which is
    ``False``. Known empty tasks receive ``judge_status="no_open_items"``; unknown task
    or step joins are rejected. Already spent items are not offered again.

    A cut-off row is judged and credited nothing, and spends nothing for the rows after
    it. The turn ended in a loop; crediting it would let a policy empty the open-item set
    by asking its way into a cut-off and still be paid for it.
    """
    per_task: dict[str, list[Parameter]] = {}
    for item in items:
        per_task.setdefault(
            item.task_key, list(parameters_from_details(item.open_items))
        )
    instructions = ask_step_items(items, rows)
    spent: dict[str, set[str]] = {key: set() for key in per_task}

    judged: dict[int, dict[str, Any]] = {}
    calls: list[InferenceRow] = []
    order = sorted(
        range(len(rows)),
        key=lambda i: (
            str(rows[i]["task_key"]),
            int(rows[i]["step_index"]),
            int(rows[i]["ask_index"]),
        ),
    )
    for index in order:
        row = rows[index]
        key = str(row["task_key"])
        offered = [p for p in per_task.get(key, []) if p.name not in spent.get(key, ())]
        if not offered:
            judged[index] = {
                **row,
                "is_request": None,
                "resolved": [],
                "relevant": False,
                "judge_status": (
                    "no_open_items" if not per_task[key] else "already_resolved"
                ),
            }
            continue
        verdict = matcher(
            instructions[(key, int(row["step_index"]))].instruction,
            str(row["question"]),
            offered,
        )
        calls.append(_judge_row(row, verdict))
        credited = () if row.get("cut_off") else tuple(verdict.targets)
        spent[key].update(credited)
        judged[index] = {
            **row,
            "is_request": verdict.is_request,
            "resolved": list(credited),
            "relevant": bool(credited),
            "judge_status": "judged",
        }
    return [judged[i] for i in range(len(rows))], calls


def _judge_row(row: Mapping[str, Any], verdict: AskVerdict) -> InferenceRow:
    """The log row for one judge call, against the ask it judged."""
    return InferenceRow(
        task_key=str(row["task_key"]),
        step_index=int(row["step_index"]),
        attempt=int(row["ask_index"]),
        role=JUDGE,
        prompt=verdict.prompt,
        text=verdict.raw,
        serve=verdict.serve,
        retries=verdict.retries,
    )


def judge_config(registry_key: str) -> ModelConfig:
    """The catalogue entry to pose the judge with: the one named by ``registry_key``.

    The entry sets the judge's token budget, so it is always named explicitly; a judge
    posed with the policy's entry would get the policy's budget, and a reasoning judge cut
    off mid-``<think>`` makes every call unusable.

    ``vision=False`` always: the judge reads text. Posing it the screens would change its
    verdict without changing what it was asked.
    """
    return replace(get_model(registry_key), vision=False)

def write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Replace ``path`` with ``rows``, one JSON object per line, in one step.

    Written to a sibling ``.tmp`` and renamed onto the target rather than opened with
    ``"w"``, which truncates before the first row is written. Both files this stage
    replaces are the only record of what the run asked and what it was told, and a
    crash partway through an in-place rewrite leaves a half-length file that reads to
    the scorer as a complete one. ``os.replace`` is atomic within a filesystem, so the
    target is either the old file or the whole new one.

    The temporary file is removed when the write fails, so a failed stage leaves the
    directory as it found it.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def rewrite_inference(path: Path, calls: Sequence[InferenceRow]) -> None:
    """The run's inference log with ``calls`` in place of any judge rows already in it.

    Rewritten, not appended to. Judging twice used to leave two rows for one question
    with nothing on either saying which verdict was live, in a file whose whole purpose
    is that a report can be checked against the run behind it.

    Only the judge's own role is dropped. The policy and simulated-user rows are the
    run's, cost GPU hours, and are not reproducible, so they are carried through
    untouched and in order. A run whose log was never written -- a hand-assembled
    directory -- gets one holding this stage's calls alone.

    The replacement goes through :func:`write_rows`, so the run's own rows cannot be
    lost to a crash partway through writing this stage's.
    """
    kept = (
        [row for row in read_jsonl(path) if row.get("role") != JUDGE]
        if path.exists()
        else []
    )
    write_rows(path, [*kept, *(asdict(call) for call in calls)])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dir", type=Path, help="a finished run directory; asks.jsonl is rewritten"
    )
    parser.add_argument(
        "--judge-endpoint",
        default="",
        help="serve that decides which open items a question asks about. Required",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="the judge serve's --served-model-name",
    )
    parser.add_argument(
        "--judge-registry-key",
        required=True,
        help="the judge's own entry under configs/models, which is what sets its token "
        "budget",
    )
    parser.add_argument("--seed", type=int, default=0, help="the judge's decoding seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The judge is not optional, and it is never the policy: a policy that marks its own
    # questions is a self-agreeing measurement.
    if not args.judge_endpoint:
        raise SystemExit("FATAL: --judge-endpoint is required to judge a run's asks")
    if not args.judge_model:
        raise SystemExit("FATAL: --judge-model is required with --judge-endpoint")
    # A typo here would otherwise fall through to the policy's entry and judge the run on
    # the policy's token budget, which is the failure this flag exists to prevent.
    if args.judge_registry_key not in models():
        raise SystemExit(
            f"FATAL: --judge-registry-key {args.judge_registry_key!r} is not under "
            f"configs/models; known entries are {sorted(models())}"
        )
    run_dir = Path(args.run_dir)
    try:
        require_complete_run(run_dir)
        rows = list(read_jsonl(run_dir / "asks.jsonl"))
        items = read_substrate(run_dir / "substrate.jsonl")
        ask_step_items(items, rows)
    except ValueError as broken:
        raise SystemExit(f"FATAL: {broken}") from broken
    judges = serve_targets(args.judge_endpoint, args.judge_model)
    config = judge_config(args.judge_registry_key)
    print(
        f"[judge] posed as registry entry {config.key}, max_tokens={config.max_tokens}",
        flush=True,
    )
    # A serve that is down, moved, or published under another name looks identical to a
    # model that judges badly.
    # A hosted judge needs a key; it is read from the environment, never from the
    # command line, and a local serve needs none.
    judge_key = api_key_from_env(JUDGE_API_KEY_ENV)
    for host, name in judges:
        assert_endpoint_live(host, name, api_key=judge_key)

    pool = ServePool(
        tuple(
            ServeClient(
                endpoint=host,
                served_name=name,
                config=config,
                timeout=JUDGE_TIMEOUT_S,
                seed=args.seed,
                api_key=judge_key,
            )
            for host, name in judges
        ),
        attempts=JUDGE_ATTEMPTS,
        pause=JUDGE_RETRY_PAUSE,
    )
    print(f"[judge] {len(rows)} asks, judge: {describe(judges)}", flush=True)
    try:
        judged, calls = judge_rows(rows, items, matcher=make_matcher(pool))
    except ServeUnavailable as dead:
        # Nothing is written. A half-judged file reads as a finished one to the scorer,
        # and the rows it never reached would score as questions that earned nothing.
        raise SystemExit(f"FATAL: {dead}") from dead

    # Zero judge calls is valid when all joined tasks genuinely have no open items.
    write_rows(run_dir / "asks.jsonl", judged)
    rewrite_inference(run_dir / "inference.jsonl", calls)

    relevant = [row for row in judged if row["relevant"]]
    requests = [row for row in judged if row["is_request"]]
    print(f"[judge] {len(calls)} judge calls -> {run_dir / 'asks.jsonl'}")
    print(f"[judge]   read as a request: {len(requests)}/{len(judged)}")
    print(f"[judge]   credited an open item: {len(relevant)}/{len(judged)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - main() itself is covered
    raise SystemExit(main())
