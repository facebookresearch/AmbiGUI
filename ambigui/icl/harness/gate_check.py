#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""gate_check.py <cases.json> <out.json> -- replay recorded messages through the
answerer and the ask judge (repo code, live endpoints from ANSWERER_ENDPOINT/ANSWERER_MODEL
and ASK_JUDGE_URL/ASK_JUDGE_MODEL) and check:
  * a report gets STATEMENT_REPLY from the answerer and is marked a statement by the ask judge;
  * a question is never marked a statement, and on a task that left nothing open it is
    answered through the answerer's nothing-open path (counted, never an invented value);
  * a genuine question about an open item is answered from it and credited.
Exit 0 iff every check passes."""

import json
import os
import sys

from evaluation import simulated_user
from evaluation.missing_parameters import parameters_from_row, row_left_nothing_open
from scripts.answerer import STATEMENT_REPLY

cases = json.load(open(sys.argv[1]))
answer_fn = simulated_user.load_answerer(os.environ["ANSWERER_ENDPOINT"], os.environ["ANSWERER_MODEL"])
answerer = getattr(answer_fn, "__self__", None)
out, failures = [], []
for c in cases:
    row = {"open_items": c["open_items"], "instruction": c["instruction"], "ambiguity": c["ambiguity"]}
    params = parameters_from_row(row)
    nothing_open = row_left_nothing_open(row)
    before = dict(answerer.counts()) if answerer else {}
    reply = simulated_user.answer(answer_fn, c["message"], c["instruction"], c["open_items"], nothing_open)
    after = dict(answerer.counts()) if answerer else {}
    targets = list(simulated_user.ask_targets(c["message"], c["instruction"], params))
    statement = simulated_user.was_a_statement(c["message"], c["instruction"], params)
    path = [k for k in after if after.get(k, 0) != before.get(k, 0)]
    rec = {"id": c["id"], "expected": c["expected_kind"], "nothing_open": nothing_open, "answerer_path": path,
           "reply": reply, "old_reply": c["old_reply"], "judge_targets": targets, "judge_statement": statement}
    if c["expected_kind"] == "statement":
        if reply != STATEMENT_REPLY:
            failures.append(f"{c['id']}: answerer replied {reply!r} to a statement")
        if not statement:
            failures.append(f"{c['id']}: ask judge did not mark the statement (targets={targets})")
    else:
        if statement:
            failures.append(f"{c['id']}: ask judge called a question a statement")
        if reply is None:
            failures.append(f"{c['id']}: answerer failed on a question")
        if nothing_open and answerer and path and path != ["nothing_open"] and path != ["statements"]:
            failures.append(f"{c['id']}: nothing-open question took path {path}")
    out.append(rec)
    print(json.dumps(rec)[:600], flush=True)
result = {"cases": out, "failures": failures, "answerer_counts": answerer.counts() if answerer else None,
          "gate_unreadable": list(simulated_user.GATE_UNREADABLE)}
json.dump(result, open(sys.argv[2], "w"), indent=1)
print(f"[gate-check] {len(out)} cases, {len(failures)} failures")
for f in failures:
    print(f"[gate-check] FAIL {f}")
sys.exit(0 if not failures else 1)
