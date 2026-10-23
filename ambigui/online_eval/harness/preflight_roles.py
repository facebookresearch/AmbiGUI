#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Prove every dependency role answers a real request at its configured endpoint, using
the repo's own client code (not a hand-rolled call), before any browser opens.

Roles: answerer (scripts/answerer.Answerer), task judge + describer
(evaluation.score_rollout 'lookup'), ask judge (evaluation.simulated_user.ask_targets),
and each policy model (one chat call with the real system prompt + an image).

Exit non-zero if any dependency role fails; policies are reported, never fatal
(a model that is not servable is reported as such and skipped by the launcher).
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from PIL import Image, ImageDraw  # noqa: E402


def frame(path: str) -> str:
    img = Image.new("RGB", (1280, 720), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 40), "Mortgage calculator", fill="black")
    d.text((40, 100), "Home price: $400,000   Down payment: 20%   Loan type: 30-year fixed", fill="black")
    d.text((40, 160), "Estimated monthly payment: $2,415", fill="black")
    img.save(path)
    return path


def main(argv=None) -> int:
    import argparse
    from pathlib import Path
    from evaluation.judge_v2_adapter import add_activation_arguments
    from evaluation.score_rollout_v2 import VERSION
    ap = argparse.ArgumentParser()
    add_activation_arguments(ap)
    ap.add_argument("--judge-trajectory", type=Path)
    ap.add_argument("--judge-kind", default="lookup")
    ap.add_argument("--judge-url")
    ap.add_argument("--judge-model")
    ap.add_argument("--describer-url")
    ap.add_argument("--describer-model")
    args = ap.parse_args(argv)
    if args.judge_version == VERSION:
        # Future explicit judge-only gate: original source, native encoder and budgets.
        # No policy, answerer or ask-judge call is made by this branch.
        from evaluation.judge_v2_preflight import preflight_203
        try:
            code, report = preflight_203(args)
        except Exception as exc:
            code, report = 3, {"scope": "task_judge_only_preflight", "version": VERSION,
                "judge": {"verdict": None, "why": f"source_error: {type(exc).__name__}: {exc}"}}
        print(json.dumps(report, indent=1))
        return code
    out: dict = {"t": time.time(), "judge_version": "judge_v1"}
    ok = True
    judge_url = os.environ["JUDGE_URL"]
    describer_url = os.environ.get("DESCRIBER_URL") or judge_url
    # --- answerer
    from scripts.answerer import Answerer

    a = Answerer(endpoint=os.environ["ANSWERER_ENDPOINT"], model=os.environ["ANSWERER_MODEL"])
    items = [{"name": "home_price", "question": "What home price should I use?", "answer": "$400,000"}]
    instr = "Calculate what a 30 year fixed rate mortgage monthly payment would be"
    r1 = a.answer("What home price should I use for the calculation?", instr, open_items=items)
    r2 = a.answer("Should I click the blue button at the top of the page?", instr, open_items=items)
    out["answerer"] = {"on_item": r1, "off_item": r2, "counts": a.counts()}
    if not r1:
        ok = False
    # --- task judge + describer
    from evaluation.score_rollout import score_rollout

    shot = frame(os.path.join(os.environ.get("TMPDIR", "/tmp"), "preflight_frame.png"))
    try:
        verdict = score_rollout(
            "lookup",
            judge_url,
            os.environ["JUDGE_MODEL"],
            instr,
            [shot],
            [{"action": "fill('12', '400000')", "status": "ok"}, {"action": "report_task_complete('Computed payment')", "status": "ok"}],
            describer_url=describer_url,
            describer_model=os.environ["DESCRIBER_MODEL"],
        )
        out["judge"] = {"verdict": verdict[0], "why": verdict[1], "version": "judge_v1"}
    except Exception as e:  # noqa: BLE001
        out["judge"] = {"error": f"{type(e).__name__}: {e}"[:400]}
        ok = False
    # --- ask judge
    from evaluation.missing_parameters import parameters_from_row
    from evaluation import simulated_user

    params = parameters_from_row({"open_items": items, "instruction": instr, "ambiguity": "Ambiguous"})
    try:
        hit = simulated_user.ask_targets("What price is the home you want to calculate for?", instr, params)
        miss = simulated_user.ask_targets("Which browser are you using?", instr, params)
        out["ask_judge"] = {"on_item": list(hit), "off_item": list(miss), "n_params": len(params)}
        if not hit:
            ok = False
    except Exception as e:  # noqa: BLE001
        out["ask_judge"] = {"error": f"{type(e).__name__}: {e}"[:400]}
        ok = False
    # --- policies
    from openai import OpenAI
    import prompt_variant  # noqa: PLC0415

    prompt_variant.apply()
    from training.cascp_common import CASCP_SYSTEM
    from evaluation.tool_contract import tool_json_schema
    from scripts.eval_cascp_online import parse_tool

    buf = io.BytesIO()
    Image.open(shot).resize((768, 432)).save(buf, "PNG")
    du = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    # An open policy is served by its own local vLLM; an API policy by its OpenAI-compatible URL.
    from evaluation.endpoint_auth import api_key_for

    policy_url = os.environ["POLICY_URL"]
    c = OpenAI(base_url=policy_url, api_key=api_key_for(policy_url), max_retries=0, timeout=900)
    pol = {}
    for m in [x for x in os.environ.get("POLICIES", "").split(",") if x]:
        t = time.time()
        try:
            r = c.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": CASCP_SYSTEM},
                    {"role": "user", "content": [{"type": "text", "text": f"Instruction: {instr}"}, {"type": "image_url", "image_url": {"url": du}}]},
                ],
                tools=[tool_json_schema()],
                tool_choice="none",
                max_tokens=int(os.environ["MAX_TOKENS"]),
                temperature=0.0,
                seed=0,
            )
            txt = r.choices[0].message.content or ""
            pol[m] = {"ok": True, "s": round(time.time() - t, 1), "finish": r.choices[0].finish_reason, "parsed": parse_tool(txt), "text": txt[:400]}
        except Exception as e:  # noqa: BLE001
            pol[m] = {"ok": False, "s": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}"[:300]}
    out["policies"] = pol
    out["ok_dependencies"] = ok
    print(json.dumps(out, indent=1, default=str))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
