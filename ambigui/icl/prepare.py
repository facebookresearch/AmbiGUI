# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Build and verify the four fixed demonstrations once, on CPU, before any evaluated task runs.

Reads only training-side inputs (the training task file and the harvest chunks) plus the
evaluated and benchmark task files for the exclusion check. Only training tasks from approved
sources are used (``demos.SOURCE_RULE``). The only model calls are the fresh re-verification of the
two selected questions by the ask judge (``JUDGE_URL``): four calls, two per question. Writes
``demonstrations.json``, ``demo_messages.json``, ``demo_source_records.json``, ``tools.json`` and
``system_prompt.txt`` to ``--out``; pass that folder to
``python -m ambigui.icl.prepare_condition --demos-dir``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .common import write_json
from .demos import build, verify_frozen
from .journal import Journal


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--train-tasks',type=Path,required=True)
    parser.add_argument('--tasks',type=Path,required=True,help='the task file that will be evaluated (exclusion only)')
    parser.add_argument('--exclusion-tasks',type=Path,nargs='*',default=[])
    parser.add_argument('--harvest-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    inputs={'train':args.train_tasks,'eval':args.tasks,'exclusion':list(args.exclusion_tasks)}
    model=cfg['judge_model']; url=os.environ['JUDGE_URL']
    try:
        journal=Journal(args.out/'validation',context={'task_id':'training-demo-verification','seed':cfg['seed'],
                        'stage':'cpu-training-only-preparation'},grader_model=model)
        journal.install()
        try:
            from openai import OpenAI
            judge=OpenAI(base_url=url,api_key='EMPTY',timeout=900,max_retries=0)
            manifest=build(cfg,inputs,args.harvest_root,args.out,judge,model)
        finally:
            journal.restore()
        verify_frozen(manifest,inputs,cfg,json.loads((args.out/'demo_source_records.json').read_text()))
        from evaluation.tool_contract import tool_json_schema
        from training.cascp_common import CASCP_SYSTEM
        write_json(args.out/'tools.json',[tool_json_schema()])
        (args.out/'system_prompt.txt').write_text(CASCP_SYSTEM)
        validation={'status':'completed','demo_fingerprint':manifest['fingerprint'],
                    'demonstration_ids':[d['task_id'] for d in manifest['demonstrations']],
                    'actual_demonstration_messages_sha256':manifest['demo_messages_sha256'],
                    'evaluated_task_policy_calls':0,'training_only':True}
        write_json(args.out/'preparation.json',validation)
        print('[ICL] demonstrations prepared '+json.dumps(validation),flush=True)
        return 0
    except Exception as exc:
        import traceback
        error={'status':'failed','error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(),
               'evaluated_task_policy_calls':0}
        write_json(args.out/'preparation_failure.json',error)
        print('[ICL] demonstration preparation failed '+json.dumps(error),flush=True)
        return 5


if __name__=='__main__':
    sys.exit(main())
