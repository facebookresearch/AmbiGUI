# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Strict re-check of a selected demonstration question: question-or-statement gate, then which open items it targets."""
from __future__ import annotations

from typing import Any

JUDGE_PROTOCOL_VERSION = 'strict-question-and-set-v2'


def verdict_text(raw: str) -> str:
    text=(raw or '').strip()
    if '<think>' in text and '</think>' not in text:
        raise ValueError('unclosed/truncated judge reasoning')
    return text.rsplit('</think>',1)[-1].strip()


def parse_kind_strict(raw: str) -> str:
    value=verdict_text(raw).casefold()
    if value not in {'question','statement'}:
        raise ValueError(f'judge kind must be exactly QUESTION or STATEMENT: {raw[:120]!r}')
    return value


def parse_targets_strict(raw: str, parameters: tuple) -> list[str]:
    value=verdict_text(raw)
    if value.casefold()=='none':return []
    names={p.name.casefold():p.name for p in parameters}
    pieces=[p.strip().casefold() for p in value.split(',')]
    if any(not p or p not in names for p in pieces):
        raise ValueError(f'judge target list contains unknown/mixed/malformed tokens: {raw[:160]!r}')
    if len(pieces)!=len(set(pieces)):
        raise ValueError('judge target list repeats parameter names')
    return [names[p] for p in pieces]


def grade_message(client: Any, model: str, instruction: str, message: str,
                  parameters: tuple, *, max_tokens: int = 1024) -> dict:
    """Posthoc only: malformed, contradictory and unavailable grades are missing."""
    from evaluation.message_kind import PROMPT
    from evaluation.missing_parameters import targets_prompt
    result={'message':message,'kind':None,'targets':[],'valid':False,'error':None,
            'raw_verdicts':[],'judge_model':model,'judge_protocol':JUDGE_PROTOCOL_VERSION}

    def call(prompt: str, stage: str) -> str:
        response=client.chat.completions.create(model=model,messages=[{'role':'user','content':prompt}],
                                              max_tokens=max_tokens,temperature=0.0)
        choice=response.choices[0];text=choice.message.content or ''
        result['raw_verdicts'].append({'stage':stage,'prompt':prompt,'output':text,'finish_reason':choice.finish_reason})
        if choice.finish_reason=='length':raise ValueError(f'{stage} judge truncated')
        return text

    try:
        kind=parse_kind_strict(call(PROMPT.format(instruction=instruction,message=message),'question_gate'))
        result['kind']=kind
        if kind=='question' and parameters:
            raw=call(targets_prompt(instruction=instruction,question=message,parameters=parameters),'set_targets')
            result['targets']=parse_targets_strict(raw,parameters)
        result['valid']=True
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    return result
