# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Reads the tool call (name and arguments) out of a raw policy reply."""
from __future__ import annotations

import json
import re


def extract_call(raw: str) -> tuple[str | None, dict]:
    visible=(raw or '').rsplit('</think>',1)[-1]
    functions=list(re.finditer(r'<function=([A-Za-z_][\w]*)>',visible,re.I))
    if functions:
        last=functions[-1]
        args={k:v.strip() for k,v in re.findall(r'<parameter=([A-Za-z_][\w]*)>(.*?)</parameter>',visible[last.end():],re.S|re.I)}
        return last.group(1),args
    decoder=json.JSONDecoder();position=0
    while True:
        start=visible.find('{',position)
        if start<0:return None,{}
        try:obj,length=decoder.raw_decode(visible[start:])
        except json.JSONDecodeError:position=start+1;continue
        if isinstance(obj,dict) and 'name' in obj:
            args=obj.get('arguments') or obj.get('parameters') or {}
            return str(obj['name']),args if isinstance(args,dict) else {}
        position=start+length
