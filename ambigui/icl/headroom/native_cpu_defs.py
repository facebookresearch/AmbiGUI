# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Run selected EXACT definitions of the installed vLLM 0.23.0 and transformers 5.14.1 sources
without importing vLLM or a model backend.

The headroom guard must count prompt tokens exactly as the vLLM server will after it expands
each screenshot into image tokens. Rather than re-implement that, it executes the server's own
function bodies (image resizing, image-token count, chat-template content format, tool-schema
serialisation), selected by name from the installed source files. Every file is checked against
the sha256 pinned in ``native_sources.json`` (the bytes of the reported run; identical to the
PyPI vLLM 0.23.0 wheel and transformers 5.14.1), and every selected definition is recorded.
"""
from __future__ import annotations
import ast
from collections import deque
from dataclasses import dataclass,field
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any,ClassVar,Literal,TypeVar
import importlib.util
import os
ROOT=Path(__file__).resolve().parent
RECEIPT=json.loads((ROOT/'native_sources.json').read_bytes())
ENTRY_BY_REL={e['path']:e for e in RECEIPT['files']}
SELECTIONS=[]


def package_root(package):
    """The site-packages folder holding an INSTALLED package, without importing it.

    vLLM is usually installed in a separate serving environment; ICL_VLLM_SITE_PACKAGES points
    at that environment's site-packages (run_online.sh fills it in from VLLM_PY).
    """
    if package=='vllm' and os.environ.get('ICL_VLLM_SITE_PACKAGES'):
        return Path(os.environ['ICL_VLLM_SITE_PACKAGES'])
    spec=importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise ValueError(f'{package} is not installed here; for vllm set ICL_VLLM_SITE_PACKAGES')
    return Path(list(spec.submodule_search_locations)[0]).parent


def extract(relative,name,namespace,owner=None):
    p=package_root(relative.split('/',1)[0])/relative;b=p.read_bytes()
    if hashlib.sha256(b).hexdigest()!=ENTRY_BY_REL[relative]['sha256']:raise ValueError('pinned native bytes differ: '+relative)
    tree=ast.parse(b)
    if owner is not None:tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner)
    found=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    if len(found)!=1:raise ValueError('ambiguous native AST selector: '+name)
    n=found[0]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module,str(p),'exec'),namespace)
    SELECTIONS.append({'source':relative,'owner':owner,'definition':name,'lines':[n.lineno,n.end_lineno],
                       'source_file_sha256':hashlib.sha256(b).hexdigest(),'ast_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()})
    return namespace[name]


def base_namespace():
    from types import ModuleType
    name='isolated_pinned_native_defs_'+str(len(SELECTIONS))
    module=ModuleType(name);sys.modules[name]=module
    module.__dict__.update({'__name__':name,'Any':Any,'ClassVar':ClassVar,'Literal':Literal,'TypeVar':TypeVar,
            'dataclass':dataclass,'field':field,'math':math,'sys':sys,'_S':TypeVar('_S'),
            'logger':SimpleNamespace(debug=lambda *a,**k:None,warning=lambda *a,**k:None,exception=lambda *a,**k:None)})
    return module.__dict__


def vision_definitions():
    ns=base_namespace()
    resize=extract('transformers/models/qwen2_vl/image_processing_qwen2_vl.py','smart_resize',ns)
    patches=extract('transformers/models/qwen2_vl/image_processing_qwen2_vl.py','get_number_of_image_patches',ns,'Qwen2VLImageProcessor')
    class ImageSize:
        def __init__(self,*,width,height):self.width=width;self.height=height
    class VideoProcessor:pass
    ns.update(ImageSize=ImageSize,Qwen3VLVideoProcessor=VideoProcessor,image_smart_resize=resize,
              round_up=lambda n,d:(n+d-1)//d*d)
    vision_info=extract('vllm/model_executor/models/qwen3_vl.py','_get_vision_info',ns,'Qwen3VLProcessingInfo')
    return resize,patches,vision_info,ns


def detect_content_format(template):
    import jinja2
    from transformers.utils.chat_template_utils import _compile_jinja_template
    ns=base_namespace();ns.update(jinja2=jinja2,deque=deque,lru_cache=lru_cache)
    for name in ['_is_var_access','_is_attr_access','_is_var_or_elems_access','_iter_nodes_assign_var_or_elems',
                 '_iter_nodes_assign_messages_item','_iter_nodes_assign_content_item']:
        extract('vllm/renderers/hf.py',name,ns)
    ns['_try_extract_ast']=lambda s:_compile_jinja_template(s).environment.parse(s)
    fn=extract('vllm/renderers/hf.py','_detect_content_format',ns)
    return fn(template,default='string')


def native_tool_dicts(tools):
    from pydantic import BaseModel,ConfigDict,model_validator,model_serializer
    ns=base_namespace();ns.update(BaseModel=BaseModel,ConfigDict=ConfigDict,model_validator=model_validator,model_serializer=model_serializer)
    extract('vllm/entrypoints/openai/engine/protocol.py','OpenAIBaseModel',ns)
    extract('vllm/entrypoints/openai/engine/protocol.py','FunctionDefinition',ns)
    cls=extract('vllm/entrypoints/openai/chat_completion/protocol.py','ChatCompletionToolsParam',ns)
    return [cls.model_validate(t).model_dump() for t in tools]
