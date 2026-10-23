# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Strict post-multimodal headroom guard for the ICL arm: keep the full 8192-token output budget.

vLLM 0.23.0 checks the prompt length against ``context - max_tokens`` BEFORE it expands each
screenshot placeholder into image tokens, and afterwards silently lowers ``max_tokens`` if the
expanded prompt no longer leaves 8192 tokens. The four demonstrations make ICL prompts longer, so
without a guard an ICL request could get a smaller output budget than the matched zero-shot
request, unnoticed. This guard counts the fully expanded prompt exactly as the server will
(the server's own resize and image-token functions, the checkpoint's tokenizer and template) and
REJECTS a request when ``expanded prompt tokens + 8192 > 65536``. It never trims history, drops
images or lowers the budget. A rejected request raises; the harness records it as a
``generation_error`` ending of that rollout. Every check, accepted or rejected, is appended to
``strict_headroom_checks.jsonl`` beside the shard's journal (``status``: ``accepted`` with the
count evidence, or ``rejected`` with the reason and the evidence, written before the error is raised).

Installed inside the lossless recorder, after the demonstration injection:
injection -> recorder -> guard -> original transport.
"""
from __future__ import annotations
import base64
import copy
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import time
from .native_cpu_defs import vision_definitions,detect_content_format,native_tool_dicts
CONTEXT=65536
RESERVE=8192


class HeadroomRejected(ValueError):
    def __init__(self,reason,evidence=None):
        self.evidence=evidence or {}
        super().__init__('ICL4_STRICT_HEADROOM_REJECTED: '+reason+'; '+json.dumps(self.evidence,ensure_ascii=False,sort_keys=True))


def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()


def exact_full_reserve(post_mm_prompt_tokens):
    if type(post_mm_prompt_tokens) is not int or post_mm_prompt_tokens<1:
        raise HeadroomRejected('invalid processed prompt length')
    if post_mm_prompt_tokens+RESERVE>CONTEXT:
        raise HeadroomRejected('fully expanded prompt cannot retain the original8192 budget',
                               {'post_MM_prompt_tokens':post_mm_prompt_tokens,'required_output_tokens':RESERVE,
                                'context':CONTEXT,'available_output_tokens':CONTEXT-post_mm_prompt_tokens,
                                'clamp_performed':False,'trim_performed':False})
    return post_mm_prompt_tokens


class PinnedImageTextCounter:
    def __init__(self,condition):
        # condition: ambigui.icl.condition.FrozenCondition (already verified: fingerprint and
        # every asset sha256, including the nine processor/tokenizer/template files).
        asset_root=condition.root
        self.asset_root=asset_root
        self.model=condition.served_model
        self.template=(asset_root/'assets/processor/chat_template.jinja').read_text()
        self.config=json.loads((asset_root/'assets/processor/config.json').read_bytes())
        self.pconfig=json.loads((asset_root/'assets/processor/preprocessor_config.json').read_bytes())
        self.tools=json.loads((asset_root/'assets/tools.json').read_bytes())
        self.system=(asset_root/'assets/system_prompt.txt').read_text()
        from transformers.utils.chat_template_utils import _compile_jinja_template
        from tokenizers import Tokenizer
        self.compiled=_compile_jinja_template(self.template)
        self.tokenizer=Tokenizer.from_file(str(asset_root/'assets/processor/tokenizer.json'))
        if self.tokenizer.token_to_id('<|image_pad|>')!=self.config['image_token_id']:
            raise HeadroomRejected('native image token id differs')
        if detect_content_format(self.template)!='openai':raise HeadroomRejected('native content format is not the pinned openai format')
        self.native_tools=native_tool_dicts(self.tools)
        if self.native_tools!=self.tools:raise HeadroomRejected('native schema serialization differs')
        self.resize,self.patch_count,self.vision_info,_=vision_definitions()

    def image_geometry(self,url):
        from PIL import Image
        from types import SimpleNamespace
        if not isinstance(url,str) or not url.startswith('data:image/png;base64,'):
            raise HeadroomRejected('only original inline PNG screenshots are supported by this exact candidate')
        raw=base64.b64decode(url.split(',',1)[1],validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if image.format!='PNG':raise HeadroomRejected('PNG carrier does not contain PNG bytes')
            width,height=image.size
        processor=SimpleNamespace(size=self.pconfig['size'],patch_size=self.pconfig['patch_size'],merge_size=self.pconfig['merge_size'])
        hf=SimpleNamespace(vision_config=SimpleNamespace(**self.config['vision_config']))
        info=SimpleNamespace(ctx=SimpleNamespace(get_merged_mm_kwargs=lambda kw:kw),get_hf_config=lambda:hf)
        size,count=self.vision_info(info,image_width=width,image_height=height,image_processor=processor,mm_kwargs={})
        independently=self.patch_count(processor,height,width,{})//self.pconfig['merge_size']**2
        if count!=independently:raise HeadroomRejected('native vision sizing and native HF patch sizing disagree')
        return {'raw_image_sha256':sha(raw),'width':width,'height':height,'resized_width':size.width,'resized_height':size.height,
                'merged_image_tokens':count,'grid_t':1,'grid_h':size.height//self.pconfig['patch_size'],
                'grid_w':size.width//self.pconfig['patch_size'],'merge_size':self.pconfig['merge_size']}

    def count(self,request):
        allowed={'model','messages','tools','tool_choice','max_tokens','temperature','seed','stream'}
        unknown=set(request)-allowed
        if unknown:raise HeadroomRejected('unpinned request fields could alter rendering/expansion: '+str(sorted(unknown)))
        if request.get('model')!=self.model or request.get('max_tokens')!=8192 or request.get('temperature')!=0 or request.get('seed')!=0:
            raise HeadroomRejected('original model/output/decoding differs')
        if request.get('tools')!=self.tools or request.get('tool_choice')!='none' or request.get('stream') is True:
            raise HeadroomRejected('original schema/protocol differs')
        messages=request.get('messages')
        if not isinstance(messages,list) or not messages or messages[0]!={'role':'system','content':self.system}:
            raise HeadroomRejected('original p0 differs')
        conversation=[];images=[]
        for m in messages:
            if not isinstance(m,dict) or set(m)-{'role','content'}:
                raise HeadroomRejected('unpinned message representation; native tool_calls/reasoning/embed fields not admitted')
            if m['role'] not in {'system','user','assistant'}:raise HeadroomRejected('unpinned role')
            content=m['content']
            if isinstance(content,str):parts=[{'type':'text','text':content}] if content else []
            elif isinstance(content,list):
                parts=[]
                for part in content:
                    if not isinstance(part,dict):raise HeadroomRejected('unpinned content part')
                    if part.get('type')=='text' and set(part)=={'type','text'} and isinstance(part['text'],str):
                        if part['text']:parts.append(copy.deepcopy(part))
                    elif part.get('type')=='image_url' and set(part)=={'type','image_url'} and set(part['image_url'])=={'url'}:
                        images.append(self.image_geometry(part['image_url']['url']));parts.append({'type':'image'})
                    else:raise HeadroomRejected('only original text and inline screenshot parts are supported')
            else:raise HeadroomRejected('unpinned message content')
            conversation.append({'role':m['role'],'content':parts})
        if len(images)>4:raise HeadroomRejected('image4 exceeded; do not prune or trim in guard')
        # Native build_chat_params defaults: add_generation_prompt=True,
        # continue_final_message=False, reasoning_effort=None; original template
        # ignores unsupported None fields. No thinking/history/tool rewrite.
        rendered=self.compiled.render(messages=conversation,tools=self.native_tools,add_generation_prompt=True)
        encoded=self.tokenizer.encode(rendered,add_special_tokens=False).ids
        positions=[i for i,t in enumerate(encoded) if t==self.config['image_token_id']]
        if len(positions)!=len(images):raise HeadroomRejected('placeholder count differs from actual screenshots',
                                                {'image_placeholders':len(positions),'images':len(images)})
        expanded=[];image_index=0
        for token in encoded:
            if token==self.config['image_token_id']:
                expanded.extend([token]*images[image_index]['merged_image_tokens']);image_index+=1
            else:expanded.append(token)
        count=len(expanded)
        return {'counter_version':'pinned_Qwen35_native_geometry_and_raw_tokenizer_v1',
                'request_sha256':sha(canonical(request)),'rendered_prompt_sha256':sha(rendered.encode()),
                'pre_MM_token_count':len(encoded),'post_MM_token_count':count,'image_geometries':images,
                'fully_expanded_token_ids_sha256':sha(canonical(expanded)),
                'required_output_tokens':8192,'context':65536,'available_output_tokens':65536-count,
                'messages_or_images_or_budget_modified':False,
                'provider_parity':'assumes vLLM 0.23.0 served by harness/policy_serve.sh with its default processor and template settings'}

    def enforce(self,request):
        evidence=self.count(request)
        exact_full_reserve(evidence['post_MM_token_count'])
        return evidence


def _log_check(entry):
    shot=os.environ.get('CASCP_SHOT_DIR')
    if shot:
        out=Path(shot).parent;out.mkdir(parents=True,exist_ok=True)
        with (out/'strict_headroom_checks.jsonl').open('a') as f:f.write(json.dumps(dict(entry,checked_unix=time.time()),default=str)+'\n')


def install_inside_recorder(counter,completions_type=None):
    """MUST run before original recorder.apply and outer frozen demo injection.

    Order becomes original injection -> original recorder -> STRICT guard ->
    original transport. Both accepted requests and guard exceptions are recorded
    with expanded messages by the unchanged original recorder.
    """
    if completions_type is None:
        from openai.resources.chat.completions import Completions
        completions_type=Completions
    original=completions_type.create
    if getattr(original,'_ambigui_recorded',False):raise HeadroomRejected('guard must be installed INSIDE recording, not outside')
    if getattr(original,'_icl4_strict_headroom_v1',False):return original
    @functools.wraps(original)
    def create(self,*args,**kwargs):
        if kwargs.get('model')==counter.model:
            if args:raise HeadroomRejected('positional policy request is not the native matched contract')
            try:
                proof=dict(counter.enforce(kwargs),status='accepted')
            except HeadroomRejected as exc:
                try:request_sha=sha(canonical(kwargs))
                except (TypeError,ValueError):request_sha=None
                _log_check({'status':'rejected','reason':str(exc),'evidence':exc.evidence,'request_sha256':request_sha})
                raise
            # Separate counter evidence only; no extra field enters API payload.
            _log_check(proof)
        return original(self,*args,**kwargs)
    create._icl4_strict_headroom_v1=True
    completions_type.create=create
    return create


def install_for_runtime(condition):
    """Build the counter from the frozen condition and install the guard (ICL arm only)."""
    counter=PinnedImageTextCounter(condition)
    install_inside_recorder(counter)
    return counter
