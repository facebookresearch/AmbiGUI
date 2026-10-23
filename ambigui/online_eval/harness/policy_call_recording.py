#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Record lossless OpenAI policy request/response payloads without altering inference.

Images are extracted into content-addressed blobs; the request JSON refers to
those bytes rather than repeating base64 at every turn. Text and all response
fields, including provider reasoning, are retained without truncation.
"""
from __future__ import annotations

import base64
import copy
import functools
import hashlib
import json
import os
import time
from pathlib import Path


def preserve_images(value, directory):
    if isinstance(value, dict):
        return {key: preserve_images(item, directory) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [preserve_images(item, directory) for item in value]
    if isinstance(value, str) and value.startswith('data:image/') and ';base64,' in value:
        header, raw = value.split(',', 1)
        data = base64.b64decode(raw, validate=True)
        digest = hashlib.sha256(data).hexdigest()
        suffix = '.png' if 'image/png' in header else '.jpg'
        target = directory / (digest + suffix)
        directory.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        return {'__lossless_image_blob__': str(target.name), 'sha256': digest,
                'data_url_header': header, 'bytes': len(data)}
    return value


def apply():
    if os.environ.get('RECORD_POLICY_CALLS') != '1':
        return
    from openai.resources.chat.completions import Completions
    original = Completions.create
    if getattr(original, '_ambigui_recorded', False):
        return
    directory = Path(os.environ['CASCP_SHOT_DIR']).parent
    blobs = directory / 'policy_image_blobs'
    journal = directory / 'policy_calls.jsonl'

    @functools.wraps(original)
    def create(self, *args, **kwargs):
        started = time.time()
        record = {'started_unix': started,
                  'request': preserve_images(copy.deepcopy(kwargs), blobs),
                  'request_positional': preserve_images(copy.deepcopy(args), blobs)}
        try:
            result = original(self, *args, **kwargs)
            record.update(status='ok', response=result.model_dump(mode='json'))
            return result
        except Exception as exc:
            record.update(status='error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record['elapsed_s'] = time.time() - started
            directory.mkdir(parents=True, exist_ok=True)
            with journal.open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

    create._ambigui_recorded = True
    Completions.create = create
