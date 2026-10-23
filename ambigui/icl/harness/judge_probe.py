#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""judge_probe.py <url> <model>: one real multimodal completion (3 small images); exit 0 iff it answers."""
import base64, io, sys, time
import requests
from PIL import Image, ImageDraw
url, model = sys.argv[1:]
parts = [{"type": "text", "text": "Three screenshots follow. Reply with the word written in each, comma-separated."}]
for w in ("alpha", "bravo", "charlie"):
    im = Image.new("RGB", (1000, 600), "white"); ImageDraw.Draw(im).text((400, 280), w, fill="black")
    b = io.BytesIO(); im.save(b, "PNG")
    parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}})
t = time.time()
try:
    from evaluation.endpoint_auth import auth_headers  # src/ is on PYTHONPATH
    r = requests.post(url + "/chat/completions", json={"model": model, "messages": [{"role": "user", "content": parts}], "max_tokens": 50, "temperature": 0}, timeout=600, headers=auth_headers(url))
    print(f"[judge-probe] {r.status_code} {time.time()-t:.1f}s {r.text[:300]}")
    sys.exit(0 if r.status_code == 200 else 1)
except Exception as e:
    print(f"[judge-probe] {type(e).__name__}: {e}"); sys.exit(1)
