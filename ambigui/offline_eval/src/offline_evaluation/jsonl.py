# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reading the line-per-record files this package writes and reads.

One definition, because four programs in :mod:`offline_evaluation.run` need it and four
copies of a loop is four chances for them to disagree about what a blank line means.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each JSON object in ``path``.

    A blank line is skipped rather than raising. These files are written a record at a time
    and are routinely concatenated or truncated by hand, so a stray newline is a formatting
    accident and not a corrupt record.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
