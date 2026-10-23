# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Frozen-weight in-context learning (ICL) baseline for AmbiGUI; deliberately no training imports.

The modules of this part import the copied online-evaluation harness in ``src/`` as top-level
packages (``evaluation``, ``scripts``, ``training``, ...). Importing ``ambigui.icl`` puts that
folder on ``sys.path``; nothing outside ``ambigui/icl`` is read.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

__version__ = "1.0.0"
