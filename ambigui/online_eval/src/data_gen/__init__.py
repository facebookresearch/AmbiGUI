# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""``data_gen`` -- helpers for the benchmark's task data.

In this release only :mod:`data_gen.underspecification` ships. It renders a task's open
items (the details its instruction leaves unstated, each with the answer the person chose)
in the form the simulated user reads them. The task file is never in the repository; see
``docs/online.md`` for how it is supplied.
"""
