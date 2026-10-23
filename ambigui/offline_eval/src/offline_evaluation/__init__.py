# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Offline evaluation of an ask-or-act web agent, from recorded trajectories.

An ask-or-act agent is shown a task and a screen and must either take an action or stop and
ask the user a clarifying question. This package scores that choice, and the action when it
acts, without driving a live browser: the trajectories are already recorded.

Every recorded step is scored, with the recording's own history handed to the agent at each
one, so that one early mistake cannot make the rest unanswerable. This is called *teacher
forcing*. Each open item also carries a deadline by which a question about it must have
been asked.

The protocol runs in three stages, each its own program (see :mod:`offline_evaluation.run`):

``run_ask_protocol``
    the policy acts or asks at every step; a simulated user answers its questions.
``judge_asks``
    a judge model reads every recorded question and says which open items it covered.
``score_ask_protocol``
    pure arithmetic over the files the first two wrote; no model and no network.

Asking a model and marking its answers are separate programs because asking costs GPU hours
and marking costs none: a changed grading rule must never force the expensive half to run
again.

The modules the stages share:

:mod:`~offline_evaluation.gold_benchmark`
    Reading the released benchmark file into one item per recorded step.
:mod:`~offline_evaluation.substrate`
    The per-step item the stages pass to each other.
:mod:`~offline_evaluation.calls`
    Reading one emission back into the action the policy meant.
:mod:`~offline_evaluation.prompts`
    The message an item is posed with. Every published number is a number about this text.
:mod:`~offline_evaluation.client`
    The model side: which models are known, how a screen is encoded, and one call against
    an OpenAI-compatible endpoint.
:mod:`~offline_evaluation.metrics`, :mod:`~offline_evaluation.ask_variants`, :mod:`~offline_evaluation.ask_timing`
    The action, ask-content and ask-timing metrics.
:mod:`~offline_evaluation.jsonl`
    Reading the line-per-record files the programs write and read.
"""
