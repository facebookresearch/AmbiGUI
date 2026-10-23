# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Did the agent finish the task? Decided by a reader that can ask about the screen.

The judge reads the action history. It cannot see anything, so when the verdict
turns on something visual it asks a second, vision-capable model one or two
specific questions and decides with the answers in hand.

That split is the point. Three earlier designs failed in instructive ways, and
the control numbers below are all from the same sixty recordings of **people**
doing the task, so the correct answer is sixty out of sixty every time:

===========================================  ========  ===============
 design                                       correct   false alarms
===========================================  ========  ===============
 this one: ask, then look, then decide        26-28         0-1
 same judge, no lookup                           4           0
 the published screenshot judge                  9           0
 a judge handed the whole page text               2           0
===========================================  ========  ===============

Two findings shaped it.

**More evidence made judges worse, not better.** Handing a judge the full page
contents dropped it from 13 to 9, and on another prompt from 8 to 2. The rubric
fails whatever it cannot confirm, so extra detail is extra doubt. Asking one
question returns the single fact needed and nothing to be unsure about.

**A small describer agrees rather than sees.** A 2B model scored 41 of 60 --
better than anything here -- by answering "yes" to almost everything, including
"is the toggle enabled?" when the toggle was off. Three cases were adjudicated
by hand and it was wrong in all three. Its score was agreeableness, and the
false-alarm column is what exposes it: 5, against 0 here.

**Evidence is usually not on the last screen.** All three adjudicated failures
had the thing to check part-way through, while the session ended on a terminal
or a modal. Spreading the frames across the whole recording, and telling the
describer that a ticked checkbox counts as much as a displayed result, took the
8B model from 16 to 28.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

#: Frames shown to the describer: the opening screen, an even spread through
#: the middle, and the closing few. Clustering at the end missed evidence that
#: sat mid-session, which was the single largest source of wrong verdicts.
FIRST_FRAMES = 1
SPREAD_FRAMES = 5
LAST_FRAMES = 4

#: Questions the judge may ask in one round. More than a few and it starts
#: asking things that cannot change the verdict.
MAX_QUESTIONS = 3

ASK_OR_DECIDE = """You are an expert evaluator of a web navigation agent.

Decide whether the agent completed the task.

Important criteria:
1. Requirements such as "latest", "best", "cheapest", "highest-rated" must be
   satisfied. They may be satisfied by a filter the agent applied, OR by the
   page already being ordered that way. You cannot see the page, so if this is
   the only thing in doubt, ask about it rather than assuming.
2. A task that needs a result displayed is not complete until it is displayed.
3. Repeating actions with no progress is a failure.
4. An action marked DID NOT HAPPEN was never carried out. Do not credit it.
4b. An action marked THE AGENT DID THIS; THE HARNESS DID NOT EXECUTE IT is the
   opposite case: the agent performed it, and only the real-world effect was
   blocked, because the control commits something on a live site. Credit the
   press itself as having happened.
   It is NOT success on its own. The task is complete only if that control was
   the right final step for THIS task AND everything leading up to it was
   correct. Pressing the wrong control is a failure. Reaching the right control
   with the wrong item, the wrong search, or a form filled in wrongly is a
   failure. Judge the trajectory, not the marker.
5. An agent that asked the user a question and then acted on the answer has
   behaved correctly. Asking is not a failure. Questions about how to operate
   the site are neither to its credit nor against it.
6. A claim by the agent that it finished is not evidence that it did. The
   trajectory may end with the agent declaring the task complete; that ends the
   episode and nothing more. Decide from the screens and from the actions that
   were actually performed. Never treat the declaration itself as success.

TASK
{task}

ACTION HISTORY
{actions}

You cannot see the screen. A colleague can look at screenshots from across the
whole session -- the start, the middle and the end -- and answer specific
questions about what is visible in any of them.

A requirement can be met by a setting being changed, not only by a result being
displayed: a checkbox ticked, a filter applied, a button becoming enabled. If
that is what is in doubt, ask about it directly.

If the action history alone settles it, answer:
Status: "success" or "failure"

If you need to see the screen, instead ask up to {max_q} short yes/no questions
about what the screenshots show. Ask only what would change your verdict.
Answer in this form:
Questions:
1. <question>
2. <question>"""

LOOK = """You are looking at screenshots of one session, in order. Image 1 is
the start. The later images run through to the end.

Answer each question about what is visible across ALL the images.

Rules you must follow:

- **Look at every image, not just the last one.** What a question asks about is
  often part-way through. A session can end on a terminal, a modal or a
  confirmation page while the thing being asked about was two screens earlier.
  Say which image number your evidence comes from.

- **A setting changed counts, not only a result displayed.** These all count as
  the thing having been done, even if the outcome is not on screen:
    - a checkbox or toggle ticked in a later image and not in an earlier one
    - a filter chip, tag or "applied" label that appears
    - a button such as Apply, Save or Search becoming enabled
    - a field that now holds a value it did not hold before
  If you see one of these, say so and name the two images it differs between.

- **Compare across images before answering a question about change.**
  Never restate the question as your evidence. Quote something you can see: a number,
  a label, a position, a state.

- If nothing in any image answers the question, say "cannot tell". That is a
  correct answer, not a failure. Do not guess.

- Do not judge whether the overall task was completed. Report what is visible.

You have {n_images} images.

{questions}

Answer in this form, one line per question:
1. yes / no / cannot tell -- <evidence, naming the image number>"""

DECIDE = """You are an expert evaluator of a web navigation agent.

TASK
{task}

ACTION HISTORY
{actions}

You asked a colleague to look at the screenshots. They report:

{answers}

Now decide. A requirement such as "latest" or "highest-rated" counts as met if
the screen shows it holding, whether or not the agent clicked a filter. A
setting visibly changed counts even if the result is not displayed.
"cannot tell" is not confirmation.

An action marked DID NOT HAPPEN was never carried out; do not credit it. An
action marked THE AGENT DID THIS; THE HARNESS DID NOT EXECUTE IT did happen as
far as the agent is concerned -- the control commits something on a live site,
so only its real-world effect was blocked. Credit the press as having happened,
and then judge the trajectory exactly as you would any other: it is success only
if that was the right final step for this task and everything leading up to it
was correct. The marker is not a verdict.

A claim by the agent that it finished is not evidence that it did. If the
trajectory ends with the agent declaring the task complete, that declaration ends
the episode and nothing more. Decide from the screens and from the actions that
were actually performed.

Format your response into two lines:
Thoughts: <your reasoning>
Status: "success" or "failure"
"""


class JudgeUnreadable(RuntimeError):
    """The judge's reply carried no verdict.

    Raised rather than returning False. A reasoning model that runs out of
    tokens before its answer returns no verdict, and reading that as "not
    successful" turns a broken judge into a page of plausible zeroes.
    """


def select_frames(shots: Sequence[str]) -> list[str]:
    """The screenshots to show, spread across the whole session in order."""
    every = list(shots)
    if not every:
        return []
    picked = list(every[:FIRST_FRAMES])
    tail_at = max(FIRST_FRAMES, len(every) - LAST_FRAMES)
    body = every[FIRST_FRAMES:tail_at]
    if body:
        step = max(1, len(body) // SPREAD_FRAMES)
        picked += body[::step][:SPREAD_FRAMES]
    picked += every[tail_at:]
    out: list[str] = []
    for path in picked:
        if path not in out:
            out.append(path)
    return out


def parse_status(reply: str) -> bool | None:
    """The verdict in a reply, or None when it carries none."""
    low = str(reply or "").lower()
    if "status:" not in low:
        return None
    return "success" in low.rsplit("status:", 1)[1]


def parse_questions(reply: str) -> list[str]:
    """The questions the judge wants answered, capped."""
    out = []
    for line in str(reply or "").splitlines():
        match = re.match(r"\s*\d+[.)]\s*(.+)", line)
        if match and "?" in match.group(1):
            out.append(match.group(1).strip())
    return out[:MAX_QUESTIONS]


def build_ask_or_decide(task: str, actions: str) -> str:
    return ASK_OR_DECIDE.format(
        task=str(task).strip(), actions=actions, max_q=MAX_QUESTIONS
    )


def build_look(questions: Sequence[str], n_images: int) -> str:
    return LOOK.format(
        n_images=n_images,
        questions="\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1)),
    )


def build_decide(task: str, actions: str, answers: str) -> str:
    return DECIDE.format(task=str(task).strip(), actions=actions, answers=answers)


#: Drives the judge: a prompt in, its reply out.
AskFn = Callable[[str], str]
#: Drives the describer: a prompt and the frames in, its reply out.
LookFn = Callable[[str, Sequence[str]], str]


def judge(
    task: str,
    actions: str,
    shots: Sequence[str],
    ask_fn: AskFn,
    look_fn: LookFn,
) -> tuple[bool, str]:
    """``(success, why)``, asking about the screen only when it would decide.

    ``ask_fn`` drives the judge, ``look_fn`` the describer. They are passed in
    so the decision logic is testable without a serve, and so the two can be
    different models -- which they must be: the describer reports, the judge
    decides, and letting one model do both was measurably worse.
    """
    first = ask_fn(build_ask_or_decide(task, actions))
    verdict = parse_status(first)
    questions = parse_questions(first)
    frames = select_frames(shots)

    if verdict is not None and not questions:
        return verdict, f"decided without looking: {_tail(first)}"
    if not questions or not frames:
        if verdict is None:
            raise JudgeUnreadable(f"no verdict and nothing to look at: {_tail(first)}")
        return verdict, f"decided without looking: {_tail(first)}"

    answers = look_fn(build_look(questions, len(frames)), frames)
    final = ask_fn(build_decide(task, actions, answers))
    verdict = parse_status(final)
    if verdict is None:
        raise JudgeUnreadable(f"no verdict after looking: {_tail(final)}")
    return verdict, f"decided after looking: {_tail(final)}"


def _tail(text: str, n: int = 200) -> str:
    return " ".join(str(text or "").split())[-n:]



def build_ask_or_decide_with_intent(
    task: str, actions: list[dict], curated_row: dict, final_url: str = ""
) -> str:
    """V2 prompt entry point; raw task remains distinct from authoritative intent."""
    from evaluation.score_rollout_v2 import build_prompt, curated_intent
    return build_prompt("ask", task, actions, curated_intent(curated_row), final_url)


def build_decide_with_intent(
    task: str, actions: list[dict], curated_row: dict, answers: str, final_url: str = ""
) -> str:
    from evaluation.score_rollout_v2 import build_prompt, curated_intent
    return build_prompt("decide", task, actions, curated_intent(curated_row), final_url, answers)
