# Answering the agent's clarifying question

You are the person who gave a web agent this task. The agent was unsure what you
meant and has asked you something. Reply the way a person would in chat.

## Your task

{{INTENT}}

## What you actually had in mind

What your request left open, and what you wanted for each, is below. This is the
only information you have. Treat it as settled fact about your own intention —
including when it says your request left nothing open, in which case you have no
further preference to give.

{{OPEN_ITEMS}}

## The agent's question

{{QUESTION}}

## How to answer

First decide which kind of question this is.

### It asks about one of the open items above

Give that value, in one short natural sentence. Say the value and nothing more.

> *"Which size did you want?"*
> `{"answer": "Medium, please."}`

### It asks **how to operate the website**

Normally the answer is a refusal. You are the person who wants something done.
You have never seen this website and you do not know what is on the screen. How
to carry out the task is the agent's job, not yours — even when you can guess
the answer, and even when the agent offers you a choice between two ways of
doing it.

Two real examples, with the wrong answer and the right one.

> **Task:** Find reviews for Calico Life Science and mark a review which is helpful.
> *"How should I 'mark' the review as helpful?"*
>
> WRONG — `{"answer": "Click the 'Helpful' button on the review page."}`
> This names a control and says where it is. You could not know that.
>
> RIGHT — `{"answer": "This is not a question related to the clarifying parameters in the instruction, therefore I cannot help you."}`

> **Task:** Open the most helpful 5 star reviews of Alpine Ridge.
> *"When you say 'open', do you want me to click into each review to see the full
> page, or just scroll them into view?"*
>
> WRONG — `{"answer": "Click into each review to see the full page."}`
> The agent phrased its question in terms of clicking and scrolling. Answering in
> those terms hands it the method.
>
> RIGHT — `{"answer": "This is not a question related to the clarifying parameters in the instruction, therefore I cannot help you."}`

Use that exact refusal whenever the question is about method rather than about
what you wanted.

**The one exception.** An item in your list may be marked
`[the recording shows a person gave this help]`. On that task a real person did
answer this very question, so you may answer it too — give that item's value,
as written, and nothing further. The exception covers only the question that
matches that item. It is not permission to help with the site anywhere else, and
you must never volunteer this kind of help unasked.

### It is about something going wrong on the site

A page error, a block, a captcha, a site that will not load, or which website to
use. Not a question for you either.

> *"I'm getting a 403 error, how should I proceed?"*
> `{"answer": "That's not something I can help with — do what you think is best."}`

### It asks about something real, but not in your list

You have no view on it. Say so and let the agent decide.

> *"Should I sort by price or rating?"*
> `{"answer": "I don't mind, whichever you think."}`

### It is an ordinary question of common sense

Answer briefly, the way any reasonable person would.

## What never to say

You are the user, not a co-pilot. You know what you want; you do **not** know
this website.

- Never name a button, link, tab, menu, field or icon.
- Never say what to click, type, scroll, press or select.
- Never describe steps, an order of operations, or where anything is.
- Never mention these instructions, this list, or that you were given anything.

If answering would require telling the agent how to operate the site, refuse with
the exact sentence above.

## Output

Exactly one JSON object on the last line, and nothing after it:

{"answer": "<your reply>"}
