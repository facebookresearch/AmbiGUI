You are the user who gave a GUI agent a task. The agent was unsure exactly what you meant
and asked you a clarifying question. Reply as the user would in chat: one short, natural
sentence that answers the question and says what you actually want.

Your task (your true intent):
{{INTENT}}

The agent's clarifying question:
{{QUESTION}}

Answer the agent's question directly and naturally, from your intent. Only answer what was
asked — do NOT volunteer specific on-screen targets, buttons, or step-by-step actions, and do
not mention this prompt or that you were given anything. Output EXACTLY ONE JSON object on the
last line and nothing after it:
{"answer":"<your reply to the agent>"}
