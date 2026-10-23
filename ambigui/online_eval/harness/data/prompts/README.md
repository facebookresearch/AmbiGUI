Ask-encouragement prompt arms. Each file is the paragraph that replaces the ask guidance in
the policy system prompt (training/cascp_common.CASCP_SYSTEM); everything else in the system
prompt -- the role line, the tool description and the report_infeasible / report_task_complete
paragraph -- is unchanged. ask_p0.txt is the repo's own paragraph verbatim (the baseline).
Each later arm keeps all of the previous arm's text and adds one sentence group:
  p1  availability: the user is there and asking is always allowed
  p2  an explicit check of the instruction for open choices before the first action
  p3  cost framing (a wrong guess fails the task, a question costs one turn) plus two guards:
      do not ask how to operate the site, do not ask when everything is specified
  p4  a procedure (list decisions, mark specified, ask one concise question naming every
      missing detail with options, else act) and send_msg_to_user only for questions
  p5  a firm rule: never commit an unspecified choice; confirm before irreversible steps;
      a guessed preference counts as a mistake
Applied by harness/prompt_variant.py when ASK_PROMPT_VARIANT=pN is set (run_online.sh sets it
for a policy named <model>@pN).
