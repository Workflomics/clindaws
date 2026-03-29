% TOOL REPETITION CONSTRAINTS
% Activation: multi_run(Tool). in user_constraints.lp opts a tool out of this limit
% Docs: DOCS/constraints/06_tool_repetition.md

:- tool(Tool),
   not multi_run(Tool),
   #count { T : occurs(T, run(Tool)) } > 1.
