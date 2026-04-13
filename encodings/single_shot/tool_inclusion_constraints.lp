% TOOL INCLUSION / EXCLUSION CONSTRAINTS
% Activation: constraint_must_use(Sel). / constraint_must_not_use(Sel). in
% translated constraint facts.
% Docs: DOCS/constraints/03_tool_inclusion.md

:- constraint_must_use(Sel),
   #count { T, Tool : occurs(T, run(Tool)), constraint_selector_matches_tool(Sel, Tool) } = 0.

:- constraint_must_not_use(Sel),
   occurs(_, run(Tool)),
   constraint_selector_matches_tool(Sel, Tool).
