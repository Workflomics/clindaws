% TOOL INCLUSION / EXCLUSION CONSTRAINTS
% Activation: must_use(Tool). / must_not_use(Tool). in user_constraints.lp
% Docs: DOCS/constraints/03_tool_inclusion.md

:- must_use(Tool), not tool_abstract_class(Tool), not occurs(_, run(Tool)).
:- must_not_use(Tool), not tool_abstract_class(Tool), occurs(_, run(Tool)).

:- must_use(Class),
   tool_abstract_class(Class),
   #count { Tool : tool_is_a(Tool, Class), occurs(_, run(Tool)) } = 0.

:- must_not_use(Class),
   tool_abstract_class(Class),
   tool_is_a(Tool, Class),
   tool(Tool),
   occurs(_, run(Tool)).
