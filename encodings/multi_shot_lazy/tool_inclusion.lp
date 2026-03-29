#program check(t).

:- query(t),
   must_use(Tool),
   not tool_abstract_class(Tool),
   #count { T : occurs(T, run(Tool)) } = 0.

:- query(t),
   must_not_use(Tool),
   not tool_abstract_class(Tool),
   occurs(_, run(Tool)).

:- query(t),
   must_use(Class),
   tool_abstract_class(Class),
   #count { Tool : tool_is_a(Tool, Class), occurs(_, run(Tool)) } = 0.

:- query(t),
   must_not_use(Class),
   tool_abstract_class(Class),
   tool_is_a(Tool, Class),
   tool(Tool),
   occurs(_, run(Tool)).
