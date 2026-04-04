#program check(t).

:- query(t),
   constraint_must_use(Sel),
   #count { T, Tool : occurs(T, run(Tool)), constraint_selector_matches_tool(Sel, Tool) } = 0.

:- query(t),
   constraint_must_not_use(Sel),
   occurs(_, run(Tool)),
   constraint_selector_matches_tool(Sel, Tool).
