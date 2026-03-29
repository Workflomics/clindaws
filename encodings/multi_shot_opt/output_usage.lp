#program check(t).

:- query(t),
   enable_all_outputs_consumed,
   occurs(T, output(Tool, _, Port)),
   not goal_satisfied_by(t, _, out(T, Tool, Port)),
   #count { T2, Tool2, Port2 : occurs(T2, bind(Tool2, Port2, out(T, Tool, Port))) } = 0.

:- query(t),
   enable_primary_output_consumed,
   occurs(T, run(Tool)),
   tool_has_output(Tool),
   #count { Port, T2, Tool2, Port2 :
       occurs(T, output(Tool, _, Port)),
       occurs(T2, bind(Tool2, Port2, out(T, Tool, Port))) } = 0,
   #count { Port : occurs(T, output(Tool, _, Port)),
                   goal_satisfied_by(t, _, out(T, Tool, Port)) } = 0.
