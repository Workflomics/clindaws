#program check_usage(t).
#external query(t).

useful_check(t, T, Tool) :-
   query(t),
   occurs(T, run(Tool)),
   occurs(T, output(Tool, _, Port)),
   chosen_goal_output(t, _, out(T, Tool, Port)).

useful_check(t, T1, Tool1) :-
   query(t),
   occurs(T1, run(Tool1)),
   occurs(T1, output(Tool1, _, Port)),
   occurs(T2, bind(Tool2, _, out(T1, Tool1, Port))),
   useful_check(t, T2, Tool2),
   T2 > T1.

useful_check(t, T, Tool) :-
   query(t),
   occurs(T, run(Tool)),
   not tool_has_output(Tool).

:- query(t), occurs(T, run(Tool)), not useful_check(t, T, Tool).
% SAT permits reusing outputs from earlier runs of the same tool, so do not
% forbid these chains here.

:- query(t),
   enable_all_outputs_consumed,
   occurs(T, output(Tool, _, Port)),
   not chosen_goal_output(t, _, out(T, Tool, Port)),
   #count { T2, Tool2, Port2 : occurs(T2, bind(Tool2, Port2, out(T, Tool, Port))) } = 0.

:- query(t),
   enable_primary_output_consumed,
   occurs(T, run(Tool)),
   tool_has_output(Tool),
   #count { Port, T2, Tool2, Port2 :
       occurs(T, output(Tool, _, Port)),
       occurs(T2, bind(Tool2, Port2, out(T, Tool, Port))) } = 0,
   #count { Port : occurs(T, output(Tool, _, Port)),
                   chosen_goal_output(t, _, out(T, Tool, Port)) } = 0.
