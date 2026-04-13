#program check_usage(t).
#external query(t).

% OUTPUT USAGE / USEFULNESS (optimized multi-shot)
% The optimized backend keeps these checks separate from main goal validation so
% usefulness pruning and output-consumption constraints can run against the
% smaller compressed candidate/output surface.

useful_check(t, T, Tool) :-
   query(t),
   occurs(T, run(Tool)),
   occurs(T, output(Tool, _, Port)),
   chosen_goal_output(t, _, out(T, Tool, Port)).

useful_check(t, T1, Tool1) :-
   query(t),
   occurs(T1, run(Tool1)),
   output_used_by(T1, Tool1, Port, T2, Tool2, Port2),
   useful_check(t, T2, Tool2),
   T2 > T1.

useful_check(t, T, Tool) :-
   query(t),
   occurs(T, run(Tool)),
   not tool_has_output(Tool).

:- query(t),
   enable_usefulness_pruning,
   occurs(T, run(Tool)),
   not useful_check(t, T, Tool).
% SAT permits reusing outputs from earlier runs of the same tool, so do not
% forbid these chains here.

:- query(t),
   enable_all_outputs_consumed,
   occurs(T, output(Tool, Candidate, Port)),
   dynamic_candidate_output_multiplicity(Candidate, Port, RequiredUses),
   RequiredUses > #count {
       goal(GoalID) : chosen_goal_output(t, GoalID, out(T, Tool, Port));
       bind(T2, Tool2, Port2) : output_used_by(T, Tool, Port, T2, Tool2, Port2)
    }.

:- query(t),
   enable_primary_output_consumed,
   occurs(T, run(Tool)),
   tool_has_output(Tool),
   #count { Port, T2, Tool2, Port2 :
       occurs(T, output(Tool, _, Port)),
       output_used_by(T, Tool, Port, T2, Tool2, Port2) } = 0,
   #count { Port : occurs(T, output(Tool, _, Port)),
                   chosen_goal_output(t, _, out(T, Tool, Port)) } = 0.
