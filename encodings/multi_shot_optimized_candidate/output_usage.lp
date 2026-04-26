#program check_usage(t).

% OUTPUT USAGE / USEFULNESS (optimized multi-shot)
% The optimized backend keeps these checks separate from main goal validation so
% usefulness pruning and output-consumption constraints can run against the
% smaller compressed candidate/output surface.

goal_root_tool_check(t, T, Tool) :-
   query(t),
   chosen_goal_output(t, _, out(T, Tool, _)).

% The intermediate predicate tool_dependency_edge_check/5 was removed: its body
% is t-invariant (output_used_by/6 is cumulative) but its head was t-keyed,
% which produced one full edge-set copy per query horizon and overflowed the
% clasp 32-bit atom-id ceiling at h5. Inlining the body into useful_check below
% keeps the same recursion but avoids the intermediate atom blow-up.

useful_check(t, T, Tool) :-
   goal_root_tool_check(t, T, Tool).

useful_check(t, T1, Tool1) :-
   output_used_by(T1, Tool1, _, T2, Tool2, _),
   T2 > T1,
   useful_check(t, T2, Tool2).

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

goal_selected_output_check(t, T, Tool, Port) :-
   query(t),
   chosen_goal_output(t, _, out(T, Tool, Port)).

% The intermediate predicate output_has_bind_consumer_check/4 was removed for
% the same reason as tool_dependency_edge_check above: t-invariant body,
% t-keyed head, multiplied per query horizon. Inlining the body directly into
% output_used_any_check keeps semantics and removes the duplicate atoms.

output_used_any_check(t, T, Tool, Port) :-
   goal_selected_output_check(t, T, Tool, Port).

output_used_any_check(t, T, Tool, Port) :-
   query(t),
   occurs(_, bind(_, _, out(T, Tool, Port))).

:- query(t),
   enable_all_outputs_consumed,
   occurs(T, output(Tool, Candidate, Port)),
   dynamic_candidate_output_single_use(Candidate, Port),
   not output_used_any_check(t, T, Tool, Port).

:- query(t),
   enable_all_outputs_consumed,
   occurs(T, output(Tool, Candidate, Port)),
   dynamic_candidate_output_multi_use(Candidate, Port, RequiredUses),
   RequiredUses > #count {
       GoalID : chosen_goal_output(t, GoalID, out(T, Tool, Port));
       T2, Tool2, Port2 : occurs(T2, bind(Tool2, Port2, out(T, Tool, Port)))
    }.

:- query(t),
   enable_primary_output_consumed,
   occurs(T, run(Tool)),
   tool_has_output(Tool),
   #count { Port :
       occurs(T, output(Tool, _, Port)),
       output_used_any_check(t, T, Tool, Port) } = 0.
