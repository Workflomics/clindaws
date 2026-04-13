#program step_query(t).
#external query(t).

% STEP_QUERY — exact-horizon query-time output filtering
% When exact horizon mode is active, the current horizon can restrict itself to
% the subset of tools/candidates/output values that can still satisfy the goal.

query_output_filter_active(t) :-
    query(t),
    exact_horizon_mode.

:- query_output_filter_active(t), occurs(t, run(Tool)), not dynamic_query_goal_tool(Tool).
:- query_output_filter_active(t), occurs(t, use_dynamic_candidate(_, Candidate)), not dynamic_query_goal_candidate(Candidate).

1 { occurs(t, output_dim(Tool, Port, V, Category)) :
    dynamic_query_output_value(Candidate, Port, V, Category) } 1 :-
    query_output_filter_active(t),
    occurs(t, output(Tool, Candidate, Port)),
    dynamic_query_output_value(Candidate, Port, _, Category).
