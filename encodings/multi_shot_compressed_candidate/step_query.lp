#program step_query(t).
#external query(t).

query_output_filter_active(t) :-
    query(t),
    exact_horizon_mode.

:- query_output_filter_active(t), occurs(t, run(Tool)), not lazy_query_goal_tool(Tool).
:- query_output_filter_active(t), occurs(t, use_lazy_candidate(_, Candidate)), not lazy_query_goal_candidate(Candidate).

1 { occurs(t, output_dim(Tool, Port, V, Category)) :
    lazy_query_output_value(Candidate, Port, V, Category) } 1 :-
    query_output_filter_active(t),
    occurs(t, output(Tool, Candidate, Port)),
    lazy_query_output_value(Candidate, Port, _, Category).
