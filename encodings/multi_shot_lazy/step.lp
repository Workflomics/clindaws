#program step(t).

time(t).

holds(t, avail(WF)) :- holds(t-1, avail(WF)).
holds(t, dim(WF, V, Category)) :- holds(t-1, dim(WF, V, Category)), holds(t, avail(WF)).

lazy_port_category_match(t, Candidate, Port, WF, Category) :-
    holds(t, avail(WF)),
    lazy_candidate_input_value(Candidate, Port, ReqV, Category),
    holds(t, dim(WF, ReqV, Category)).

eligible(t, Candidate, Port, WF) :-
    holds(t, avail(WF)),
    lazy_candidate_input_port(Candidate, Port),
    lazy_port_category_match(t, Candidate, Port, WF, Category) :
        lazy_candidate_input_category(Candidate, Port, Category).

1 { occurs(t, run(Tool)) : tool(Tool) } 1.

1 { occurs(t, use_lazy_candidate(Tool, Candidate)) : lazy_tool_candidate(Candidate, Tool) } 1 :-
    occurs(t, run(Tool)).

occurs(t, run(Tool)) :- occurs(t, use_lazy_candidate(Tool, _)).

1 { occurs(t, bind(Tool, Port, WF)) : eligible(t-1, Candidate, Port, WF) } 1 :-
    occurs(t, use_lazy_candidate(Tool, Candidate)),
    lazy_candidate_input_port(Candidate, Port).

occurs(t, output(Tool, Candidate, Port)) :-
    occurs(t, use_lazy_candidate(Tool, Candidate)),
    lazy_candidate_output_port(Candidate, Port).

holds(t, avail(out(t, Tool, Port))) :-
    occurs(t, output(Tool, _, Port)).

1 { occurs(t, output_dim(Tool, Port, V, Category)) :
    lazy_candidate_output_choice_value(Candidate, Port, V, Category) } 1 :-
    occurs(t, output(Tool, Candidate, Port)),
    lazy_candidate_output_choice_value(Candidate, Port, _, Category).

holds(t, dim(out(t, Tool, Port), V, Category)) :-
    occurs(t, output_dim(Tool, Port, V, Category)).

holds(t, dim(out(t, Tool, Port), V, Category)) :-
    occurs(t, output(Tool, Candidate, Port)),
    lazy_candidate_output_singleton(Candidate, Port, V, Category).

consumes_output(t, Tool) :-
    occurs(t, bind(Tool, _, out(_, _, _))).

is_root(t, Tool) :-
    occurs(t, run(Tool)),
    not tool_has_input(Tool).

is_root(t, Tool) :-
    occurs(t, run(Tool)),
    tool_has_input(Tool),
    not consumes_output(t, Tool).

reachable(t, Tool) :- is_root(t, Tool).

reachable(t, Tool2) :-
    reachable(T1, Tool1),
    occurs(T1, output(Tool1, _, Port)),
    occurs(t, bind(Tool2, _, out(T1, Tool1, Port))),
    t > T1.

:- occurs(t, run(Tool)), not reachable(t, Tool).

goal_dim_match(t, GoalID, WF, GoalV, Category) :-
    goal_output(GoalID, GoalV, Category),
    holds(t, avail(WF)),
    holds(t, dim(WF, ActualV, Category)),
    compatible(ActualV, GoalV).

goal_dim_missing(t, GoalID, WF) :-
    goal_output(GoalID, GoalV, Category),
    holds(t, avail(WF)),
    not goal_dim_match(t, GoalID, WF, GoalV, Category).

goal_satisfied_by(t, GoalID, WF) :-
    holds(t, avail(WF)),
    goal_output(GoalID, _, _),
    goal_dim_match(t, GoalID, WF, _, _),
    not goal_dim_missing(t, GoalID, WF).

goal_satisfied(t, GoalID) :-
    goal_output(GoalID, _, _),
    goal_satisfied_by(t, GoalID, WF).

holds(t, goal) :-
    goal_satisfied(t, GoalID) : goal_output(GoalID, _, _).

holds(t, goal) :- holds(t-1, goal).

:- requirement_filtering_active, occurs(t, run(Tool)), not tool_req_compatible(Tool).
:- requirement_filtering_active, occurs(t, bind(Tool, Port, _)), not port_env_compatible(Tool, Port).

:- t > 1, occurs(t, run(_)), not occurs(t-1, run(_)).
:- occurs(t, run(_)), holds(t-1, goal).

:- occurs(T1, run(Tool)), occurs(t, run(Tool)), T1 < t,
   occurs(T1, bind(Tool, Port, WF)),
   occurs(t, bind(Tool, Port, WF)).

:- occurs(t, run(Tool)), not multi_run(Tool), occurs(T1, run(Tool)), T1 < t.
