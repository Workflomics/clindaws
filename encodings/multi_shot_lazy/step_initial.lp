#program step_initial(t).

time(t).

holds(t, avail(WF)) :- holds(t-1, avail(WF)).
holds(t, dim(WF, V, Category)) :- holds(t-1, dim(WF, V, Category)), holds(t, avail(WF)).

initial_goal_filter_active(t) :-
    t = 1,
    query(t),
    exact_horizon_mode,
    not initial_goal_holds.

1 { occurs(t, run(Tool)) : lazy_initial_goal_tool(Tool) } 1 :-
    initial_goal_filter_active(t).

1 { occurs(t, run(Tool)) : tool(Tool) } 1 :-
    not initial_goal_filter_active(t).
:- initial_goal_filter_active(t), not occurs(t, run(_)).

1 { occurs(t, use_lazy_candidate(Tool, Candidate)) :
    lazy_tool_candidate(Candidate, Tool),
    lazy_initial_goal_candidate(Candidate) } 1 :-
    initial_goal_filter_active(t),
    occurs(t, run(Tool)).

1 { occurs(t, use_lazy_candidate(Tool, Candidate)) : lazy_tool_candidate(Candidate, Tool) } 1 :-
    not initial_goal_filter_active(t),
    occurs(t, run(Tool)).

occurs(t, run(Tool)) :- occurs(t, use_lazy_candidate(Tool, _)).

1 { occurs(t, bind(Tool, Port, WF)) : lazy_initial_bindable(Candidate, Port, WF) } 1 :-
    occurs(t, use_lazy_candidate(Tool, Candidate)),
    lazy_candidate_input_port(Candidate, Port).

artifact_parent(out(t, Tool, PortOut), WF) :-
    occurs(t, output(Tool, _, PortOut)),
    occurs(t, bind(Tool, _, WF)).

artifact_ancestor(t, Artifact, Ancestor) :-
    artifact_parent(Artifact, Ancestor).

artifact_ancestor(t, Artifact, Ancestor) :-
    artifact_parent(Artifact, Mid),
    artifact_ancestor(t, Mid, Ancestor).

:- occurs(t, use_lazy_candidate(Tool, Candidate)),
   occurs(t, bind(Tool, Port1, out(T1, Tool1, _))),
   occurs(t, bind(Tool, Port2, out(T2, Tool2, _))),
   Port1 < Port2,
   same_candidate_input_signature(Candidate, Port1, Candidate, Port2),
   T1 < T2.

paired_symmetric_producers(t, T1, Tool1, Candidate1, Port1, WF1, T2, Tool2, Candidate2, Port2, WF2) :-
    T1 < T2,
    occurs(T1, use_lazy_candidate(Tool1, Candidate1)),
    occurs(T2, use_lazy_candidate(Tool2, Candidate2)),
    candidate_single_input(Candidate1),
    candidate_single_input(Candidate2),
    occurs(T1, bind(Tool1, Port1, WF1)),
    occurs(T2, bind(Tool2, Port2, WF2)),
    same_candidate_input_signature(Candidate1, Port1, Candidate2, Port2),
    occurs(TJ, use_lazy_candidate(JoinTool, JoinCandidate)),
    occurs(TJ, bind(JoinTool, JoinPort1, out(T1, Tool1, _))),
    occurs(TJ, bind(JoinTool, JoinPort2, out(T2, Tool2, _))),
    TJ > T2,
    JoinPort2 < JoinPort1,
    same_candidate_input_signature(JoinCandidate, JoinPort1, JoinCandidate, JoinPort2).

:- paired_symmetric_producers(t, T1, Tool1, Candidate1, Port1, WF, T2, Tool2, Candidate2, Port2, WF).

:- paired_symmetric_producers(t, T1, Tool1, Candidate1, Port1, WF1, T2, Tool2, Candidate2, Port2, WF2),
   artifact_ancestor(t, WF1, WF2).

occurs(t, output(Tool, Candidate, Port)) :-
    occurs(t, use_lazy_candidate(Tool, Candidate)),
    lazy_candidate_output_port(Candidate, Port).

holds(t, avail(out(t, Tool, Port))) :-
    occurs(t, output(Tool, _, Port)).

1 { occurs(t, output_dim(Tool, Port, V, Category)) :
    lazy_initial_goal_output_value(Candidate, Port, V, Category) } 1 :-
    occurs(t, output(Tool, Candidate, Port)),
    lazy_initial_goal_output_value(Candidate, Port, _, Category).

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

:- occurs(t, run(_)), holds(t-1, goal).

:- occurs(t, run(Tool)), not reachable(t, Tool).
