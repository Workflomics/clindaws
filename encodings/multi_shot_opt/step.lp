#program step(t).

time(t).

holds(t, avail(WF)) :- holds(t-1, avail(WF)).
holds(t, dim(WF, V, Cat)) :- holds(t-1, dim(WF, V, Cat)), holds(t, avail(WF)).

port_cat_ok(t, Port, WF, Cat) :-
    holds(t, avail(WF)),
    port_requires(Port, ReqV, Cat),
    holds(t, dim(WF, V, Cat)),
    compatible(ReqV, V).

eligible(t, Port, WF) :-
    holds(t, avail(WF)),
    input_port(_, Port),
    port_cat_ok(t, Port, WF, Cat) : port_category(Port, Cat).

% SINGLE aggregate choice — replaces N individual { occurs(t, run(Tool)) } rules.
% Grounds to one rule per step regardless of tool count.
1 { occurs(t, run(Tool)) : tool(Tool) } 1.

1 { occurs(t, use_variant(Tool, TI_ID)) : tool_input_variant(Tool, TI_ID) } 1 :-
    occurs(t, run(Tool)),
    tool_input_variant(Tool, _).

occurs(t, run(Tool)) :- occurs(t, use_variant(Tool, _)).

1 { occurs(t, bind(Tool, Port, WF)) : eligible(t-1, Port, WF) } 1 :-
    occurs(t, use_variant(Tool, TI_ID)),
    input_port(TI_ID, Port).

occurs(t, output(Tool, TI_ID, Port)) :-
    occurs(t, use_variant(Tool, TI_ID)),
    candidate_output_port(TI_ID, Port).

holds(t, avail(out(t, Tool, Port))) :-
    occurs(t, output(Tool, _, Port)).

holds(t, dim(out(t, Tool, Port), V, Cat)) :-
    occurs(t, output(Tool, TI_ID, Port)),
    candidate_output_value(TI_ID, Port, V, Cat).

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

goal_dim_match(t, GoalID, WF, GoalV, Cat) :-
    goal_output(GoalID, GoalV, Cat),
    holds(t, avail(WF)),
    holds(t, dim(WF, ActualV, Cat)),
    compatible(ActualV, GoalV).

goal_dim_missing(t, GoalID, WF) :-
    goal_output(GoalID, GoalV, Cat),
    holds(t, avail(WF)),
    not goal_dim_match(t, GoalID, WF, GoalV, Cat).

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

:- t > 1, occurs(t, run(_)), not occurs(t-1, run(_)).
:- occurs(t, run(_)), holds(t-1, goal).

:- occurs(T1, run(Tool)), occurs(t, run(Tool)), T1 < t,
   occurs(T1, bind(Tool, Port, WF)),
   occurs(t, bind(Tool, Port, WF)).

:- occurs(t, run(Tool)), not multi_run(Tool), occurs(T1, run(Tool)), T1 < t.
