#program base.
#show tool_at_time/2.
#show ape_bind/3.
#show ape_holds_dim/3.
#show ape_goal_out/3.

#program step(t).

tool_at_time(t, Tool) :- occurs(t, run(Tool)).
ape_bind(t, Port, WF) :- occurs(t, bind(_, Port, WF)).
ape_holds_dim(out(t, Tool, Port), V, Category) :-
    occurs(t, output(Tool, _, Port)),
    holds(t, dim(out(t, Tool, Port), V, Category)).
ape_goal_out(t, GoalID, WF) :- goal_satisfied_by(t, GoalID, WF).
