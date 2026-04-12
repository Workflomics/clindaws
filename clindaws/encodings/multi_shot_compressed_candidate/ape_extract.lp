#program base.
#show tool_at_time/2.
#show ape_bind/3.
#show ape_output/1.
#show ape_goal_out/3.

#program step_initial(t).

#project tool_at_time/2.
tool_at_time(t, Tool) :- occurs(t, run(Tool)).
ape_bind(t, Port, WF) :- occurs(t, bind(_, Port, WF)).
ape_output(out(t, Tool, Port)) :- occurs(t, output(Tool, _, Port)).
ape_goal_out(t, GoalID, WF) :- chosen_goal_output(t, GoalID, WF).

#program step(t).

#project tool_at_time/2.
tool_at_time(t, Tool) :- occurs(t, run(Tool)).
ape_bind(t, Port, WF) :- occurs(t, bind(_, Port, WF)).
ape_output(out(t, Tool, Port)) :- occurs(t, output(Tool, _, Port)).
ape_goal_out(t, GoalID, WF) :- chosen_goal_output(t, GoalID, WF).
