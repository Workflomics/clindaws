#program base.

% ================================================================
% APE EXTRACTION — symbols read back by Python reconstruction
% ================================================================
% The optimized backend still exposes the same external workflow surface as the
% direct backend: tool runs, input bindings, produced outputs, and chosen goal
% outputs. The only difference is that candidate ids stay hidden and are used
% only internally by the optimized rules.

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
