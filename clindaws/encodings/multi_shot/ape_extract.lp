#program step(t).

% ================================================================
% APE EXTRACTION — per-timestep atoms read back by SolutionWorkflow
% ================================================================
% Predicate-based #show directives work correctly across incremental
% groundings: re-registration at each step(t) invocation is idempotent.

% Which tool runs at this step.
#project tool_at_time/2.
tool_at_time(t, Tool) :- occurs(t, run(Tool)).
#show tool_at_time/2.

% Input bindings at this step: (step, port-id, workflow-artifact).
ape_bind(t, Port, WF) :-
    occurs(t, bind(_, Port, WF)),
    not workflow_input_slot(WF, _, _).
ape_bind(t, Port, WF) :-
    occurs(t, bind(_, Port, Slot)),
    workflow_input_slot_source(Slot, WF).
#show ape_bind/3.

% Produced outputs created at THIS step only.
ape_output(out(t, Tool, Port)) :- occurs(t, output(Tool, _, Port)).
#show ape_output/1.

% Workflow outputs are emitted from the chosen goal-satisfying artifacts only.
ape_goal_out(t, GoalID, WF) :- chosen_goal_output(t, GoalID, WF).
#show ape_goal_out/3.
