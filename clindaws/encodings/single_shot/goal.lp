% GOAL — goal satisfaction check and output-dimension canonicalization
% Docs: DOCS/encoding/goal.md

% --- Goal holds at T when all GoalIDs are satisfied ---
holds(T, goal) :-
    time(T),
    goal_satisfied(T, GoalID) : goal_output(GoalID, _, _).

% Goal inertia: once reached, it persists.
holds(T, goal) :-
    time(T),
    T > 1,
    holds(T-1, goal).

% First time goal is reached.
holds(T, goal_time(T)) :-
    holds(T, goal),
    not holds(T-1, goal).

% A specific goal requirement is satisfied if there's an available item that matches.
goal_satisfied(T, GoalID) :-
    time(T),
    goal_output(GoalID, _, _),
    goal_satisfied_by(T, GoalID, _).

% Helper: item satisfies one category of a goal requirement.
goal_dim_match(T, GoalID, WF, Cat) :-
    goal_output(GoalID, ReqV, Cat),
    holds(T, dim(WF, V, Cat)),
    compatible(V, ReqV).

% Item is eligible for a goal if ALL categories match.
goal_satisfied_by(T, GoalID, WF) :-
    goal_output(GoalID, _, _),
    holds(T, avail(WF)),
    goal_dim_match(T, GoalID, WF, Cat) : goal_output(GoalID, _, Cat).

% Tie-breaker for multiple items satisfying the same goal requirement.
% Pick exactly one witness item for the goal at time T.
1 { chosen_goal_output(T, GoalID, WF) : goal_satisfied_by(T, GoalID, WF) } 1 :-
    holds(T, goal_time(T)),
    goal_output(GoalID, _, _).

% --- Output-dimension canonicalization ---
% Keep the smallest terminal value that still satisfies all actual downstream
% consumers and any goal the produced output satisfies at horizon T.

output_dim_violates_consumer(TProd, ToolProd, Port, V, Cat) :-
    occurs(TProd, output(ToolProd, _, Port)),
    dimension(Port, (BaseV, Cat)),
    not is_terminal(BaseV),
    compatible(V, BaseV),
    is_terminal(V),
    occurs(TUse, bind(ToolUse, PortIn, out(TProd, ToolProd, Port))),
    port_requires(PortIn, ReqV, Cat),
    not compatible(V, ReqV).

chosen_output_goal_requirement(TGoal, TProd, ToolProd, Port, GoalID, GoalV, Cat) :-
    holds(TGoal, goal_time(TGoal)),
    chosen_goal_output(TGoal, GoalID, out(TProd, ToolProd, Port)),
    goal_output(GoalID, GoalV, Cat).

output_dim_violates_goal(TProd, ToolProd, Port, V, Cat) :-
    occurs(TProd, output(ToolProd, _, Port)),
    dimension(Port, (BaseV, Cat)),
    not is_terminal(BaseV),
    compatible(V, BaseV),
    is_terminal(V),
    chosen_output_goal_requirement(_, TProd, ToolProd, Port, _, GoalV, Cat),
    not compatible(V, GoalV).

output_dim_supported(TProd, ToolProd, Port, V, Cat) :-
    occurs(TProd, output(ToolProd, _, Port)),
    dimension(Port, (BaseV, Cat)),
    not is_terminal(BaseV),
    compatible(V, BaseV),
    is_terminal(V),
    not output_dim_violates_consumer(TProd, ToolProd, Port, V, Cat),
    not output_dim_violates_goal(TProd, ToolProd, Port, V, Cat).

output_dim_smaller_supported(TProd, ToolProd, Port, V, Cat) :-
    output_dim_supported(TProd, ToolProd, Port, V, Cat),
    output_dim_supported(TProd, ToolProd, Port, V2, Cat),
    V2 < V.

:- occurs(TProd, output_dim(ToolProd, Port, V, Cat)),
   output_dim_supported(TProd, ToolProd, Port, V, Cat),
   output_dim_smaller_supported(TProd, ToolProd, Port, V, Cat).
