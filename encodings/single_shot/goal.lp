% GOAL — checks whether the pipeline satisfies all declared goal dimensions
% Docs: DOCS/encoding/goal.md

% --- Dimension matching at timestep T ---
goal_dim_match(T, GoalID, WF, GoalV, Cat) :-
    goal_output(GoalID, GoalV, Cat),
    holds(T, avail(WF)),
    holds(T, dim(WF, ActualV, Cat)),
    compatible(ActualV, GoalV).

% --- Missing dimension detection at timestep T ---
goal_dim_missing(T, GoalID, WF) :-
    goal_output(GoalID, GoalV, Cat),
    holds(T, avail(WF)),
    not goal_dim_match(T, GoalID, WF, GoalV, Cat).

% --- Goal satisfaction at timestep T ---
goal_satisfied_by(T, GoalID, WF) :-
    holds(T, avail(WF)),
    goal_output(GoalID, _, _),
    goal_dim_match(T, GoalID, WF, _, _),
    not goal_dim_missing(T, GoalID, WF).

% Direct modes should only count produced outputs toward the workflow goal.
goal_satisfied_by_output(T, GoalID, WF) :-
    goal_satisfied_by(T, GoalID, WF),
    not holds(0, avail(WF)).

goal_satisfied(T, GoalID) :-
    goal_output(GoalID, _, _),
    goal_satisfied_by_output(T, GoalID, WF).

% --- Goal holds at T when all GoalIDs are satisfied ---
holds(T, goal) :-
    time(T),
    goal_satisfied(T, GoalID) : goal_output(GoalID, _, _).

% Goal inertia: once reached, it persists.
holds(T, goal) :-
    time(T),
    T > 1,
    holds(T-1, goal).

% --- Goal timing: first timestep goal is reached ---
holds(T, goal_time(T)) :-
    holds(T, goal),
    not holds(T-1, goal).

% --- Output-dimension canonicalization ---
output_dim_violates_consumer(TProd, ToolProd, Port, V, Cat) :-
    occurs(TProd, output(ToolProd, _, Port)),
    dimension(Port, (BaseV, Cat)),
    not is_terminal(BaseV),
    compatible(V, BaseV),
    is_terminal(V),
    occurs(TUse, bind(ToolUse, PortIn, out(TProd, ToolProd, Port))),
    port_requires(PortIn, ReqV, Cat),
    not compatible(V, ReqV).

output_dim_violates_goal(TProd, ToolProd, Port, V, Cat) :-
    occurs(TProd, output(ToolProd, _, Port)),
    dimension(Port, (BaseV, Cat)),
    not is_terminal(BaseV),
    compatible(V, BaseV),
    is_terminal(V),
    holds(TGoal, goal_time(TGoal)),
    goal_satisfied_by_output(TGoal, GoalID, out(TProd, ToolProd, Port)),
    goal_output(GoalID, GoalV, Cat),
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

% --- Hard constraint: goal must be reached ---
:- not holds(_, goal).
