% GOAL — checks whether the pipeline satisfies all declared goal dimensions
% Docs: DOCS/encoding/goal.md

% --- Dimension matching ---
goal_dim_match(GoalID, WF, GoalV, Cat) :-
    time(T),
    goal_output(GoalID, GoalV, Cat),
    holds(T, avail(WF)),
    holds(T, dim(WF, ActualV, Cat)),
    compatible(ActualV, GoalV).

% --- Missing dimension detection ---
goal_dim_missing(GoalID, WF) :-
    time(T),
    goal_output(GoalID, GoalV, Cat),
    holds(T, avail(WF)),
    not goal_dim_match(GoalID, WF, GoalV, Cat).

% --- Goal satisfaction ---
goal_satisfied_by(GoalID, WF) :-
    time(T),
    holds(T, avail(WF)),
    goal_output(GoalID, _, _),
    goal_dim_match(GoalID, WF, _, _),
    not goal_dim_missing(GoalID, WF).

% --- Goal holds at T when all GoalIDs are satisfied ---
holds(T, goal) :-
    time(T),
    holds(T, avail(WF)),
    goal_satisfied_by(GoalID, WF),
    goal_output(GoalID, _, _),
    holds(T, avail(WF)) : goal_output(GoalID, _, _).

% --- Goal timing: first timestep goal is reached ---
holds(T, goal_time(T)) :-
    holds(T, goal),
    not holds(T-1, goal).

% --- Hard constraint: goal must be reached ---
:- not holds(_, goal).
