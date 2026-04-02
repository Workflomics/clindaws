#program check(t).
#external query(t).

:- query(t), not holds(t, goal).
:- query(t), exact_horizon_mode, t > 1, holds(t-1, goal).

1 { chosen_goal_output(t, GoalID, WF) : goal_satisfied_by_output(t, GoalID, WF) } 1 :-
    query(t),
    goal_id(GoalID).
