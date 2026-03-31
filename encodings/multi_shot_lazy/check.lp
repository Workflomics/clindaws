#program check(t).
#external query(t).

:- query(t), not holds(t, goal).
:- query(t), exact_horizon_mode, t > 1, holds(t-1, goal).

1 { chosen_goal_output(t, GoalID, WF) : goal_satisfied_by(t, GoalID, WF) } 1 :-
    query(t),
    goal_id(GoalID).

% Mirror SAT/APE semantics: workflow inputs cannot be reused as final outputs.
:- query(t), holds(0, avail(WF)), goal_satisfied_by(t, _, WF).
