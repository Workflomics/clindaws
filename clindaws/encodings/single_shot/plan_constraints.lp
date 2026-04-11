% PLAN LENGTH CONSTRAINTS
% Activation: enable_shortest_plan. in user_constraints.lp
% Docs: DOCS/constraints/01_plan_length.md

:- enable_shortest_plan,
   occurs(T, run(_)),
   holds(T2, goal_time(T2)),
   T > T2.
