#program check(t).
#external query(t).

:- query(t), not holds(t, goal).
:- query(t), exact_horizon_mode, t > 1, holds(t-1, goal).

useful_check(t, T, Tool) :-
    query(t),
    occurs(T, run(Tool)),
    occurs(T, output(Tool, _, Port)),
    goal_satisfied_by(t, _, out(T, Tool, Port)).

useful_check(t, T1, Tool1) :-
    query(t),
    occurs(T1, run(Tool1)),
    occurs(T1, output(Tool1, _, Port)),
    occurs(T2, bind(Tool2, _, out(T1, Tool1, Port))),
    useful_check(t, T2, Tool2),
    T2 > T1.

useful_check(t, T, Tool) :-
    query(t),
    occurs(T, run(Tool)),
    not tool_has_output(Tool).

:- query(t), occurs(T, run(Tool)), not useful_check(t, T, Tool).
:- query(t), occurs(T2, bind(Tool, _, out(T1, Tool, _))), T2 > T1.
