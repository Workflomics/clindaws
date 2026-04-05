% USEFULNESS — prunes tool runs not contributing to the goal
% Docs: DOCS/encoding/usefulness.md

% --- Direct goal usefulness ---
useful(T, Tool) :-
    occurs(T, run(Tool)),
    occurs(T, output(Tool, _, Port)),
    goal_satisfied_by(T, _, out(T, Tool, Port)).

% --- Indirect usefulness: backward propagation from goal ---
useful(T1, Tool1) :-
    occurs(T1, run(Tool1)),
    occurs(T1, output(Tool1, _, Port)),
    occurs(T2, bind(Tool2, _, out(T1, Tool1, Port))),
    useful(T2, Tool2),
    T2 > T1.

% --- Trivial usefulness: tools with no output ---
useful(T, Tool) :-
    occurs(T, run(Tool)),
    not tool_has_output(Tool).

% --- Constraint: every tool that runs must be useful ---
:- enable_usefulness_pruning,
   occurs(T, run(Tool)),
   not useful(T, Tool).

% SAT-aligned behavior permits reusing outputs from earlier runs of the same
% tool when the resulting workflow remains useful.
