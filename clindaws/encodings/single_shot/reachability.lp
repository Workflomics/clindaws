% REACHABILITY — ensures every tool run is causally connected to initial inputs
% Docs: DOCS/encoding/reachability.md

% --- Output consumption detection ---
consumes_output(T, Tool) :-
    occurs(T, bind(Tool, _, out(_, _, _))).

% --- Root detection ---
% Case A: no-input tools are always roots
is_root(T, Tool) :-
    occurs(T, run(Tool)),
    not tool_has_input(Tool).

% Case B: has inputs but none are tool outputs
is_root(T, Tool) :-
    occurs(T, run(Tool)),
    tool_has_input(Tool),
    not consumes_output(T, Tool).

% --- Reachability propagation ---
reachable(T, Tool) :- is_root(T, Tool).

reachable(T2, Tool2) :-
    reachable(T1, Tool1),
    occurs(T1, output(Tool1, _, Port)),
    occurs(T2, bind(Tool2, _, out(T1, Tool1, Port))),
    T2 > T1.

% --- Constraint: every tool that runs must be reachable ---
:- occurs(T, run(Tool)), not reachable(T, Tool).
