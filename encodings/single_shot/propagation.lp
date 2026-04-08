% PROPAGATION — inertia for workflow item availability and dimensions
% Docs: DOCS/encoding/propagation.md

holds(T, avail(WF)) :-
    holds(T-1, avail(WF)),
    time(T).

holds(T, dim(WF, V, Cat)) :-
    holds(T-1, dim(WF, V, Cat)),
    holds(T, avail(WF)),
    time(T).

holds(T, avail(out(T, Tool, Port))) :-
    occurs(T, output(Tool, _, Port)).

holds(T, dim(out(T, Tool, Port), V, Cat)) :-
    occurs(T, output(Tool, _, Port)),
    dimension(Port, (V, Cat)),
    is_terminal(V).

holds(T, dim(out(T, Tool, Port), Descendant, Cat)) :-
    occurs(T, output_dim(Tool, Port, Descendant, Cat)).
