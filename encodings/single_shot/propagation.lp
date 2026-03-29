% PROPAGATION — inertia for workflow item availability and dimensions
% Docs: DOCS/encoding/propagation.md

holds(T, avail(WF)) :-
    holds(T-1, avail(WF)),
    time(T).

holds(T, dim(WF, V, Cat)) :-
    holds(T-1, dim(WF, V, Cat)),
    holds(T, avail(WF)),
    time(T).
