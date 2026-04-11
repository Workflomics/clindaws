% OUTPUT PRODUCTION — output port selection and dimension assignment
% Docs: DOCS/encoding/output_production.md

% --- Output selection: exactly one port per tool_output group ---
1 { occurs(T, output(Tool, TOut_ID, Port)) : output_port(TOut_ID, Port) } 1 :-
    occurs(T, run(Tool)),
    tool_has_output(Tool),
    tool_output(Tool, TOut_ID).

% --- Output availability ---
holds(T, avail(out(T, Tool, Port))) :-
    occurs(T, output(Tool, _, Port)).

% --- Output dimensions ---
% Case A: terminal output value — taken directly from port definition
holds(T, dim(out(T, Tool, Port), V, Cat)) :-
    occurs(T, output(Tool, _, Port)),
    dimension(Port, (V, Cat)),
    is_terminal(V).

% Case B: non-terminal output value — planner picks one terminal descendant
1 { occurs(T, output_dim(Tool, Port, Descendant, Cat)) :
    compatible(Descendant, V),
    is_terminal(Descendant) } 1 :-
    occurs(T, output(Tool, _, Port)),
    dimension(Port, (V, Cat)),
    not is_terminal(V).

holds(T, dim(out(T, Tool, Port), Descendant, Cat)) :-
    occurs(T, output_dim(Tool, Port, Descendant, Cat)).
