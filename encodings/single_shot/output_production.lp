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
precomputed_output_port_has_category(Port, Cat) :-
    precomputed_output_port_candidate(Port, _, Cat).

precomputed_output_port_singleton_terminal(Port, Value, Cat) :-
    precomputed_output_port_candidate(Port, Value, Cat),
    1 = #count {
        CandidateValue :
            precomputed_output_port_candidate(Port, CandidateValue, Cat)
    }.

% Case A: terminal output value — taken directly from port definition
holds(T, dim(out(T, Tool, Port), V, Cat)) :-
    occurs(T, output(Tool, _, Port)),
    dimension(Port, (V, Cat)),
    is_terminal(V).

% Case B: non-terminal output value — planner picks one retained terminal
% representative. Prefer Python-precomputed compressed candidates when they are
% available, otherwise fall back to the dense terminal expansion.
1 { occurs(T, output_dim(Tool, Port, Descendant, Cat)) :
    precomputed_output_port_candidate(Port, Descendant, Cat) } 1 :-
    occurs(T, output(Tool, _, Port)),
    precomputed_output_port_has_category(Port, Cat),
    not precomputed_output_port_singleton_terminal(Port, _, Cat).

1 { occurs(T, output_dim(Tool, Port, Descendant, Cat)) :
    compatible(Descendant, V),
    is_terminal(Descendant) } 1 :-
    occurs(T, output(Tool, _, Port)),
    not precomputed_output_port_has_category(Port, Cat),
    dimension(Port, (V, Cat)),
    not is_terminal(V).

holds(T, dim(out(T, Tool, Port), Descendant, Cat)) :-
    occurs(T, output(Tool, _, Port)),
    precomputed_output_port_singleton_terminal(Port, Descendant, Cat).

holds(T, dim(out(T, Tool, Port), Descendant, Cat)) :-
    occurs(T, output_dim(Tool, Port, Descendant, Cat)).
