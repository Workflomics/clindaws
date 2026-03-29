% PRE-COMPUTE — static ground-time derivations from taxonomy and tool definitions
% Docs: DOCS/encoding/pre_compute.md

% --- Flattened taxonomy ---
ancestor(Child, Parent) :- taxonomy(_, _, (_, Child, Parent)).
ancestor(Child, Ancestor) :- ancestor(Child, Mid), ancestor(Mid, Ancestor).

% Terminal: a value that no other value has as parent
is_terminal(V) :- taxonomy(_, _, (_, V, _)), not taxonomy(_, _, (_, _, V)).

% compatible(V, Ancestor): V is terminal and equal to or descends from Ancestor
compatible(V, V)        :- is_terminal(V).
compatible(V, Ancestor) :- is_terminal(V), ancestor(V, Ancestor).

% --- Tool structure ---
tool_has_input(Tool)  :- tool_input(Tool, _).
tool_has_output(Tool) :- tool_output(Tool, _).

% Input variants per tool
tool_input_variant(Tool, TI_ID) :- tool_input(Tool, TI_ID).

% Static port requirements
port_requires(Port, PortValue, Category) :-
    input_port(_, Port),
    dimension(Port, (PortValue, Category)).
