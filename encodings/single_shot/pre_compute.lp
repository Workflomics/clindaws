% PRE-COMPUTE — static ground-time derivations from taxonomy and tool definitions
% Docs: DOCS/encoding/pre_compute.md

% --- Terminal detection ---
% compatible(V, Ancestor) facts are pre-computed by the Python translator and
% emitted as ground facts, replacing the O(n²) recursive ancestor/2 derivation.
is_terminal(V) :- taxonomy(_, _, (_, V, _)), not taxonomy(_, _, (_, _, V)).

% --- Tool structure ---
tool_has_input(Tool)  :- tool_input(Tool, _).
tool_has_output(Tool) :- tool_output(Tool, _).

% Input variants per tool
tool_input_variant(Tool, TI_ID) :- tool_input(Tool, TI_ID).

% Static port requirements
port_requires(Port, PortValue, Category) :-
    input_port(_, Port),
    dimension(Port, (PortValue, Category)).

variant_single_input(TI_ID) :-
    tool_input_variant(_, TI_ID),
    1 = #count { Port : input_port(TI_ID, Port) }.

input_signature_diff(PortA, PortB) :-
    input_port(_, PortB),
    port_requires(PortA, V, Cat),
    not port_requires(PortB, V, Cat).

same_input_signature(PortA, PortB) :-
    input_port(_, PortA),
    input_port(_, PortB),
    not input_signature_diff(PortA, PortB),
    not input_signature_diff(PortB, PortA).
