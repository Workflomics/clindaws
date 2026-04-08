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

constraint_data_selector_matches_value(Sel, Actual) :-
    constraint_data_selector(Sel, Wanted),
    compatible(Actual, Wanted).

constraint_composite_data_selector(Sel) :-
    constraint_data_selector_category(Sel, _).

constraint_data_selector_matches_dim(Sel, Cat, Actual) :-
    constraint_data_selector_dim(Sel, Cat, Wanted),
    compatible(Actual, Wanted).

workflow_input(WF) :-
    holds(0, avail(WF)).
