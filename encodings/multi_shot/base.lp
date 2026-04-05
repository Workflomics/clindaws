#program base.

% ================================================================
% BASE — static derivations grounded once
% ================================================================

% ----------------------------------------------------------------
% From pre_compute.lp
% ----------------------------------------------------------------

% Flattened taxonomy
ancestor(Child, Parent)   :- taxonomy(_, _, (_, Child, Parent)).
ancestor(Child, Ancestor) :- ancestor(Child, Mid), ancestor(Mid, Ancestor).

% Terminal: a value that no other value has as parent
is_terminal(V) :- taxonomy(_, _, (_, V, _)), not taxonomy(_, _, (_, _, V)).

% compatible(V, Ancestor): V is terminal and equal to or descends from Ancestor
compatible(V, V)        :- is_terminal(V).
compatible(V, Ancestor) :- is_terminal(V), ancestor(V, Ancestor).

% Tool structure
tool_has_input(Tool)  :- tool_input(Tool, _).
tool_has_output(Tool) :- tool_output(Tool, _).

% Input variants per tool
tool_input_variant(Tool, TI_ID) :- tool_input(Tool, TI_ID).
tool_input_port(Tool, Port) :- tool_input(Tool, TI_ID), input_port(TI_ID, Port).
tool_single_input(Tool) :- tool(Tool), 1 = #count { Port : tool_input_port(Tool, Port) }.

% Static port requirements
port_requires(Port, PortValue, Category) :-
    input_port(_, Port),
    dimension(Port, (PortValue, Category)).

port_signature_mismatch(PortA, PortB) :-
    input_port(_, PortA),
    input_port(_, PortB),
    port_requires(PortA, ReqV, Cat),
    not port_requires(PortB, ReqV, Cat).

same_input_signature(PortA, PortB) :-
    input_port(_, PortA),
    input_port(_, PortB),
    not port_signature_mismatch(PortA, PortB),
    not port_signature_mismatch(PortB, PortA).

% ----------------------------------------------------------------
% From tool_taxonomy_logic.lp
% ----------------------------------------------------------------

tool_abstract_class(Class) :- tool_taxonomy(_, (_, Class)).

tool_concrete(Tool) :-
    tool(Tool),
    not tool_abstract_class(Tool).

tool_parent(Child, Parent) :- tool_taxonomy(_, (Child, Parent)).

tool_ancestor(Child, Parent) :-
    tool_parent(Child, Parent).

tool_ancestor(Child, Ancestor) :-
    tool_parent(Child, Mid),
    tool_ancestor(Mid, Ancestor).

tool_is_a(Tool, Tool) :- tool(Tool).

tool_is_a(Tool, Class) :-
    tool(Tool),
    tool_ancestor(Tool, Class).

tool_is_a(Class, Ancestor) :-
    tool_abstract_class(Class),
    tool_ancestor(Class, Ancestor).

tool_is_a(Class, Class) :-
    tool_abstract_class(Class).

tool_direct_selector(Tool, Tool) :-
    tool(Tool).

tool_direct_selector(Tool, Selector) :-
    tool(Tool),
    tool_taxonomy(_, (Tool, Selector)).

constraint_selector_matches_tool(Sel, Tool) :-
    constraint_selector(Sel, Selector),
    constraint_selector_mode(Sel, "transitive"),
    tool_is_a(Tool, Selector).

constraint_selector_matches_tool(Sel, Tool) :-
    constraint_selector(Sel, Selector),
    constraint_selector_mode(Sel, "exact"),
    tool_direct_selector(Tool, Selector).

constraint_data_selector_matches_value(Sel, Actual) :-
    constraint_data_selector(Sel, Wanted),
    compatible(Actual, Wanted).

tools_in_class(Class, Tool) :-
    tool_is_a(Tool, Class),
    tool(Tool).

% ----------------------------------------------------------------
% From requirements.lp — static derivations only
% (constraints are in step.lp)
% ----------------------------------------------------------------

requirement_filtering_active :- target_capability(_, _).

tool_has_req(Tool, Type) :-
    tool_requirement(Tool, (Type, _)).

% Case (a): no requirement for this type → satisfied by default
tool_requirement_met(Tool, Type) :-
    tool(Tool),
    target_capability(Type, _),
    not tool_has_req(Tool, Type).

% Case (b): requirement declared and matches target
tool_requirement_met(Tool, Type) :-
    tool(Tool),
    tool_requirement(Tool, (Type, Val)),
    target_capability(Type, Val).

tool_req_type_undeclared(Tool) :-
    tool_has_req(Tool, Type),
    requirement_filtering_active,
    not target_capability(Type, _).

tool_req_compatible(Tool) :-
    tool(Tool),
    not tool_req_type_undeclared(Tool),
    tool_requirement_met(Tool, Type) : target_capability(Type, _).

port_has_req(Tool, Port, Type) :-
    port_requirement(Tool, Port, (Type, _)).

port_requirement_met(Tool, Port, Type) :-
    tool(Tool),
    input_port(_, Port),
    target_capability(Type, _),
    not port_has_req(Tool, Port, Type).

port_requirement_met(Tool, Port, Type) :-
    tool(Tool),
    input_port(_, Port),
    port_requirement(Tool, Port, (Type, Val)),
    target_capability(Type, Val).

port_env_compatible(Tool, Port) :-
    tool(Tool),
    input_port(_, Port),
    port_requirement_met(Tool, Port, Type) : target_capability(Type, _).

% Diagnostics (uncomment #show tool_req_incompatible/1. in show.lp to enable)
tool_req_incompatible(Tool) :-
    tool(Tool),
    requirement_filtering_active,
    not tool_req_compatible(Tool).

% ----------------------------------------------------------------
% From unit_reasoning.lp — static derivations only
% (unit carry-forward and production are in step.lp)
% ----------------------------------------------------------------

workflow_input(WF) :-
    holds(0, avail(WF)).

workflow_input_profile_mismatch(WF1, WF2) :-
    workflow_input(WF1),
    workflow_input(WF2),
    holds(0, dim(WF1, V, Cat)),
    not holds(0, dim(WF2, V, Cat)).

equivalent_workflow_inputs(WF1, WF2) :-
    workflow_input(WF1),
    workflow_input(WF2),
    WF1 < WF2,
    not workflow_input_profile_mismatch(WF1, WF2),
    not workflow_input_profile_mismatch(WF2, WF1).

unit_ancestor(U, U) :- unit_is_a(U, _).
unit_ancestor(U, U) :- unit_is_a(_, U).

unit_ancestor(U, A) :-
    unit_is_a(U, P),
    unit_ancestor(P, A).

unit_root(U, R) :-
    unit_ancestor(U, R),
    not unit_is_a(R, _).

units_compatible(U1, U2) :-
    unit_root(U1, R),
    unit_root(U2, R).

units_compatible(U, U) :- unit_is_a(U, _).
units_compatible(U, U) :- unit_is_a(_, U).

units_compatible(none, U) :- unit_is_a(U, _).
units_compatible(none, U) :- unit_is_a(_, U).
units_compatible(U, none) :- unit_is_a(U, _).
units_compatible(U, none) :- unit_is_a(_, U).
