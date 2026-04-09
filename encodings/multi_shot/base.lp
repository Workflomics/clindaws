#program base.

% Taxonomy — used for goal compatibility (goals may use non-terminal values)
ancestor(Child, Parent)   :- taxonomy(_, _, (_, Child, Parent)).
ancestor(Child, Ancestor) :- ancestor(Child, Mid), ancestor(Mid, Ancestor).

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

port_category(Port, Category) :-
    input_port(_, Port),
    dimension(Port, (_, Category)).

goal_requirement(GoalID, goalreq(GoalID, GoalValue, Category), GoalValue, Category) :-
    goal_output(GoalID, GoalValue, Category).

goal_requirement_accepts(GoalReq, ActualValue, Category) :-
    goal_requirement(_, GoalReq, GoalValue, Category),
    compatible(ActualValue, GoalValue).

port_property_accepts(Tool, Port, ActualValue, Category, Prop) :-
    port_property(Tool, Port, RequiredValue, Category, Prop),
    compatible(ActualValue, RequiredValue).

variant_single_input(TI_ID) :-
    tool_input_variant(_, TI_ID),
    1 = #count { Port : input_port(TI_ID, Port) }.

input_signature_diff(PortA, PortB) :-
    input_port(_, PortB),
    port_requires(PortA, V, Cat),
    not port_requires(PortB, V, Cat).

port_signature_mismatch(PortA, PortB) :-
    input_port(_, PortA),
    input_port(_, PortB),
    port_requires(PortA, V, Cat),
    not port_requires(PortB, V, Cat).

same_input_signature(PortA, PortB) :-
    input_port(_, PortA),
    input_port(_, PortB),
    not port_signature_mismatch(PortA, PortB),
    not port_signature_mismatch(PortB, PortA).

variant_sym_ports(TI_ID, PortHigh, PortLow) :-
    input_port(TI_ID, PortHigh),
    input_port(TI_ID, PortLow),
    PortLow < PortHigh,
    same_input_signature(PortHigh, PortLow).

% ----------------------------------------------------------------
% From unit_reasoning.lp — static derivations only
% (unit carry-forward and production are in step.lp)
% ----------------------------------------------------------------

% Bridge monotonic properties for internal symmetry breakers
available(A)          :- holds(0, avail(A)).
artifact_dim(A, V, C) :- holds(0, dim(A, V, C)).
artifact_unit(A, U)   :- holds(0, unit(A, U)).

artifact_dim(WF, V, Category) :-
    holds(0, dim(WF, V, Category)).

base_workflow_input(WF) :-
    holds(0, avail(WF)).

workflow_input(WF) :-
    base_workflow_input(WF).

workflow_input_profile_mismatch(WF1, WF2) :-
    base_workflow_input(WF1),
    base_workflow_input(WF2),
    artifact_dim(WF1, V, Cat),
    not artifact_dim(WF2, V, Cat).

workflow_input_unit_mismatch(WF1, WF2) :-
    base_workflow_input(WF1),
    base_workflow_input(WF2),
    artifact_unit(WF1, U),
    not artifact_unit(WF2, U).

equivalent_workflow_inputs(WF1, WF2) :-
    base_workflow_input(WF1),
    base_workflow_input(WF2),
    WF1 < WF2,
    not workflow_input_profile_mismatch(WF1, WF2),
    not workflow_input_profile_mismatch(WF2, WF1),
    not workflow_input_unit_mismatch(WF1, WF2),
    not workflow_input_unit_mismatch(WF2, WF1).

equivalent_workflow_input_pair(WF1, WF2) :-
    equivalent_workflow_inputs(WF1, WF2).

equivalent_workflow_input_pair(WF1, WF2) :-
    equivalent_workflow_inputs(WF2, WF1).

smaller_equivalent_workflow_input(WF, Smaller) :-
    base_workflow_input(WF),
    base_workflow_input(Smaller),
    equivalent_workflow_input_pair(WF, Smaller),
    Smaller < WF.

canonical_workflow_input(WF) :-
    base_workflow_input(WF),
    not smaller_equivalent_workflow_input(WF, _).

workflow_input_class_member(Rep, Rep) :-
    canonical_workflow_input(Rep).

workflow_input_class_member(Rep, WF) :-
    canonical_workflow_input(Rep),
    equivalent_workflow_input_pair(WF, Rep).

workflow_input_class_repeated(Rep) :-
    canonical_workflow_input(Rep),
    1 < #count { WF : workflow_input_class_member(Rep, WF) }.

workflow_input_collapsed_member(WF) :-
    workflow_input_class_member(Rep, WF),
    workflow_input_class_repeated(Rep).

workflow_input_slot(slot(Rep, WF), Rep, Rank) :-
    workflow_input_class_repeated(Rep),
    workflow_input_class_member(Rep, WF),
    Rank = #count {
        Other : workflow_input_class_member(Rep, Other), Other <= WF
    }.

workflow_input_slot_source(slot(Rep, WF), WF) :-
    workflow_input_slot(slot(Rep, WF), Rep, _).

workflow_input_slot_class_size(Rep, N) :-
    workflow_input_class_repeated(Rep),
    N = #count { Slot : workflow_input_slot(Slot, Rep, _) }.

planner_workflow_input(WF) :-
    base_workflow_input(WF),
    full_workflow_input_witnesses.

planner_workflow_input(WF) :-
    base_workflow_input(WF),
    not full_workflow_input_witnesses,
    not workflow_input_collapsed_member(WF).

planner_workflow_input(Slot) :-
    workflow_input_slot(Slot, _, _),
    not full_workflow_input_witnesses.

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
