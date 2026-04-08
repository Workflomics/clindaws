#program init.

% ================================================================
% INIT — timestep-0 derivations grounded once for multi-shot
% ================================================================

holds(0, avail(Slot)) :-
    workflow_input_slot(Slot, _, _),
    not full_workflow_input_witnesses.

holds(0, dim(Slot, V, Cat)) :-
    workflow_input_slot_source(Slot, WF),
    holds(0, dim(WF, V, Cat)),
    not full_workflow_input_witnesses.

holds(0, unit(Slot, U)) :-
    workflow_input_slot_source(Slot, WF),
    holds(0, unit(WF, U)),
    not full_workflow_input_witnesses.

% Monotonic versions for slots (initial ones are bridged in base.lp)
available(Slot)     :- holds(0, avail(Slot)).
artifact_dim(Slot, V, Cat) :- holds(0, dim(Slot, V, Cat)).
artifact_unit(Slot, U)     :- holds(0, unit(Slot, U)).

port_cat_ok(0, Port, WF, Cat) :-
    planner_workflow_input(WF),
    holds(0, avail(WF)),
    port_requires(Port, _, Cat),
    not holds(0, dim(WF, _, Cat)).

port_cat_ok(0, Port, WF, Cat) :-
    planner_workflow_input(WF),
    holds(0, avail(WF)),
    port_requires(Port, ReqV, Cat),
    holds(0, dim(WF, V, Cat)),
    compatible(V, ReqV).

port_cat_ok(0, Port, WF, Cat) :-
    holds(0, avail(WF)),
    not planner_workflow_input(WF),
    port_requires(Port, ReqV, Cat),
    holds(0, dim(WF, V, Cat)),
    compatible(V, ReqV).

eligible(0, Port, WF) :-
    holds(0, avail(WF)),
    input_port(_, Port),
    port_cat_ok(0, Port, WF, Cat) : port_requires(Port, _, Cat).



