#program init.

% ================================================================
% INIT — timestep-0 derivations grounded once for multi-shot
% ================================================================

holds(0, avail(Slot)) :-
    workflow_input_slot(Slot, _, _).

holds(0, dim(Slot, V, Cat)) :-
    workflow_input_slot_source(Slot, WF),
    holds(0, dim(WF, V, Cat)).

holds(0, unit(Slot, U)) :-
    workflow_input_slot_source(Slot, WF),
    holds(0, unit(WF, U)).

port_cat_ok(0, Port, WF, Cat) :-
    holds(0, avail(WF)),
    not base_workflow_input(WF),
    port_requires(Port, ReqV, Cat),
    holds(0, dim(WF, V, Cat)),
    compatible(V, ReqV).

eligible(0, Port, WF) :-
    holds(0, avail(WF)),
    not base_workflow_input(WF),
    input_port(_, Port),
    port_cat_ok(0, Port, WF, Cat) : port_requires(Port, _, Cat).
