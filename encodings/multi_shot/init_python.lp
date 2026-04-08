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

% Monotonic versions for slots (initial ones are bridged in base.lp)
available(Slot)     :- holds(0, avail(Slot)).
artifact_dim(Slot, V, Cat) :- holds(0, dim(Slot, V, Cat)).
artifact_unit(Slot, U)     :- holds(0, unit(Slot, U)).

eligible(0, Port, WF) :-
    direct_port_signature(Port, Signature),
    direct_signature_initial_artifact(Signature, WF),
    holds(0, avail(WF)),
    workflow_input(WF).

eligible(0, Port, Slot) :-
    direct_port_signature(Port, Signature),
    direct_signature_initial_artifact(Signature, Slot),
    holds(0, avail(Slot)),
    workflow_input_slot(Slot, _, _).


