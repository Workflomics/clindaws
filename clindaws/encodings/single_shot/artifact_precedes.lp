% Canonical artifact order for same-signature ports. This removes duplicate
% witnesses where identical ports are filled by the same artifacts in a
% different order, including mixed workflow-input / produced-output swaps.
artifact_precedes(WF1, WF2) :-
    workflow_input(WF1),
    workflow_input(WF2),
    WF1 < WF2.

artifact_precedes(WF, Slot) :-
    workflow_input(WF),
    workflow_input_slot(Slot, Rep, _),
    WF < Rep.

artifact_precedes(Slot1, Slot2) :-
    workflow_input_slot(Slot1, Rep1, Rank1),
    workflow_input_slot(Slot2, Rep2, Rank2),
    Rep1 < Rep2.

artifact_precedes(Slot, WF) :-
    workflow_input_slot(Slot, Rep, _),
    workflow_input(WF),
    Rep < WF.

artifact_precedes(Slot1, Slot2) :-
    workflow_input_slot(Slot1, Rep, Rank1),
    workflow_input_slot(Slot2, Rep, Rank2),
    Rank1 < Rank2.

artifact_precedes(Slot, out(T, Tool, Port)) :-
    workflow_input_slot(Slot, _, _),
    occurs(T, output(Tool, _, Port)).

artifact_precedes(WF, out(T, Tool, Port)) :-
    workflow_input(WF),
    occurs(T, output(Tool, _, Port)).

artifact_precedes(out(T1, Tool1, Port1), out(T2, Tool2, Port2)) :-
    occurs(T1, output(Tool1, _, Port1)),
    occurs(T2, output(Tool2, _, Port2)),
    T1 < T2.

artifact_precedes(out(T, Tool, Port1), out(T, Tool, Port2)) :-
    occurs(T, output(Tool, _, Port1)),
    occurs(T, output(Tool, _, Port2)),
    Port1 < Port2,
    occurs(T, run(Tool)).
