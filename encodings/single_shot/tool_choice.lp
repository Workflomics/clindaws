% TOOL CHOICE — run choice, input variant selection, and port binding
% Docs: DOCS/encoding/tool_choice.md

% --- Run choice ---
{ occurs(T, run(Tool)) } :- time(T), tool(Tool), not tool_has_input(Tool).
{ occurs(T, run(Tool)) } :- time(T), tool(Tool), tool_has_input(Tool).

% --- Variant selection: exactly one input group per run ---
1 { occurs(T, use_variant(Tool, TI_ID)) : tool_input_variant(Tool, TI_ID) } 1 :-
    occurs(T, run(Tool)),
    tool_has_input(Tool).

occurs(T, run(Tool)) :- occurs(T, use_variant(Tool, _)).

% --- Port binding: exactly one eligible item per port ---
% Item satisfies one category requirement of a port at time T
port_cat_ok(T, Port, WF, Cat) :-
    holds(T, avail(WF)),
    port_requires(Port, ReqV, Cat),
    holds(T, dim(WF, V, Cat)),
    compatible(V, ReqV).

% Item is eligible if ALL categories are satisfied
eligible(T, Port, WF) :-
    holds(T, avail(WF)),
    input_port(_, Port),
    port_cat_ok(T, Port, WF, Cat) : port_requires(Port, _, Cat).

1 { occurs(T, bind(Tool, Port, WF)) : eligible(T-1, Port, WF) } 1 :-
    occurs(T, use_variant(Tool, TI_ID)),
    input_port(TI_ID, Port).

:- occurs(T, use_variant(Tool, TI_ID)),
   occurs(T, bind(Tool, Port1, WF1)),
   occurs(T, bind(Tool, Port2, WF2)),
   Port1 < Port2,
   input_port(TI_ID, Port1),
   input_port(TI_ID, Port2),
   same_input_signature(Port1, Port2),
   equivalent_workflow_inputs(WF2, WF1).

artifact_parent(out(T, Tool, PortOut), WF) :-
    occurs(T, output(Tool, _, PortOut)),
    occurs(T, bind(Tool, _, WF)).

artifact_ancestor(Artifact, Ancestor) :-
    artifact_parent(Artifact, Ancestor).

artifact_ancestor(Artifact, Ancestor) :-
    artifact_parent(Artifact, Mid),
    artifact_ancestor(Mid, Ancestor).

:- occurs(T, use_variant(Tool, TI_ID)),
   occurs(T, bind(Tool, Port1, out(T1, Tool1, _))),
   occurs(T, bind(Tool, Port2, out(T2, Tool2, _))),
   Port1 < Port2,
   input_port(TI_ID, Port1),
   input_port(TI_ID, Port2),
   same_input_signature(Port1, Port2),
   T1 < T2.

paired_symmetric_producers(T1, Tool1, TI1, Port1, WF1, T2, Tool2, TI2, Port2, WF2) :-
    T1 < T2,
    occurs(T1, use_variant(Tool1, TI1)),
    occurs(T2, use_variant(Tool2, TI2)),
    variant_single_input(TI1),
    variant_single_input(TI2),
    occurs(T1, bind(Tool1, Port1, WF1)),
    occurs(T2, bind(Tool2, Port2, WF2)),
    same_input_signature(Port1, Port2),
    occurs(TJ, use_variant(JoinTool, JoinTI)),
    occurs(TJ, bind(JoinTool, JoinPort1, out(T1, Tool1, _))),
    occurs(TJ, bind(JoinTool, JoinPort2, out(T2, Tool2, _))),
    TJ > T2,
    input_port(JoinTI, JoinPort1),
    input_port(JoinTI, JoinPort2),
    JoinPort2 < JoinPort1,
    same_input_signature(JoinPort1, JoinPort2).

:- paired_symmetric_producers(T1, Tool1, TI1, Port1, WF, T2, Tool2, TI2, Port2, WF).

:- paired_symmetric_producers(T1, Tool1, TI1, Port1, WF1, T2, Tool2, TI2, Port2, WF2),
   artifact_ancestor(WF1, WF2).
