% TOOL CHOICE — run choice, input variant selection, and port binding
% Docs: DOCS/encoding/tool_choice.md

% --- Run choice ---
{ occurs(T, run(Tool)) } :- time(T), precomputed_tool_allowed_at_step(Tool, T), not tool_has_input(Tool).
{ occurs(T, run(Tool)) } :- time(T), precomputed_tool_allowed_at_step(Tool, T), tool_has_input(Tool).

% --- Variant selection: exactly one input group per run ---
1 { occurs(T, use_variant(Tool, TI_ID)) : tool_input_variant(Tool, TI_ID) } 1 :-
    occurs(T, run(Tool)),
    tool_has_input(Tool).

occurs(T, run(Tool)) :- occurs(T, use_variant(Tool, _)).

% --- Port binding: exactly one eligible item per port ---
% Item satisfies one category requirement of a port at time T.
port_cat_ok(T, Port, WF, Cat) :-
    precomputed_base_artifact_bindable(Port, WF),
    holds(T, avail(WF)),
    port_requires(Port, ReqV, Cat),
    holds(T, dim(WF, V, Cat)),
    compatible(V, ReqV).

port_cat_ok(T, Port, out(TPrev, ToolPrev, ProducerPort), Cat) :-
    precomputed_output_port_bindable(Port, ProducerPort),
    holds(T, avail(out(TPrev, ToolPrev, ProducerPort))),
    port_requires(Port, ReqV, Cat),
    holds(T, dim(out(TPrev, ToolPrev, ProducerPort), V, Cat)),
    compatible(V, ReqV).

% Item is eligible if ALL categories are satisfied.
eligible(T, Port, WF) :-
    precomputed_base_artifact_bindable(Port, WF),
    holds(T, avail(WF)),
    port_cat_ok(T, Port, WF, Cat) : port_requires(Port, _, Cat).

eligible(T, Port, out(TPrev, ToolPrev, ProducerPort)) :-
    precomputed_output_port_bindable(Port, ProducerPort),
    holds(T, avail(out(TPrev, ToolPrev, ProducerPort))),
    port_cat_ok(T, Port, out(TPrev, ToolPrev, ProducerPort), Cat) : port_requires(Port, _, Cat).

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
   variant_sym_ports(TI_ID, PortHigh, PortLow),
   occurs(T, bind(Tool, PortLow, ArtifactLow)),
   occurs(T, bind(Tool, PortHigh, ArtifactHigh)),
   artifact_precedes(ArtifactHigh, ArtifactLow).

