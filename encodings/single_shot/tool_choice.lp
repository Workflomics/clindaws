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
