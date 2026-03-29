% TOOL CHOICE — candidate-schema single-shot run choice and binding

{ occurs(T, run(Tool)) } :- time(T), tool(Tool), not tool_has_input(Tool).
{ occurs(T, run(Tool)) } :- time(T), tool(Tool), tool_has_input(Tool).

1 { occurs(T, use_variant(Tool, TI_ID)) : tool_input_variant(Tool, TI_ID) } 1 :-
    occurs(T, run(Tool)),
    tool_input_variant(Tool, _).

occurs(T, run(Tool)) :- occurs(T, use_variant(Tool, _)).

port_cat_ok(T, Port, WF, Cat) :-
    holds(T, avail(WF)),
    port_requires(Port, ReqV, Cat),
    holds(T, dim(WF, V, Cat)),
    compatible(ReqV, V).

port_category(Port, Category) :- port_requires(Port, _, Category).

eligible(T, Port, WF) :-
    holds(T, avail(WF)),
    input_port(_, Port),
    port_cat_ok(T, Port, WF, Cat) : port_category(Port, Cat).

1 { occurs(T, bind(Tool, Port, WF)) : eligible(T-1, Port, WF) } 1 :-
    occurs(T, use_variant(Tool, TI_ID)),
    input_port(TI_ID, Port).
