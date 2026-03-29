% PRE-COMPUTE — candidate-schema single-shot static derivations

ancestor(Child, Parent) :- taxonomy(_, _, (_, Child, Parent)).
ancestor(Child, Ancestor) :- ancestor(Child, Mid), ancestor(Mid, Ancestor).

is_terminal(V) :- taxonomy(_, _, (_, V, _)), not taxonomy(_, _, (_, _, V)).

compatible(V, V)        :- is_terminal(V).
compatible(V, Ancestor) :- is_terminal(V), ancestor(V, Ancestor).

tool_has_input(Tool)            :- tool_candidate(Candidate, Tool), candidate_in(Candidate, _, _, _).
tool_has_output(Tool)           :- tool_candidate(Candidate, Tool), candidate_out(Candidate, _, _, _).
tool_input_variant(Tool, TI_ID) :- tool_candidate(TI_ID, Tool).

input_port(TI_ID, port(TI_ID, Port)) :- candidate_in(TI_ID, Port, _, _).
candidate_output_port(TI_ID, port(TI_ID, Port)) :- candidate_out(TI_ID, Port, _, _).
candidate_output_value(TI_ID, port(TI_ID, Port), Value, Category) :-
    candidate_out(TI_ID, Port, Value, Category).

port_requires(port(TI_ID, Port), PortValue, Category) :-
    candidate_in(TI_ID, Port, PortValue, Category).

port_category(Port, Category) :- port_requires(Port, _, Category).

port_cat_ok(0, Port, WF, Category) :-
    holds(0, avail(WF)),
    port_requires(Port, PortValue, Category),
    holds(0, dim(WF, ActualValue, Category)),
    compatible(PortValue, ActualValue).

eligible(0, Port, WF) :-
    holds(0, avail(WF)),
    input_port(_, Port),
    port_cat_ok(0, Port, WF, Category) : port_category(Port, Category).
