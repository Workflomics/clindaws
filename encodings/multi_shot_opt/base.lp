#program base.

% Taxonomy — used for goal compatibility (goals may use non-terminal values)
ancestor(Child, Parent)   :- taxonomy(_, _, (_, Child, Parent)).
ancestor(Child, Ancestor) :- ancestor(Child, Mid), ancestor(Mid, Ancestor).
is_terminal(V) :- taxonomy(_, _, (_, V, _)), not taxonomy(_, _, (_, _, V)).
compatible(V, V)        :- is_terminal(V).
compatible(V, Ancestor) :- is_terminal(V), ancestor(V, Ancestor).

% Tool structure helpers for candidate-based optimized grounding
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

% Tool taxonomy
tool_abstract_class(Class) :- tool_taxonomy(_, (_, Class)).
tool_concrete(Tool)        :- tool(Tool), not tool_abstract_class(Tool).
tool_parent(Child, Parent) :- tool_taxonomy(_, (Child, Parent)).
tool_ancestor(Child, Parent)   :- tool_parent(Child, Parent).
tool_ancestor(Child, Ancestor) :- tool_parent(Child, Mid), tool_ancestor(Mid, Ancestor).
tool_is_a(Tool, Tool)      :- tool(Tool).
tool_is_a(Tool, Class)     :- tool(Tool), tool_ancestor(Tool, Class).
tool_is_a(Class, Ancestor) :- tool_abstract_class(Class), tool_ancestor(Class, Ancestor).
tool_is_a(Class, Class)    :- tool_abstract_class(Class).
tools_in_class(Class, Tool):- tool_is_a(Tool, Class), tool(Tool).

requirement_filtering_active :- target_capability(_, _).

tool_has_req(Tool, Type)         :- tool_requirement(Tool, (Type, _)).
tool_requirement_met(Tool, Type) :- tool(Tool), target_capability(Type, _), not tool_has_req(Tool, Type).
tool_requirement_met(Tool, Type) :- tool(Tool), tool_requirement(Tool, (Type, Val)), target_capability(Type, Val).
tool_req_type_undeclared(Tool)   :- tool_has_req(Tool, Type), requirement_filtering_active, not target_capability(Type, _).
tool_req_compatible(Tool)        :- tool(Tool), not tool_req_type_undeclared(Tool),
                                    tool_requirement_met(Tool, Type) : target_capability(Type, _).

% Initial eligibility at t=0 — candidate values are terminal, but available artifacts may
% still be expressed at ancestor categories, so compatibility is checked per category.
port_cat_ok(0, Port, WF, Category) :-
    holds(0, avail(WF)),
    port_requires(Port, PortValue, Category),
    holds(0, dim(WF, ActualValue, Category)),
    compatible(PortValue, ActualValue).

eligible(0, Port, WF) :-
    holds(0, avail(WF)),
    input_port(_, Port),
    port_cat_ok(0, Port, WF, Category) : port_category(Port, Category).
