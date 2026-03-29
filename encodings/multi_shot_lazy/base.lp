#program base.

ancestor(Child, Parent)   :- taxonomy(_, _, (_, Child, Parent)).
ancestor(Child, Ancestor) :- ancestor(Child, Mid), ancestor(Mid, Ancestor).

is_terminal(V) :- taxonomy(_, _, (_, V, _)), not taxonomy(_, _, (_, _, V)).

compatible(V, V)        :- is_terminal(V).
compatible(V, Ancestor) :- is_terminal(V), ancestor(V, Ancestor).

tool_has_input(Tool) :-
    lazy_tool_candidate(Candidate, Tool),
    lazy_candidate_input_port(Candidate, _).

tool_has_output(Tool) :-
    lazy_tool_candidate(Candidate, Tool),
    lazy_candidate_output_port(Candidate, _).

tool_input_port(Tool, Port) :-
    lazy_tool_candidate(Candidate, Tool),
    lazy_candidate_input_port(Candidate, Port).

lazy_candidate_input_category(Candidate, Port, Category) :-
    lazy_candidate_input_value(Candidate, Port, _, Category).

lazy_candidate_output_category(Candidate, Port, Category) :-
    lazy_candidate_output_singleton(Candidate, Port, _, Category).

lazy_candidate_output_category(Candidate, Port, Category) :-
    lazy_candidate_output_choice_value(Candidate, Port, _, Category).

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

port_has_req(Tool, Port, Type) :-
    port_requirement(Tool, Port, (Type, _)).

port_requirement_met(Tool, Port, Type) :-
    tool(Tool),
    tool_input_port(Tool, Port),
    target_capability(Type, _),
    not port_has_req(Tool, Port, Type).

port_requirement_met(Tool, Port, Type) :-
    tool(Tool),
    tool_input_port(Tool, Port),
    port_requirement(Tool, Port, (Type, Val)),
    target_capability(Type, Val).

port_env_compatible(Tool, Port) :-
    tool(Tool),
    tool_input_port(Tool, Port),
    port_requirement_met(Tool, Port, Type) : target_capability(Type, _).

eligible(0, Candidate, Port, WF) :-
    lazy_initial_bindable(Candidate, Port, WF).
