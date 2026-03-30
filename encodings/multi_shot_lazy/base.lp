#program base.

ancestor(Child, Parent)   :- taxonomy(_, _, (_, Child, Parent)).
ancestor(Child, Ancestor) :- ancestor(Child, Mid), ancestor(Mid, Ancestor).

is_terminal(V) :- taxonomy(_, _, (_, V, _)), not taxonomy(_, _, (_, _, V)).

compatible(V, V)        :- is_terminal(V).
compatible(V, Ancestor) :- is_terminal(V), ancestor(V, Ancestor).

artifact_dim(WF, V, Category) :-
    holds(0, dim(WF, V, Category)).

tool_has_input(Tool) :-
    lazy_tool_candidate(Candidate, Tool),
    lazy_candidate_input_port(Candidate, _).

tool_has_output(Tool) :-
    lazy_tool_candidate(Candidate, Tool),
    lazy_candidate_output_port(Candidate, _).

tool_input_port(Tool, Port) :-
    lazy_tool_candidate(Candidate, Tool),
    lazy_candidate_input_port(Candidate, Port).

candidate_single_input(Candidate) :-
    lazy_tool_candidate(Candidate, _),
    1 = #count { Port : lazy_candidate_input_port(Candidate, Port) }.

lazy_candidate_input_category(Candidate, Port, Category) :-
    lazy_candidate_input_value(Candidate, Port, _, Category).

lazy_candidate_output_category(Candidate, Port, Category) :-
    lazy_candidate_output_singleton(Candidate, Port, _, Category).

lazy_candidate_output_category(Candidate, Port, Category) :-
    lazy_candidate_output_choice_value(Candidate, Port, _, Category).

same_candidate_input_signature(CandidateA, PortA, CandidateB, PortB) :-
    lazy_candidate_input_signature_id(CandidateA, PortA, SigID),
    lazy_candidate_input_signature_id(CandidateB, PortB, SigID).

goal_id(GoalID) :- goal_output(GoalID, _, _).

lazy_candidate_output_goal_match(Candidate, Port, GoalID, Category) :-
    goal_output(GoalID, GoalV, Category),
    lazy_candidate_output_singleton(Candidate, Port, ActualV, Category),
    compatible(ActualV, GoalV).

lazy_candidate_output_goal_match(Candidate, Port, GoalID, Category) :-
    goal_output(GoalID, GoalV, Category),
    lazy_candidate_output_choice_value(Candidate, Port, ActualV, Category),
    compatible(ActualV, GoalV).

lazy_initial_goal_candidate_output_mismatch(Candidate) :-
    enable_all_outputs_consumed,
    lazy_tool_candidate(Candidate, _),
    NP = #count { Port : lazy_candidate_output_port(Candidate, Port) },
    NG = #count { G : goal_id(G) },
    NP != NG.

lazy_candidate_output_goal_satisfies(Candidate, Port, GoalID) :-
    goal_id(GoalID),
    lazy_candidate_output_port(Candidate, Port),
    lazy_candidate_output_goal_match(Candidate, Port, GoalID, Category) :
        goal_output(GoalID, _, Category).

lazy_candidate_goal_satisfies(Candidate, GoalID) :-
    lazy_candidate_output_port(Candidate, Port),
    lazy_candidate_output_goal_satisfies(Candidate, Port, GoalID).

lazy_initial_goal_candidate(Candidate) :-
    lazy_tool_candidate(Candidate, _),
    lazy_candidate_goal_satisfies(Candidate, GoalID) : goal_id(GoalID),
    not lazy_initial_goal_candidate_output_mismatch(Candidate).

lazy_initial_goal_tool(Tool) :-
    lazy_tool_candidate(Candidate, Tool),
    lazy_initial_goal_candidate(Candidate).

lazy_initial_goal_output_preferred_value(Candidate, Port, V, Category) :-
    lazy_initial_goal_candidate(Candidate),
    lazy_candidate_output_choice_value(Candidate, Port, V, Category),
    goal_output(_, GoalV, Category),
    compatible(V, GoalV),
    not lazy_initial_goal_output_smaller_value(Candidate, Port, V, Category).

lazy_initial_goal_output_smaller_value(Candidate, Port, V, Category) :-
    lazy_initial_goal_candidate(Candidate),
    lazy_candidate_output_choice_value(Candidate, Port, V, Category),
    lazy_candidate_output_choice_value(Candidate, Port, V2, Category),
    goal_output(_, GoalV, Category),
    compatible(V, GoalV),
    compatible(V2, GoalV),
    V2 < V.

lazy_initial_goal_output_value(Candidate, Port, V, Category) :-
    lazy_initial_goal_output_preferred_value(Candidate, Port, V, Category).

lazy_initial_goal_output_value(Candidate, Port, V, Category) :-
    lazy_initial_goal_candidate(Candidate),
    lazy_candidate_output_choice_value(Candidate, Port, V, Category),
    not goal_output(_, _, Category).

lazy_query_output_has_goal_value(Candidate, Port, Category) :-
    lazy_candidate_output_choice_value(Candidate, Port, V, Category),
    goal_output(_, GoalV, Category),
    compatible(V, GoalV).

lazy_query_output_preferred_value(Candidate, Port, V, Category) :-
    lazy_candidate_output_choice_value(Candidate, Port, V, Category),
    lazy_query_output_has_goal_value(Candidate, Port, Category),
    goal_output(_, GoalV, Category),
    compatible(V, GoalV),
    not lazy_query_output_smaller_goal_value(Candidate, Port, V, Category).

lazy_query_output_smaller_goal_value(Candidate, Port, V, Category) :-
    lazy_candidate_output_choice_value(Candidate, Port, V, Category),
    lazy_candidate_output_choice_value(Candidate, Port, V2, Category),
    goal_output(_, GoalV, Category),
    compatible(V, GoalV),
    compatible(V2, GoalV),
    V2 < V.

lazy_query_output_preferred_value(Candidate, Port, V, Category) :-
    lazy_candidate_output_choice_value(Candidate, Port, V, Category),
    not lazy_query_output_has_goal_value(Candidate, Port, Category),
    not lazy_query_output_smaller_value(Candidate, Port, V, Category).

lazy_query_output_smaller_value(Candidate, Port, V, Category) :-
    lazy_candidate_output_choice_value(Candidate, Port, V, Category),
    lazy_candidate_output_choice_value(Candidate, Port, V2, Category),
    V2 < V.

lazy_query_output_value(Candidate, Port, V, Category) :-
    lazy_query_output_preferred_value(Candidate, Port, V, Category).

initial_goal_dim_match(GoalID, WF, GoalV, Category) :-
    goal_output(GoalID, GoalV, Category),
    holds(0, avail(WF)),
    artifact_dim(WF, ActualV, Category),
    compatible(ActualV, GoalV).

initial_goal_dim_missing(GoalID, WF) :-
    goal_output(GoalID, GoalV, Category),
    holds(0, avail(WF)),
    not initial_goal_dim_match(GoalID, WF, GoalV, Category).

initial_goal_satisfied_by(GoalID, WF) :-
    holds(0, avail(WF)),
    goal_output(GoalID, _, _),
    initial_goal_dim_match(GoalID, WF, _, _),
    not initial_goal_dim_missing(GoalID, WF).

initial_goal_satisfied(GoalID) :-
    goal_id(GoalID),
    initial_goal_satisfied_by(GoalID, WF).

initial_goal_holds :-
    initial_goal_satisfied(GoalID) : goal_id(GoalID).

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

eligible(Candidate, Port, WF) :-
    lazy_initial_bindable(Candidate, Port, WF).
