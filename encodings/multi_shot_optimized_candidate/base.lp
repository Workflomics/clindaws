#program base.

% ================================================================
% BASE — static compressed-candidate domain facts
% ================================================================
% Unlike plain multi-shot, this backend starts from translation-time candidate
% records, signature support classes, and goal-distance facts. The base program
% therefore focuses on wiring those emitted facts into the predicates used by
% the step/check programs instead of reconstructing large compatibility joins at
% grounding time.

ancestor(Child, Parent)   :- taxonomy(_, _, (_, Child, Parent)).
ancestor(Child, Ancestor) :- ancestor(Child, Mid), ancestor(Mid, Ancestor).

is_terminal(V) :- taxonomy(_, _, (_, V, _)), not taxonomy(_, _, (_, _, V)).

compatible(V, V)        :- is_terminal(V).
compatible(V, Ancestor) :- is_terminal(V), ancestor(V, Ancestor).

% Bridge monotonic properties from initial facts (emitted by Python wrapper)
available(A)          :- holds(0, avail(A)).
artifact_dim(A, V, C) :- holds(0, dim(A, V, C)).
artifact_unit(A, U)   :- holds(0, unit(A, U)).

artifact_profile_member(A, Category, ProfileId) :-
    artifact_dim(A, ActualV, Category),
    dynamic_profile_accepts(ProfileId, ActualV).

is_workflow_input(WF) :- holds(0, avail(WF)).

workflow_input(WF) :-
    is_workflow_input(WF).

workflow_input_lineage(WF, WF) :-
    holds(0, avail(WF)).



tool_has_input(Tool) :-
    dynamic_tool_candidate(Candidate, Tool),
    dynamic_candidate_input_port(Candidate, _).

tool_has_output(Tool) :-
    dynamic_tool_candidate(Candidate, Tool),
    dynamic_candidate_output_port(Candidate, _).

tool_input_port(Tool, Port) :-
    dynamic_tool_candidate(Candidate, Tool),
    dynamic_candidate_input_port(Candidate, Port).

candidate_single_input(Candidate) :-
    dynamic_tool_candidate(Candidate, _),
    1 = #count { Port : dynamic_candidate_input_port(Candidate, Port) }.

dynamic_signature_category(SignatureId, Category) :-
    dynamic_signature_profile(SignatureId, Category, _).

dynamic_candidate_input_category(Candidate, Port, Category) :-
    dynamic_candidate_input_signature_id(Candidate, Port, SignatureId),
    dynamic_signature_category(SignatureId, Category).

dynamic_candidate_output_category(Candidate, Port, Category) :-
    dynamic_candidate_output_singleton(Candidate, Port, _, Category).

dynamic_candidate_output_category(Candidate, Port, Category) :-
    dynamic_candidate_output_choice_value(Candidate, Port, _, Category).

same_candidate_input_signature(CandidateA, PortA, CandidateB, PortB) :-
    dynamic_candidate_input_signature_id(CandidateA, PortA, SigID),
    dynamic_candidate_input_signature_id(CandidateB, PortB, SigID).

candidate_has_repeated_signature_inputs(Candidate) :-
    same_candidate_input_signature(Candidate, Port1, Candidate, Port2),
    Port1 < Port2.

goal_id(GoalID) :- goal_output(GoalID, _, _).

dynamic_candidate_output_goal_match(Candidate, Port, GoalID, Category) :-
    goal_output(GoalID, GoalV, Category),
    dynamic_candidate_output_singleton(Candidate, Port, ActualV, Category),
    compatible(ActualV, GoalV).

dynamic_candidate_output_goal_match(Candidate, Port, GoalID, Category) :-
    goal_output(GoalID, GoalV, Category),
    dynamic_candidate_output_choice_value(Candidate, Port, ActualV, Category),
    compatible(ActualV, GoalV).

dynamic_initial_goal_candidate_output_mismatch(Candidate) :-
    enable_all_outputs_consumed,
    dynamic_candidate_total_output_multiplicity(Candidate, NP),
    NG = #count { G : goal_id(G) },
    NP != NG.

dynamic_candidate_output_goal_satisfies(Candidate, Port, GoalID) :-
    goal_id(GoalID),
    dynamic_candidate_output_port(Candidate, Port),
    dynamic_candidate_output_goal_match(Candidate, Port, GoalID, Category) :
        goal_output(GoalID, _, Category).

dynamic_candidate_goal_satisfies(Candidate, GoalID) :-
    dynamic_candidate_output_port(Candidate, Port),
    dynamic_candidate_output_goal_satisfies(Candidate, Port, GoalID).

dynamic_initial_goal_candidate(Candidate) :-
    dynamic_tool_candidate(Candidate, _),
    dynamic_candidate_goal_satisfies(Candidate, GoalID) : goal_id(GoalID),
    not dynamic_initial_goal_candidate_output_mismatch(Candidate).

dynamic_initial_goal_tool(Tool) :-
    dynamic_tool_candidate(Candidate, Tool),
    dynamic_initial_goal_candidate(Candidate).

dynamic_initial_goal_output_preferred_value(Candidate, Port, V, Category) :-
    dynamic_initial_goal_candidate(Candidate),
    dynamic_candidate_output_choice_value(Candidate, Port, V, Category),
    goal_output(_, GoalV, Category),
    compatible(V, GoalV),
    not dynamic_initial_goal_output_smaller_value(Candidate, Port, V, Category).

dynamic_initial_goal_output_smaller_value(Candidate, Port, V, Category) :-
    dynamic_initial_goal_candidate(Candidate),
    dynamic_candidate_output_choice_value(Candidate, Port, V, Category),
    dynamic_candidate_output_choice_value(Candidate, Port, V2, Category),
    goal_output(_, GoalV, Category),
    compatible(V, GoalV),
    compatible(V2, GoalV),
    V2 < V.

dynamic_initial_goal_output_value(Candidate, Port, V, Category) :-
    dynamic_initial_goal_output_preferred_value(Candidate, Port, V, Category).

dynamic_initial_goal_output_value(Candidate, Port, V, Category) :-
    dynamic_initial_goal_candidate(Candidate),
    dynamic_candidate_output_choice_value(Candidate, Port, V, Category),
    not goal_output(_, _, Category).

initial_goal_dim_match(GoalID, WF, GoalV, Category) :-
    goal_output(GoalID, GoalV, Category),
    available(WF),
    artifact_dim(WF, ActualV, Category),
    compatible(ActualV, GoalV).

initial_goal_dim_missing(GoalID, WF) :-
    goal_output(GoalID, GoalV, Category),
    available(WF),
    not initial_goal_dim_match(GoalID, WF, GoalV, Category).

initial_goal_satisfied_by(GoalID, WF) :-
    available(WF),
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
tool_direct_selector(Tool, Tool) :- tool(Tool).
tool_direct_selector(Tool, Selector) :-
    tool(Tool),
    tool_taxonomy(_, (Tool, Selector)).
constraint_selector_matches_tool(Sel, Tool) :-
    constraint_selector(Sel, Selector),
    constraint_selector_mode(Sel, "transitive"),
    tool_is_a(Tool, Selector).
constraint_selector_matches_tool(Sel, Tool) :-
    constraint_selector(Sel, Selector),
    constraint_selector_mode(Sel, "exact"),
    tool_direct_selector(Tool, Selector).

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
    dynamic_initial_bindable(Candidate, Port, WF).

workflow_input_profile_mismatch(WF1, WF2) :-
    workflow_input(WF1),
    workflow_input(WF2),
    artifact_dim(WF1, V, Category),
    not artifact_dim(WF2, V, Category).

equivalent_workflow_inputs(WF1, WF2) :-
    workflow_input(WF1),
    workflow_input(WF2),
    WF1 < WF2,
    not workflow_input_profile_mismatch(WF1, WF2),
    not workflow_input_profile_mismatch(WF2, WF1).

smaller_equivalent_eligible_input(Candidate, Port, WF, SmallerWF) :-
    eligible(Candidate, Port, WF),
    eligible(Candidate, Port, SmallerWF),
    equivalent_workflow_inputs(SmallerWF, WF).

