#program possible(t).
{ possible_query_assumption(t) }.
possible_query(t) :- possible_query_assumption(t).
{ possible_enforce_goal_assumption(t) }.
possible_enforce_goal(t) :- possible_enforce_goal_assumption(t).
{ possible_enforce_input_support_assumption(t) }.
possible_enforce_input_support(t) :- possible_enforce_input_support_assumption(t).
{ possible_enforce_forced_binding_assumption(t) }.
possible_enforce_forced_binding(t) :- possible_enforce_forced_binding_assumption(t).
{ possible_enforce_must_run_tool_assumption(t) }.
possible_enforce_must_run_tool(t) :- possible_enforce_must_run_tool_assumption(t).
{ possible_enforce_must_run_candidate_assumption(t) }.
possible_enforce_must_run_candidate(t) :- possible_enforce_must_run_candidate_assumption(t).

% ================================================================
% POSSIBLE — lightweight horizon feasibility check
% ================================================================
% This phase is intentionally weaker than full validation. It proves only that
% the current horizon can still support all goals under the already grounded
% step choices. If this fails, the full solve can be skipped safely.

possible_goal_output_candidate(t, GoalID, out(TProd, ToolProd, Port)) :-
    possible_query(t),
    occurs(TProd, output(ToolProd, ProducerCandidate, Port)),
    dynamic_candidate_output_goal_satisfies(ProducerCandidate, Port, GoalID).

possible_goal_dim_match(t, GoalID, WF, RequirementID) :-
    possible_query(t),
    possible_goal_output_candidate(t, GoalID, WF),
    optimized_goal_requirement_profile(GoalID, RequirementID, Category, ProfileId),
    artifact_dim(WF, ActualV, Category),
    dynamic_profile_accepts(ProfileId, ActualV).

possible_goal_dim_missing(t, GoalID, WF, RequirementID) :-
    possible_query(t),
    possible_goal_output_candidate(t, GoalID, WF),
    optimized_goal_requirement_profile(GoalID, RequirementID, _, _),
    not possible_goal_dim_match(t, GoalID, WF, RequirementID).

possible_goal_satisfied_by(t, GoalID, WF) :-
    possible_query(t),
    possible_goal_output_candidate(t, GoalID, WF),
    goal_id(GoalID),
    possible_goal_dim_match(t, GoalID, WF, _),
    not possible_goal_dim_missing(t, GoalID, WF, _).

possible_goal_satisfied(t, GoalID) :-
    possible_query(t),
    goal_id(GoalID),
    possible_goal_satisfied_by(t, GoalID, WF).

possible_goal_holds(t) :-
    possible_query(t),
    possible_goal_satisfied(t, GoalID) : goal_id(GoalID).

possible_violation_missing_goal(t) :-
    possible_query(t),
    not possible_goal_holds(t).

possible_input_structurally_supported(t, TUse, Candidate, Port) :-
    possible_query(t),
    occurs(TUse, use_dynamic_candidate(_, Candidate)),
    dynamic_initial_bindable(Candidate, Port, _).

possible_input_structurally_supported(t, TUse, Candidate, Port) :-
    possible_query(t),
    occurs(TUse, use_dynamic_candidate(_, Candidate)),
    dynamic_candidate_input_association_class(Candidate, Port, AssociationClassId),
    dynamic_association_class_bindable_producer_port(AssociationClassId, ProducerCandidate, ProducerPort),
    occurs(TPrev, use_dynamic_candidate(_, ProducerCandidate)),
    occurs(TPrev, output(_, ProducerCandidate, ProducerPort)),
    TPrev < TUse.

possible_violation_missing_input_support(t, TUse, Candidate, Port) :-
    possible_query(t),
    occurs(TUse, use_dynamic_candidate(_, Candidate)),
    dynamic_candidate_input_port(Candidate, Port),
    not possible_input_structurally_supported(t, TUse, Candidate, Port).

possible_forced_binding_available(t, TUse, Candidate, Port) :-
    possible_query(t),
    occurs(TUse, use_dynamic_candidate(_, Candidate)),
    dynamic_forced_produced_bind_at_step(Candidate, Port, ProducerCandidate, ProducerPort, TUse),
    occurs(TPrev, use_dynamic_candidate(_, ProducerCandidate)),
    occurs(TPrev, output(_, ProducerCandidate, ProducerPort)),
    TPrev < TUse.

possible_violation_forced_binding(t, TUse, Candidate, Port, ProducerCandidate, ProducerPort) :-
    possible_query(t),
    occurs(TUse, use_dynamic_candidate(_, Candidate)),
    dynamic_forced_produced_bind_at_step(Candidate, Port, ProducerCandidate, ProducerPort, TUse),
    not possible_forced_binding_available(t, TUse, Candidate, Port).

possible_violation_missing_must_run_tool(t, Tool) :-
    possible_query(t),
    dynamic_must_run_tool_global(Tool),
    not occurs(_, run(Tool)).

possible_violation_missing_must_run_candidate(t, Candidate) :-
    possible_query(t),
    dynamic_must_run_candidate_global(Candidate),
    not occurs(_, use_dynamic_candidate(_, Candidate)).

:- possible_query(t),
   possible_enforce_input_support(t),
   possible_violation_missing_input_support(t, _, _, _).

:- possible_query(t),
   possible_enforce_forced_binding(t),
   possible_violation_forced_binding(t, _, _, _, _, _).

:- possible_query(t),
   possible_enforce_must_run_tool(t),
   possible_violation_missing_must_run_tool(t, _).
:- possible_query(t),
   possible_enforce_must_run_candidate(t),
   possible_violation_missing_must_run_candidate(t, _).
:- possible_query(t),
   possible_enforce_goal(t),
   possible_violation_missing_goal(t).

#show possible_violation_missing_goal/1.
#show possible_violation_missing_input_support/4.
#show possible_violation_forced_binding/6.
#show possible_violation_missing_must_run_tool/2.
#show possible_violation_missing_must_run_candidate/2.
