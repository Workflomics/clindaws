#program certificate_check(t).
{ certificate_query_assumption(t) }.
certificate_query(t) :- certificate_query_assumption(t).

% ================================================================
% CERTIFICATE_CHECK — lightweight exact-horizon existence certificate
% ================================================================
% This stage validates that the currently grounded exact workflow can satisfy
% the goals and global inclusion requirements before the heavier query-horizon
% canonicalization and usefulness checks are grounded.

certificate_goal_output_candidate(t, GoalID, out(TProd, ToolProd, Port)) :-
    certificate_query(t),
    occurs(TProd, output(ToolProd, ProducerCandidate, Port)),
    dynamic_candidate_output_goal_satisfies(ProducerCandidate, Port, GoalID).

certificate_goal_dim_match(t, GoalID, WF, RequirementID) :-
    certificate_query(t),
    certificate_goal_output_candidate(t, GoalID, WF),
    optimized_goal_requirement_profile(GoalID, RequirementID, Category, ProfileId),
    artifact_dim(WF, ActualV, Category),
    dynamic_profile_accepts(ProfileId, ActualV).

certificate_goal_dim_missing(t, GoalID, WF, RequirementID) :-
    certificate_query(t),
    certificate_goal_output_candidate(t, GoalID, WF),
    optimized_goal_requirement_profile(GoalID, RequirementID, _, _),
    not certificate_goal_dim_match(t, GoalID, WF, RequirementID).

certificate_goal_satisfied_by(t, GoalID, WF) :-
    certificate_query(t),
    certificate_goal_output_candidate(t, GoalID, WF),
    goal_id(GoalID),
    certificate_goal_dim_match(t, GoalID, WF, _),
    not certificate_goal_dim_missing(t, GoalID, WF, _).

certificate_goal_satisfied(t, GoalID) :-
    certificate_query(t),
    goal_id(GoalID),
    certificate_goal_satisfied_by(t, GoalID, _).

:- certificate_query(t),
   goal_id(GoalID),
   not certificate_goal_satisfied(t, GoalID).

:- certificate_query(t),
   dynamic_must_run_tool_global(Tool),
   not occurs(_, run(Tool)).

:- certificate_query(t),
   dynamic_must_run_candidate_global(Candidate),
   not occurs(_, use_dynamic_candidate(_, Candidate)).
