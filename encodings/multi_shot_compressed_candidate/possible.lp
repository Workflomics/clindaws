#program possible(t).
#external possible_query(t).

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
    compressed_goal_requirement_profile(GoalID, RequirementID, Category, ProfileId),
    artifact_dim(WF, ActualV, Category),
    dynamic_profile_accepts(ProfileId, ActualV).

possible_goal_dim_missing(t, GoalID, WF, RequirementID) :-
    possible_query(t),
    possible_goal_output_candidate(t, GoalID, WF),
    compressed_goal_requirement_profile(GoalID, RequirementID, _, _),
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

:- possible_query(t), not possible_goal_holds(t).
