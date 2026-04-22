#program check(t).
{ query_assumption(t) }.
query(t) :- query_assumption(t).

% ================================================================
% CHECK — volatile goal and output-choice validation at horizon t
% ================================================================
% Goal satisfaction still happens at the query horizon, but consumer/goal
% support is now checked against precomputed profiles and retained output-choice
% values instead of the larger direct terminal domain used by plain multi-shot.

goal_output_candidate(t, GoalID, out(TProd, ToolProd, Port)) :-
    query(t),
    occurs(TProd, output(ToolProd, ProducerCandidate, Port)),
    dynamic_candidate_output_goal_satisfies(ProducerCandidate, Port, GoalID).

goal_dim_match(t, GoalID, WF, RequirementID) :-
    query(t),
    goal_output_candidate(t, GoalID, WF),
    optimized_goal_requirement_profile(GoalID, RequirementID, Category, ProfileId),
    artifact_dim(WF, ActualV, Category),
    dynamic_profile_accepts(ProfileId, ActualV).

goal_dim_missing(t, GoalID, WF, RequirementID) :-
    query(t),
    goal_output_candidate(t, GoalID, WF),
    optimized_goal_requirement_profile(GoalID, RequirementID, _, _),
    not goal_dim_match(t, GoalID, WF, RequirementID).

goal_satisfied_by(t, GoalID, WF) :-
    query(t),
    goal_output_candidate(t, GoalID, WF),
    goal_id(GoalID),
    goal_dim_match(t, GoalID, WF, _),
    not goal_dim_missing(t, GoalID, WF, _).

goal_satisfied_by_output(t, GoalID, WF) :-
    goal_satisfied_by(t, GoalID, WF).

goal_satisfied(t, GoalID) :-
    query(t),
    goal_id(GoalID),
    goal_satisfied_by_output(t, GoalID, WF).

holds(t, goal) :-
    query(t),
    goal_satisfied(t, GoalID) : goal_id(GoalID).

:- query(t), not holds(t, goal).
:- query(t),
   dynamic_must_run_tool_global(Tool),
   not occurs(_, run(Tool)).
:- query(t),
   dynamic_must_run_candidate_global(Candidate),
   not occurs(_, use_dynamic_candidate(_, Candidate)).
1 { chosen_goal_output(t, GoalID, WF) : goal_satisfied_by_output(t, GoalID, WF) } 1 :-
    query(t),
    goal_id(GoalID).

consumer_bound_output(
    t,
    TUse,
    ToolUse,
    ConsumerCandidate,
    PortUse,
    TProd,
    ToolProd,
    Port
) :-
    query(t),
    occurs(TUse, bind(ToolUse, PortUse, out(TProd, ToolProd, Port))),
    occurs(TUse, use_dynamic_candidate(ToolUse, ConsumerCandidate)).

used_output_category_profile(t, TProd, ToolProd, Port, Category, ProfileId) :-
    consumer_bound_output(
        t,
        _,
        _,
        ConsumerCandidate,
        PortUse,
        TProd,
        ToolProd,
        Port
    ),
    optimized_candidate_input_profile(ConsumerCandidate, PortUse, Category, ProfileId).

chosen_output_goal_requirement_profile(
    t,
    TProd,
    ToolProd,
    Port,
    GoalID,
    RequirementID,
    Category,
    ProfileId
) :-
    chosen_goal_output(t, GoalID, out(TProd, ToolProd, Port)),
    optimized_goal_requirement_profile(GoalID, RequirementID, Category, ProfileId).

query_output_choice_value(t, TProd, ProducerCandidate, Port, V, Category) :-
    query_output_filter_active(t),
    TProd = t,
    occurs(TProd, output(_, ProducerCandidate, Port)),
    dynamic_query_output_value(ProducerCandidate, Port, V, Category).

query_output_choice_value(t, TProd, ProducerCandidate, Port, V, Category) :-
    occurs(TProd, output(_, ProducerCandidate, Port)),
    TProd != t,
    dynamic_candidate_output_choice_value(ProducerCandidate, Port, V, Category).

query_output_choice_value(t, TProd, ProducerCandidate, Port, V, Category) :-
    occurs(TProd, output(_, ProducerCandidate, Port)),
    not query_output_filter_active(t),
    dynamic_candidate_output_choice_value(ProducerCandidate, Port, V, Category).

check_relevant_output_category(t, TProd, ToolProd, Port, Category) :-
    query(t),
    occurs(TProd, output(ToolProd, ProducerCandidate, Port)),
    dynamic_check_relevant_output_category_at_horizon(ProducerCandidate, Port, Category, t).

required_output_profile_class(t, TProd, ToolProd, Port, Category, ProfileClassId) :-
    query(t),
    occurs(TProd, output(ToolProd, ProducerCandidate, Port)),
    dynamic_check_required_profile_class_at_horizon(ProducerCandidate, Port, Category, ProfileClassId, t).

required_output_profile(t, TProd, ToolProd, Port, Category, ProfileId) :-
    required_output_profile_class(t, TProd, ToolProd, Port, Category, ProfileClassId),
    dynamic_check_profile_class_member(ProfileClassId, ProfileId).

required_output_profile(t, TProd, ToolProd, Port, Category, ProfileId) :-
    chosen_output_goal_requirement_profile(t, TProd, ToolProd, Port, _, _, Category, ProfileId).

% Keep exact support selection, but only materialize values for output
% categories that can matter at the current horizon.
output_port_value(t, TProd, ToolProd, Port, V, Category) :-
    query(t),
    occurs(TProd, output(ToolProd, ProducerCandidate, Port)),
    check_relevant_output_category(t, TProd, ToolProd, Port, Category),
    query_output_choice_value(t, TProd, ProducerCandidate, Port, V, Category).

output_value_profile_violation(t, TProd, ToolProd, Port, V, Category, ProfileId) :-
    output_port_value(t, TProd, ToolProd, Port, V, Category),
    required_output_profile(t, TProd, ToolProd, Port, Category, ProfileId),
    not dynamic_profile_accepts(ProfileId, V).

output_dim_violates_consumer(t, TProd, ToolProd, Port, V, Category) :-
    used_output_category_profile(t, TProd, ToolProd, Port, Category, ProfileId),
    output_value_profile_violation(t, TProd, ToolProd, Port, V, Category, ProfileId).

output_dim_violates_goal(t, TProd, ToolProd, Port, V, Category) :-
    chosen_output_goal_requirement_profile(t, TProd, ToolProd, Port, _, _, Category, ProfileId),
    output_value_profile_violation(t, TProd, ToolProd, Port, V, Category, ProfileId).

output_dim_supported(t, TProd, ToolProd, Port, V, Category) :-
    output_port_value(t, TProd, ToolProd, Port, V, Category),
    not output_dim_violates_consumer(t, TProd, ToolProd, Port, V, Category),
    not output_dim_violates_goal(t, TProd, ToolProd, Port, V, Category).

output_dim_smaller_supported(t, TProd, ToolProd, Port, V, Category) :-
    output_dim_supported(t, TProd, ToolProd, Port, V, Category),
    output_dim_supported(t, TProd, ToolProd, Port, V2, Category),
    V2 < V.

:- query(t),
   occurs(TProd, output(ToolProd, ProducerCandidate, Port)),
   occurs(TProd, output_dim(ToolProd, Port, V, Category)),
   output_dim_supported(t, TProd, ToolProd, Port, V, Category),
   output_dim_smaller_supported(t, TProd, ToolProd, Port, V, Category).
