#program check(t).
#external query(t).

goal_dim_match(t, GoalID, WF, GoalV, Category) :-
    query(t),
    goal_output(GoalID, GoalV, Category),
    holds(t, avail(WF)),
    artifact_dim(WF, ActualV, Category),
    compatible(ActualV, GoalV).

goal_dim_missing(t, GoalID, WF) :-
    query(t),
    goal_output(GoalID, GoalV, Category),
    holds(t, avail(WF)),
    not goal_dim_match(t, GoalID, WF, GoalV, Category).

goal_satisfied_by(t, GoalID, WF) :-
    query(t),
    holds(t, avail(WF)),
    goal_output(GoalID, _, _),
    goal_dim_match(t, GoalID, WF, _, _),
    not goal_dim_missing(t, GoalID, WF).

goal_satisfied_by_output(t, GoalID, WF) :-
    goal_satisfied_by(t, GoalID, WF),
    not holds(0, avail(WF)).

goal_satisfied(t, GoalID) :-
    query(t),
    goal_output(GoalID, _, _),
    goal_satisfied_by_output(t, GoalID, WF).

holds(t, goal) :-
    query(t),
    goal_satisfied(t, GoalID) : goal_output(GoalID, _, _).

:- query(t), not holds(t, goal).
1 { chosen_goal_output(t, GoalID, WF) : goal_satisfied_by_output(t, GoalID, WF) } 1 :-
    query(t),
    goal_id(GoalID).

used_output_category_profile(t, TProd, ToolProd, Port, Category, ProfileId) :-
    query(t),
    occurs(TUse, bind(ToolUse, PortUse, out(TProd, ToolProd, Port))),
    occurs(TUse, use_dynamic_candidate(ToolUse, ConsumerCandidate)),
    dynamic_candidate_input_signature_id(ConsumerCandidate, PortUse, SignatureId),
    dynamic_signature_profile(SignatureId, Category, ProfileId).

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

candidate_output_value(t, TProd, ToolProd, ProducerCandidate, Port, V, Category) :-
    query(t),
    occurs(TProd, output(ToolProd, ProducerCandidate, Port)),
    query_output_choice_value(t, TProd, ProducerCandidate, Port, V, Category).

output_dim_has_consumer_support(t, TProd, ToolProd, Port, V, Category) :-
    query(t),
    used_output_category_profile(t, TProd, ToolProd, Port, Category, ProfileId),
    dynamic_profile_accepts(ProfileId, V).

chosen_output_goal_requirement(t, TProd, ToolProd, Port, GoalID, GoalV, Category) :-
    query(t),
    chosen_goal_output(t, GoalID, out(TProd, ToolProd, Port)),
    goal_output(GoalID, GoalV, Category).

output_dim_consumer_profile_violation(t, TProd, ToolProd, Port, V, Category, ProfileId) :-
    candidate_output_value(t, TProd, ToolProd, _, Port, V, Category),
    used_output_category_profile(t, TProd, ToolProd, Port, Category, ProfileId),
    not dynamic_profile_accepts(ProfileId, V).

output_dim_violates_consumer(t, TProd, ToolProd, ProducerCandidate, Port, V, Category) :-
    candidate_output_value(t, TProd, ToolProd, ProducerCandidate, Port, V, Category),
    output_dim_consumer_profile_violation(t, TProd, ToolProd, Port, V, Category, _).

output_dim_violates_goal(t, TProd, ToolProd, Port, V, Category) :-
    candidate_output_value(t, TProd, ToolProd, _, Port, V, Category),
    chosen_output_goal_requirement(t, TProd, ToolProd, Port, _, GoalV, Category),
    not compatible(V, GoalV).

output_dim_supported(t, TProd, ToolProd, ProducerCandidate, Port, V, Category) :-
    candidate_output_value(t, TProd, ToolProd, ProducerCandidate, Port, V, Category),
    not output_dim_violates_consumer(t, TProd, ToolProd, ProducerCandidate, Port, V, Category),
    not output_dim_violates_goal(t, TProd, ToolProd, Port, V, Category).

output_dim_smaller_supported(t, TProd, ToolProd, ProducerCandidate, Port, V, Category) :-
    output_dim_supported(t, TProd, ToolProd, ProducerCandidate, Port, V, Category),
    output_dim_supported(t, TProd, ToolProd, ProducerCandidate, Port, V2, Category),
    V2 < V.

:- query(t),
   occurs(TProd, output(ToolProd, ProducerCandidate, Port)),
   occurs(TProd, output_dim(ToolProd, Port, V, Category)),
   output_dim_supported(t, TProd, ToolProd, ProducerCandidate, Port, V, Category),
   output_dim_smaller_supported(t, TProd, ToolProd, ProducerCandidate, Port, V, Category).
