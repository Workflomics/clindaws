#program check(t).
#external query(t).

:- query(t), not holds(t, goal).
:- query(t), exact_horizon_mode, t > 1, holds(t-1, goal).

1 { chosen_goal_output(t, GoalID, WF) : goal_satisfied_by_output(t, GoalID, WF) } 1 :-
    query(t),
    goal_id(GoalID).

output_dim_violates_consumer(t, TProd, ToolProd, ProducerCandidate, Port, V, Category) :-
    query(t),
    occurs(TProd, output(ToolProd, ProducerCandidate, Port)),
    lazy_candidate_output_choice_value(ProducerCandidate, Port, V, Category),
    occurs(TUse, bind(ToolUse, PortUse, out(TProd, ToolProd, Port))),
    occurs(TUse, use_lazy_candidate(ToolUse, ConsumerCandidate)),
    lazy_candidate_input_signature_id(ConsumerCandidate, PortUse, SignatureId),
    lazy_candidate_input_category(ConsumerCandidate, PortUse, Category),
    not lazy_bound_output_support(ProducerCandidate, Port, SignatureId, V, Category).

output_dim_violates_goal(t, TProd, ToolProd, Port, V, Category) :-
    query(t),
    occurs(TProd, output(ToolProd, ProducerCandidate, Port)),
    lazy_candidate_output_choice_value(ProducerCandidate, Port, V, Category),
    chosen_goal_output(t, GoalID, out(TProd, ToolProd, Port)),
    goal_output(GoalID, GoalV, Category),
    not compatible(V, GoalV).

output_dim_supported(t, TProd, ToolProd, ProducerCandidate, Port, V, Category) :-
    query(t),
    occurs(TProd, output(ToolProd, ProducerCandidate, Port)),
    lazy_candidate_output_choice_value(ProducerCandidate, Port, V, Category),
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
