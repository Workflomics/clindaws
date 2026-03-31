#program step_query(t).
#external query(t).

query_output_filter_active(t) :-
    query(t),
    exact_horizon_mode.

:- query(t), occurs(t, run(Tool)), not lazy_query_goal_tool(Tool).
:- query(t), occurs(t, use_lazy_candidate(_, Candidate)), not lazy_query_goal_candidate(Candidate).

1 { occurs(t, output_dim(Tool, Port, V, Category)) :
    lazy_query_output_value(Candidate, Port, V, Category) } 1 :-
    query_output_filter_active(t),
    occurs(t, output(Tool, Candidate, Port)),
    lazy_query_output_value(Candidate, Port, _, Category).

query_bound_output_value_bind_match(t, TPrev, ToolPrev, ProducerPort, V, Category, ConsumerPort) :-
    query(t),
    occurs(TPrev, output(ToolPrev, ProducerCandidate, ProducerPort)),
    occurs(t, use_lazy_candidate(ConsumerTool, ConsumerCandidate)),
    occurs(t, bind(ConsumerTool, ConsumerPort, out(TPrev, ToolPrev, ProducerPort))),
    lazy_candidate_input_signature_id(ConsumerCandidate, ConsumerPort, SignatureId),
    lazy_query_bound_output_support(ProducerCandidate, ProducerPort, SignatureId, V, Category).

query_bound_output_value_missing_support(t, TPrev, ToolPrev, ProducerPort, V, Category) :-
    query(t),
    occurs(TPrev, output(ToolPrev, ProducerCandidate, ProducerPort)),
    lazy_candidate_output_choice_value(ProducerCandidate, ProducerPort, V, Category),
    occurs(t, use_lazy_candidate(ConsumerTool, ConsumerCandidate)),
    occurs(t, bind(ConsumerTool, ConsumerPort, out(TPrev, ToolPrev, ProducerPort))),
    lazy_candidate_input_category(ConsumerCandidate, ConsumerPort, Category),
    not query_bound_output_value_bind_match(t, TPrev, ToolPrev, ProducerPort, V, Category, ConsumerPort).

query_bound_output_value_admissible(t, TPrev, ToolPrev, ProducerPort, V, Category) :-
    query(t),
    occurs(TPrev, output(ToolPrev, ProducerCandidate, ProducerPort)),
    lazy_candidate_output_choice_value(ProducerCandidate, ProducerPort, V, Category),
    not query_bound_output_value_missing_support(t, TPrev, ToolPrev, ProducerPort, V, Category).

:- query(t),
   occurs(t, bind(_, _, out(TPrev, ToolPrev, ProducerPort))),
   occurs(TPrev, output(ToolPrev, ProducerCandidate, ProducerPort)),
   occurs(TPrev, output_dim(ToolPrev, ProducerPort, V, Category)),
   lazy_candidate_output_choice_value(ProducerCandidate, ProducerPort, V, Category),
   not query_bound_output_value_admissible(t, TPrev, ToolPrev, ProducerPort, V, Category).

:- query(t),
   occurs(t, bind(_, _, out(TPrev, ToolPrev, ProducerPort))),
   occurs(TPrev, output(ToolPrev, ProducerCandidate, ProducerPort)),
   occurs(TPrev, output_dim(ToolPrev, ProducerPort, V, Category)),
   lazy_candidate_output_choice_value(ProducerCandidate, ProducerPort, V, Category),
   query_bound_output_value_admissible(t, TPrev, ToolPrev, ProducerPort, V2, Category),
   V2 < V.
