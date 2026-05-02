#program possible_fast(t).

time(t).

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

fast_structurally_supportable(t, Candidate) :-
    dynamic_structurally_supportable_candidate_at_horizon(Candidate, t).

fast_candidate_precedence_window(t, ProducerCandidate, ConsumerCandidate) :-
    fast_structurally_supportable(t, ProducerCandidate),
    fast_structurally_supportable(t, ConsumerCandidate),
    dynamic_candidate_min_step(ProducerCandidate, ProducerMinStep),
    dynamic_candidate_max_step(ConsumerCandidate, ConsumerMaxStep),
    t <= ConsumerMaxStep,
    ProducerMinStep < t.

fast_candidate_precedence_window(t, ProducerCandidate, ConsumerCandidate) :-
    fast_structurally_supportable(t, ProducerCandidate),
    fast_structurally_supportable(t, ConsumerCandidate),
    dynamic_candidate_min_step(ProducerCandidate, ProducerMinStep),
    dynamic_candidate_max_step(ConsumerCandidate, ConsumerMaxStep),
    ConsumerMaxStep < t,
    ProducerMinStep < ConsumerMaxStep.

fast_goal_stage_satisfied(t, GoalID) :-
    possible_enforce_goal(t),
    fast_structurally_supportable(t, Candidate),
    dynamic_candidate_output_goal_satisfies(Candidate, _, GoalID).

fast_forced_binding_available(t, Candidate, Port) :-
    fast_structurally_supportable(t, Candidate),
    dynamic_forced_produced_bind_at_step(Candidate, Port, ProducerCandidate, _, _),
    fast_structurally_supportable(t, ProducerCandidate),
    fast_candidate_precedence_window(t, ProducerCandidate, Candidate).

fast_tool_used(t, Tool) :-
    dynamic_goal_support_tool_at_horizon(Tool, t),
    fast_structurally_supportable(t, Candidate),
    dynamic_tool_candidate(Candidate, Tool).

:- possible_query(t),
   possible_enforce_goal(t),
   goal_id(GoalID),
   not fast_goal_stage_satisfied(t, GoalID).

:- possible_query(t),
   possible_enforce_input_support(t),
   dynamic_structurally_unsupported_input_at_horizon(Candidate, Port, t).

:- possible_query(t),
   possible_enforce_forced_binding(t),
   dynamic_goal_support_input_at_horizon(Candidate, Port, t),
   dynamic_forced_produced_bind_at_step(Candidate, Port, _, _, _),
   not fast_forced_binding_available(t, Candidate, Port).

:- possible_query(t),
   possible_enforce_must_run_tool(t),
   dynamic_must_run_tool_global(Tool),
   not dynamic_goal_support_tool_at_horizon(Tool, t).

:- possible_query(t),
   possible_enforce_must_run_tool(t),
   dynamic_goal_support_tool_at_horizon(Tool, t),
   dynamic_must_run_tool_global(Tool),
   not fast_tool_used(t, Tool).

:- possible_query(t),
   possible_enforce_must_run_candidate(t),
   dynamic_must_run_candidate_global(Candidate),
   not dynamic_goal_support_candidate_at_horizon(Candidate, t).

:- possible_query(t),
   possible_enforce_must_run_candidate(t),
   dynamic_must_run_candidate_global(Candidate),
   dynamic_goal_support_candidate_at_horizon(Candidate, t),
   not fast_structurally_supportable(t, Candidate).
