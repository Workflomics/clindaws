#program step_initial(t).

time(t).

holds(t, avail(WF)) :- holds(t-1, avail(WF)).

initial_goal_filter_active(t) :-
    t = 1,
    query(t),
    exact_horizon_mode.

1 { occurs(t, run(Tool)) : lazy_initial_goal_tool(Tool) } 1 :-
    initial_goal_filter_active(t).

1 { occurs(t, run(Tool)) : lazy_step_allowed_tool(Tool, t) } 1 :-
    not initial_goal_filter_active(t).
:- initial_goal_filter_active(t), not occurs(t, run(_)).

1 { occurs(t, use_lazy_candidate(Tool, Candidate)) :
    lazy_tool_candidate(Candidate, Tool),
    lazy_initial_goal_candidate(Candidate),
    lazy_candidate_allowed_at_step(Candidate, t) } 1 :-
    initial_goal_filter_active(t),
    occurs(t, run(Tool)).

1 {
    occurs(t, use_lazy_candidate(Tool, Candidate)) :
        lazy_tool_candidate(Candidate, Tool),
        lazy_candidate_allowed_at_step(Candidate, t)
} 1 :-
    not initial_goal_filter_active(t),
    occurs(t, run(Tool)).

occurs(t, run(Tool)) :- occurs(t, use_lazy_candidate(Tool, _)).

:- initial_goal_filter_active(t),
   occurs(t, use_lazy_candidate(_, Candidate)),
   enable_all_outputs_consumed,
   lazy_candidate_total_output_multiplicity(Candidate, NP),
   NG = #count { GoalID : goal_id(GoalID) },
   NP != NG.

1 { occurs(t, bind(Tool, Port, WF)) : lazy_initial_bindable(Candidate, Port, WF) } 1 :-
    occurs(t, use_lazy_candidate(Tool, Candidate)),
    lazy_candidate_input_port(Candidate, Port).

artifact_parent(out(t, Tool, PortOut), WF) :-
    occurs(t, output(Tool, _, PortOut)),
    occurs(t, bind(Tool, _, WF)).

% Carry forward ancestors from the previous step (artifact graph is monotone).
% At t=1, artifact_ancestor(0, ...) is empty so this is a no-op; base rules fire.
artifact_ancestor(t, Artifact, Ancestor) :-
    artifact_ancestor(t-1, Artifact, Ancestor).

% Only compute fresh entries for artifacts produced at THIS step.
artifact_ancestor(t, out(t, Tool, PortOut), Ancestor) :-
    artifact_parent(out(t, Tool, PortOut), Ancestor).

artifact_ancestor(t, out(t, Tool, PortOut), Ancestor) :-
    artifact_parent(out(t, Tool, PortOut), Mid),
    artifact_ancestor(t, Mid, Ancestor).

smaller_equivalent_producer_port_available(t, Tool, Candidate, Port, TPrev, ToolPrev, ProducerPort, SmallerPort) :-
    occurs(t, use_lazy_candidate(Tool, Candidate)),
    occurs(t, bind(Tool, Port, out(TPrev, ToolPrev, ProducerPort))),
    occurs(TPrev, use_lazy_candidate(ToolPrev, ProducerCandidate)),
    occurs(TPrev, output(ToolPrev, ProducerCandidate, SmallerPort)),
    same_producer_equivalent_bind_option(ProducerCandidate, ProducerPort, Candidate, Port, SmallerPort).

:- smaller_equivalent_producer_port_available(
       t, Tool, Candidate, Port, TPrev, ToolPrev, ProducerPort, _).

% paired_symmetric_producers and repeated_same_candidate_symmetric_join_swap are
% omitted here: both require T1 < T2 < TJ, which needs at least 3 time steps.
% At t=1 these rules are dead code. They are enforced in step.lp at t>=3.

repeated_same_candidate_ancestor_inversion(t, T1, Tool, Candidate, Port1, WF1, T2, Port2, WF2) :-
    candidate_has_repeated_signature_inputs(Candidate),
    same_candidate_input_signature(Candidate, Port1, Candidate, Port2),
    occurs(T1, use_lazy_candidate(Tool, Candidate)),
    occurs(T2, use_lazy_candidate(Tool, Candidate)),
    T1 < T2,
    occurs(T1, bind(Tool, Port1, WF1)),
    occurs(T2, bind(Tool, Port2, WF2)),
    artifact_ancestor(t, WF1, WF2).

:- repeated_same_candidate_ancestor_inversion(t, T1, Tool, Candidate, Port1, WF1, T2, Port2, WF2).

:- occurs(t, use_lazy_candidate(Tool, Candidate)),
   occurs(t, bind(Tool, Port1, WF1)),
   occurs(t, bind(Tool, Port2, WF2)),
   Port1 < Port2,
   same_candidate_input_signature(Candidate, Port1, Candidate, Port2),
   equivalent_workflow_inputs(WF2, WF1).

occurs(t, output(Tool, Candidate, Port)) :-
    occurs(t, use_lazy_candidate(Tool, Candidate)),
    lazy_candidate_output_port(Candidate, Port).

holds(t, avail(out(t, Tool, Port))) :-
    occurs(t, output(Tool, _, Port)).

1 { occurs(t, output_dim(Tool, Port, V, Category)) :
    lazy_initial_goal_output_value(Candidate, Port, V, Category) } 1 :-
    initial_goal_filter_active(t),
    occurs(t, output(Tool, Candidate, Port)),
    lazy_initial_goal_output_value(Candidate, Port, _, Category).

1 { occurs(t, output_dim(Tool, Port, V, Category)) :
    lazy_candidate_output_choice_value(Candidate, Port, V, Category) } 1 :-
    occurs(t, output(Tool, Candidate, Port)),
    lazy_candidate_output_choice_value(Candidate, Port, _, Category),
    not initial_goal_filter_active(t).

artifact_dim(out(t, Tool, Port), V, Category) :-
    occurs(t, output_dim(Tool, Port, V, Category)).

artifact_dim(out(t, Tool, Port), V, Category) :-
    occurs(t, output(Tool, Candidate, Port)),
    lazy_candidate_output_singleton(Candidate, Port, V, Category).

consumes_output(t, Tool) :-
    occurs(t, bind(Tool, _, out(_, _, _))).

is_root(t, Tool) :-
    occurs(t, run(Tool)),
    not tool_has_input(Tool).

is_root(t, Tool) :-
    occurs(t, run(Tool)),
    tool_has_input(Tool),
    not consumes_output(t, Tool).

reachable(t, Tool) :- is_root(t, Tool).

:- requirement_filtering_active, occurs(t, run(Tool)), not tool_req_compatible(Tool).
:- requirement_filtering_active, occurs(t, bind(Tool, Port, _)), not port_env_compatible(Tool, Port).

:- occurs(t, run(Tool)), not reachable(t, Tool).
