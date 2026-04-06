#program step(t).

time(t).

holds(t, avail(WF)) :- holds(t-1, avail(WF)).

1 { occurs(t, run(Tool)) : lazy_step_allowed_tool(Tool, t) } 1.

1 {
    occurs(t, use_lazy_candidate(Tool, Candidate)) :
        lazy_tool_candidate(Candidate, Tool),
        lazy_candidate_allowed_at_step(Candidate, t)
} 1 :-
    occurs(t, run(Tool)).

occurs(t, run(Tool)) :- occurs(t, use_lazy_candidate(Tool, _)).

1 {
    occurs(t, bind(Tool, Port, WF)) : eligible(Candidate, Port, WF);
    occurs(t, bind(Tool, Port, out(TPrev, ToolPrev, ProducerPort))) :
        occurs(TPrev, use_lazy_candidate(ToolPrev, ProducerCandidate)),
        lazy_candidate_allowed_at_step(ProducerCandidate, TPrev),
        lazy_candidate_bindable_producer(Candidate, Port, ProducerCandidate),
        occurs(TPrev, output(ToolPrev, ProducerCandidate, ProducerPort)),
        TPrev < t,
        lazy_candidate_port_bindable(ProducerCandidate, ProducerPort, Candidate, Port)
} 1 :-
    occurs(t, use_lazy_candidate(Tool, Candidate)),
    lazy_candidate_input_port(Candidate, Port).

bound_output_category_match(t, Tool, Candidate, Port, out(TPrev, ToolPrev, ProducerPort), Category) :-
    occurs(t, use_lazy_candidate(Tool, Candidate)),
    occurs(t, bind(Tool, Port, out(TPrev, ToolPrev, ProducerPort))),
    occurs(TPrev, use_lazy_candidate(ToolPrev, ProducerCandidate)),
    lazy_candidate_allowed_at_step(ProducerCandidate, TPrev),
    lazy_candidate_bindable_producer(Candidate, Port, ProducerCandidate),
    occurs(TPrev, output(ToolPrev, ProducerCandidate, ProducerPort)),
    lazy_candidate_input_signature_id(Candidate, Port, SignatureId),
    lazy_signature_profile(SignatureId, Category, ProfileId),
    artifact_dim(out(TPrev, ToolPrev, ProducerPort), ActualV, Category),
    lazy_profile_accepts(ProfileId, ActualV).

:- occurs(t, use_lazy_candidate(Tool, Candidate)),
   occurs(t, bind(Tool, Port, out(TPrev, ToolPrev, ProducerPort))),
   lazy_candidate_input_category(Candidate, Port, Category),
   not bound_output_category_match(t, Tool, Candidate, Port, out(TPrev, ToolPrev, ProducerPort), Category).

artifact_parent(out(t, Tool, PortOut), WF) :-
    occurs(t, output(Tool, _, PortOut)),
    occurs(t, bind(Tool, _, WF)).

% Carry forward ancestors from the previous step (artifact graph is monotone).
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
    lazy_candidate_allowed_at_step(ProducerCandidate, TPrev),
    lazy_candidate_bindable_producer(Candidate, Port, ProducerCandidate),
    occurs(TPrev, output(ToolPrev, ProducerCandidate, SmallerPort)),
    same_producer_equivalent_bind_option(ProducerCandidate, ProducerPort, Candidate, Port, SmallerPort).

:- smaller_equivalent_producer_port_available(
       t, Tool, Candidate, Port, TPrev, ToolPrev, ProducerPort, _).

paired_symmetric_producers(t, T1, Tool1, Candidate1, Port1, WF1, T2, Tool2, Candidate2, Port2, WF2) :-
    T1 < T2,
    occurs(T1, use_lazy_candidate(Tool1, Candidate1)),
    occurs(T2, use_lazy_candidate(Tool2, Candidate2)),
    paired_symmetric_producer_candidate_pair(Candidate1, Candidate2),
    occurs(T1, bind(Tool1, Port1, WF1)),
    occurs(T2, bind(Tool2, Port2, WF2)),
    symmetric_candidate_input_pair(Candidate1, Port1, Candidate2, Port2),
    occurs(TJ, use_lazy_candidate(JoinTool, JoinCandidate)),
    occurs(TJ, bind(JoinTool, JoinPort1, out(T1, Tool1, _))),
    occurs(TJ, bind(JoinTool, JoinPort2, out(T2, Tool2, _))),
    TJ > T2,
    join_candidate_sym_ports(JoinCandidate, JoinPort1, JoinPort2).

:- paired_symmetric_producers(t, T1, Tool1, Candidate1, Port1, WF, T2, Tool2, Candidate2, Port2, WF).

:- paired_symmetric_producers(t, T1, Tool1, Candidate1, Port1, WF1, T2, Tool2, Candidate2, Port2, WF2),
   artifact_ancestor(t, WF1, WF2).

repeated_same_candidate_ancestor_inversion(t, T1, Tool, Candidate, Port1, WF1, T2, Port2, WF2) :-
    T1 < T2,
    occurs(T1, use_lazy_candidate(Tool, Candidate)),
    occurs(T2, use_lazy_candidate(Tool, Candidate)),
    occurs(T1, bind(Tool, Port1, WF1)),
    occurs(T2, bind(Tool, Port2, WF2)),
    same_candidate_input_signature(Candidate, Port1, Candidate, Port2),
    artifact_ancestor(t, WF1, WF2).

:- repeated_same_candidate_ancestor_inversion(t, T1, Tool, Candidate, Port1, WF1, T2, Port2, WF2).

repeated_same_candidate_symmetric_join_swap(
    t, T1, Tool, Candidate, T2, JoinTool, JoinCandidate, JoinPortHigh, JoinPortLow
) :-
    T1 < T2,
    occurs(T1, use_lazy_candidate(Tool, Candidate)),
    occurs(T2, use_lazy_candidate(Tool, Candidate)),
    occurs(TJ, use_lazy_candidate(JoinTool, JoinCandidate)),
    TJ > T2,
    join_candidate_sym_ports(JoinCandidate, JoinPortHigh, JoinPortLow),
    occurs(TJ, bind(JoinTool, JoinPortHigh, out(T1, Tool, _))),
    occurs(TJ, bind(JoinTool, JoinPortLow, out(T2, Tool, _))).

:- repeated_same_candidate_symmetric_join_swap(
       t, T1, Tool, Candidate, T2, JoinTool, JoinCandidate, JoinPortHigh, JoinPortLow).

:- occurs(t, use_lazy_candidate(Tool, Candidate)),
   occurs(t, bind(Tool, Port1, WF1)),
   occurs(t, bind(Tool, Port2, WF2)),
   Port1 < Port2,
   same_candidate_input_signature(Candidate, Port1, Candidate, Port2),
   equivalent_workflow_inputs(WF2, WF1).

repeated_candidate_chain_input(t, TPrev, PrevWF, PortPrev, Tool, Candidate, WF, Port) :-
    TPrev < t,
    occurs(TPrev, use_lazy_candidate(PrevTool, Candidate)),
    occurs(t, use_lazy_candidate(Tool, Candidate)),
    occurs(t, bind(Tool, _, out(TPrev, PrevTool, _))),
    occurs(TPrev, bind(PrevTool, PortPrev, PrevWF)),
    occurs(t, bind(Tool, Port, WF)),
    holds(0, avail(PrevWF)),
    holds(0, avail(WF)),
    same_candidate_input_signature(Candidate, PortPrev, Candidate, Port).

:- repeated_candidate_chain_input(t, TPrev, PrevWF, PortPrev, Tool, Candidate, WF, Port),
   WF < PrevWF.

:- query(t),
   occurs(t, use_lazy_candidate(Tool, Candidate)),
   occurs(t, bind(Tool, Port, WF)),
   smaller_equivalent_eligible_input(Candidate, Port, WF, _),
   1 < #count { T2, Tool2, Port2 : occurs(T2, bind(Tool2, Port2, WF)) }.

occurs(t, output(Tool, Candidate, Port)) :-
    occurs(t, use_lazy_candidate(Tool, Candidate)),
    lazy_candidate_output_port(Candidate, Port).

holds(t, avail(out(t, Tool, Port))) :-
    occurs(t, output(Tool, _, Port)).

1 { occurs(t, output_dim(Tool, Port, V, Category)) :
    lazy_candidate_output_choice_value(Candidate, Port, V, Category) } 1 :-
    not query_output_filter_active(t),
    occurs(t, output(Tool, Candidate, Port)),
    lazy_candidate_output_choice_value(Candidate, Port, _, Category).

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

% Carry forward: tools reachable at t-1 remain reachable (includes all prior steps).
reachable(t, Tool) :- reachable(t-1, Tool).

% Use reachable(t-1, ...) instead of free-variable T1 range — semantically equivalent
% because carry-forward already accumulates all prior reachable sets.
reachable(t, Tool2) :-
    reachable(t-1, Tool1),
    occurs(T1, output(Tool1, _, Port)),
    occurs(t, bind(Tool2, _, out(T1, Tool1, Port))),
    T1 < t.

:- occurs(t, run(Tool)), not reachable(t, Tool).

goal_dim_match(t, GoalID, WF, GoalV, Category) :-
    goal_output(GoalID, GoalV, Category),
    holds(t, avail(WF)),
    artifact_dim(WF, ActualV, Category),
    compatible(ActualV, GoalV).

goal_dim_missing(t, GoalID, WF) :-
    goal_output(GoalID, GoalV, Category),
    holds(t, avail(WF)),
    not goal_dim_match(t, GoalID, WF, GoalV, Category).

goal_satisfied_by(t, GoalID, WF) :-
    holds(t, avail(WF)),
    goal_output(GoalID, _, _),
    goal_dim_match(t, GoalID, WF, _, _),
    not goal_dim_missing(t, GoalID, WF).

goal_satisfied_by_output(t, GoalID, WF) :-
    goal_satisfied_by(t, GoalID, WF),
    not holds(0, avail(WF)).

goal_satisfied(t, GoalID) :-
    goal_output(GoalID, _, _),
    goal_satisfied_by_output(t, GoalID, WF).

holds(t, goal) :-
    goal_satisfied(t, GoalID) : goal_output(GoalID, _, _).

holds(t, goal) :- holds(t-1, goal).

:- requirement_filtering_active, occurs(t, run(Tool)), not tool_req_compatible(Tool).
:- requirement_filtering_active, occurs(t, bind(Tool, Port, _)), not port_env_compatible(Tool, Port).

:- t > 1, occurs(t, run(_)), not occurs(t-1, run(_)).
% SAT mode does not forbid repeated tool occurrences during synthesis. The
% config-level tool_seq_repeat flag is applied later as a solution-distinction
% policy, so repeated runs remain legal here.
