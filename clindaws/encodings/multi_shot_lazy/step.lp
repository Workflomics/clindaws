#program step(t).

time(t).

1 { occurs(t, run(Tool)) : dynamic_step_allowed_tool(Tool, t) } 1.

1 {
    occurs(t, use_dynamic_candidate(Tool, Candidate)) :
        dynamic_tool_candidate(Candidate, Tool),
        dynamic_candidate_allowed_at_step(Candidate, t)
} 1 :-
    occurs(t, run(Tool)).

occurs(t, run(Tool)) :- occurs(t, use_dynamic_candidate(Tool, _)).

1 {
    occurs(t, bind(Tool, Port, WF)) : eligible(Candidate, Port, WF);
    occurs(t, bind(Tool, Port, out(TPrev, ToolPrev, ProducerPort))) :
        produced_bind_option(
            t,
            Candidate,
            Port,
            TPrev,
            ToolPrev,
            ProducerCandidate,
            ProducerPort
        )
} 1 :-
    occurs(t, use_dynamic_candidate(Tool, Candidate)),
    dynamic_candidate_input_port(Candidate, Port).

produced_bind_option(
    t,
    Candidate,
    Port,
    TPrev,
    ToolPrev,
    ProducerCandidate,
    ProducerPort
) :-
    occurs(t, use_dynamic_candidate(_, Candidate)),
    dynamic_candidate_input_signature_id(Candidate, Port, SignatureId),
    dynamic_signature_support_class(SignatureId, SupportClassId),
    dynamic_support_class_bindable_producer_port(SupportClassId, ProducerCandidate, ProducerPort),
    occurs(TPrev, use_dynamic_candidate(ToolPrev, ProducerCandidate)),
    occurs(TPrev, output(ToolPrev, ProducerCandidate, ProducerPort)),
    TPrev < t.

chosen_produced_bind(
    t,
    Tool,
    Candidate,
    Port,
    TPrev,
    ToolPrev,
    ProducerCandidate,
    ProducerPort
) :-
    occurs(t, use_dynamic_candidate(Tool, Candidate)),
    occurs(t, bind(Tool, Port, out(TPrev, ToolPrev, ProducerPort))),
    produced_bind_option(
        t,
        Candidate,
        Port,
        TPrev,
        ToolPrev,
        ProducerCandidate,
        ProducerPort
    ).

bound_output_category_match(t, Tool, Candidate, Port, out(TPrev, ToolPrev, ProducerPort), Category) :-
    chosen_produced_bind(
        t,
        Tool,
        Candidate,
        Port,
        TPrev,
        ToolPrev,
        ProducerCandidate,
        ProducerPort
    ),
    dynamic_candidate_input_signature_id(Candidate, Port, SignatureId),
    dynamic_signature_profile(SignatureId, Category, ProfileId),
    artifact_dim(out(TPrev, ToolPrev, ProducerPort), ActualV, Category),
    dynamic_profile_accepts(ProfileId, ActualV).

:- occurs(t, use_dynamic_candidate(Tool, Candidate)),
   occurs(t, bind(Tool, Port, out(TPrev, ToolPrev, ProducerPort))),
   dynamic_candidate_input_category(Candidate, Port, Category),
   not bound_output_category_match(t, Tool, Candidate, Port, out(TPrev, ToolPrev, ProducerPort), Category).

artifact_parent(out(t, Tool, PortOut), WF) :-
    occurs(t, output(Tool, _, PortOut)),
    occurs(t, bind(Tool, _, WF)).

% Carry forward ancestors from previous steps; the artifact graph is monotone.
artifact_ancestor(out(t, Tool, PortOut), Ancestor) :-
    artifact_parent(out(t, Tool, PortOut), Ancestor).

artifact_ancestor(out(t, Tool, PortOut), Ancestor) :-
    artifact_parent(out(t, Tool, PortOut), Mid),
    artifact_ancestor(Mid, Ancestor).

% Symmetry breaking for same-signature ports using direct term order.
:- occurs(t, use_dynamic_candidate(Tool, Candidate)),
   variant_sym_ports(Candidate, PortHigh, PortLow),
   occurs(t, bind(Tool, PortLow, ArtifactLow)),
   occurs(t, bind(Tool, PortHigh, ArtifactHigh)),
   ArtifactHigh < ArtifactLow.

repeated_same_candidate_ancestor_inversion(t, T1, Tool, Candidate, Port1, WF1, T2, Port2, WF2) :-
    candidate_has_repeated_signature_inputs(Candidate),
    same_candidate_input_signature(Candidate, Port1, Candidate, Port2),
    occurs(T1, use_dynamic_candidate(Tool, Candidate)),
    occurs(T2, use_dynamic_candidate(Tool, Candidate)),
    T1 < T2,
    occurs(T1, bind(Tool, Port1, WF1)),
    occurs(T2, bind(Tool, Port2, WF2)),
    artifact_ancestor(WF1, WF2).

:- repeated_same_candidate_ancestor_inversion(t, T1, Tool, Candidate, Port1, WF1, T2, Port2, WF2).

repeated_same_candidate_symmetric_join_swap(
    t, T1, Tool, Candidate, T2, JoinTool, JoinCandidate, JoinPortHigh, JoinPortLow
) :-
    occurs(t, use_dynamic_candidate(JoinTool, JoinCandidate)),
    occurs(t, bind(JoinTool, JoinPortHigh, out(T1, Tool, _))),
    occurs(t, bind(JoinTool, JoinPortLow, out(T2, Tool, _))),
    repeated_same_candidate_join_guard(Candidate),
    join_candidate_sym_ports(JoinCandidate, JoinPortHigh, JoinPortLow),
    occurs(T1, use_dynamic_candidate(Tool, Candidate)),
    occurs(T2, use_dynamic_candidate(Tool, Candidate)),
    T1 < T2.

:- repeated_same_candidate_symmetric_join_swap(
       t, T1, Tool, Candidate, T2, JoinTool, JoinCandidate, JoinPortHigh, JoinPortLow).

:- occurs(t, use_dynamic_candidate(Tool, Candidate)),
   occurs(t, bind(Tool, Port1, WF1)),
   occurs(t, bind(Tool, Port2, WF2)),
   Port1 < Port2,
   same_candidate_input_signature(Candidate, Port1, Candidate, Port2),
   equivalent_workflow_inputs(WF2, WF1).

:- occurs(t, use_dynamic_candidate(Tool, Candidate)),
   occurs(t, bind(Tool, Port1, WF1)),
   occurs(t, bind(Tool, Port2, WF2)),
   Port1 < Port2,
   same_candidate_input_signature(Candidate, Port1, Candidate, Port2),
   smaller_equivalent_eligible_input(Candidate, Port1, WF1, _).

occurs(t, output(Tool, Candidate, Port)) :-
    occurs(t, use_dynamic_candidate(Tool, Candidate)),
    dynamic_candidate_output_port(Candidate, Port).

available(out(t, Tool, Port)) :-
    occurs(t, output(Tool, _, Port)).


1 { occurs(t, output_dim(Tool, Port, V, Category)) :
    dynamic_candidate_output_choice_value(Candidate, Port, V, Category) } 1 :-
    not query_output_filter_active(t),
    occurs(t, output(Tool, Candidate, Port)),
    dynamic_candidate_output_choice_value(Candidate, Port, _, Category).

artifact_dim(out(t, Tool, Port), V, Category) :-
    occurs(t, output_dim(Tool, Port, V, Category)).

artifact_dim(out(t, Tool, Port), V, Category) :-
    occurs(t, output(Tool, Candidate, Port)),
    dynamic_candidate_output_singleton(Candidate, Port, V, Category).

consumes_output(t, Tool) :-
    occurs(t, bind(Tool, _, out(_, _, _))).

output_used_by(TPrev, ToolPrev, Port, t, Tool, ConsumerPort) :-
    occurs(t, bind(Tool, ConsumerPort, out(TPrev, ToolPrev, Port))),
    TPrev < t.

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
    output_used_by(T1, Tool1, ProducerPort, t, Tool2, ConsumerPort),
    T1 < t.

:- occurs(t, run(Tool)), not reachable(t, Tool).

:- requirement_filtering_active, occurs(t, run(Tool)), not tool_req_compatible(Tool).
:- requirement_filtering_active, occurs(t, bind(Tool, Port, _)), not port_env_compatible(Tool, Port).

:- t > 1, occurs(t, run(_)), not occurs(t-1, run(_)).
% SAT mode does not forbid repeated tool occurrences during synthesis. The
% config-level tool_seq_repeat flag is applied later as a solution-distinction
% policy, so repeated runs remain legal here.
