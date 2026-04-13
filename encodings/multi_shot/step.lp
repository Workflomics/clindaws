#program step(t).

% ================================================================
% STEP — per-timestep planning rules
% ================================================================

% Mark this timestep
time(t).

% ----------------------------------------------------------------
% From propagation.lp — inertia for availability and dimensions
% ----------------------------------------------------------------

holds(t, avail(WF)) :-
    holds(t-1, avail(WF)).

holds(t, dim(WF, V, Cat)) :-
    holds(t-1, dim(WF, V, Cat)),
    holds(t, avail(WF)).

% ----------------------------------------------------------------
% From unit_reasoning.lp — unit carry-forward
% ----------------------------------------------------------------

holds(t, unit(WF, U)) :-
    holds(t-1, unit(WF, U)),
    holds(t, avail(WF)).

% ----------------------------------------------------------------
% From artifact_state.lp — inertia and transition firing
% ----------------------------------------------------------------

% Positive persistence
holds(t, state(Artifact, Value)) :-
    holds(t-1, state(Artifact, Value)),
    not -holds(t, state(Artifact, Value)).

% Negative persistence
-holds(t, state(Artifact, Value)) :-
    -holds(t-1, state(Artifact, Value)),
    not holds(t, state(Artifact, Value)).

% Transition firing — initiation
holds(t, state(Artifact, NewState)) :-
    initiates(Tool, Artifact, NewState),
    occurs(t, run(Tool)).

% Transition firing — termination
-holds(t, state(Artifact, OldState)) :-
    terminates(Tool, Artifact, OldState),
    occurs(t, run(Tool)).

% Consistency: artifact cannot hold two states simultaneously
:- holds(t, state(Artifact, S1)),
   holds(t, state(Artifact, S2)),
   S1 != S2.

% ----------------------------------------------------------------
% From tool_choice.lp — run choice, variant selection, port binding
% ----------------------------------------------------------------

% Run choice
{ occurs(t, run(Tool)) } :- tool(Tool), not tool_has_input(Tool).
{ occurs(t, run(Tool)) } :- tool(Tool), tool_has_input(Tool).

% Heuristic: Branch on earlier time steps first to solidify the pipeline prefix
#heuristic occurs(t, run(Tool)). [100 - t, true]

% Variant selection: exactly one input group per run
1 { occurs(t, use_variant(Tool, TI_ID)) : tool_input_variant(Tool, TI_ID) } 1 :-
    occurs(t, run(Tool)),
    tool_has_input(Tool).

occurs(t, run(Tool)) :- occurs(t, use_variant(Tool, _)).

eligible_initial_artifact(t, Port, Artifact) :-
    holds(0, avail(Artifact)),
    holds(t, avail(Artifact)),
    initial_bindable(Port, Artifact).

produced_output_candidate(t, Port, out(TPrev, ToolPrev, ProducerPort)) :-
    holds(t, avail(out(TPrev, ToolPrev, ProducerPort))),
    bindable_output_port(Port, ProducerPort).

produced_output_cat_ok(t, Port, out(TPrev, ToolPrev, ProducerPort), Cat) :-
    produced_output_candidate(t, Port, out(TPrev, ToolPrev, ProducerPort)),
    port_requires(Port, ReqV, Cat),
    artifact_dim(out(TPrev, ToolPrev, ProducerPort), V, Cat),
    compatible(V, ReqV).

eligible_produced_output(t, Port, out(TPrev, ToolPrev, ProducerPort)) :-
    produced_output_candidate(t, Port, out(TPrev, ToolPrev, ProducerPort)),
    produced_output_cat_ok(t, Port, out(TPrev, ToolPrev, ProducerPort), Cat) :
        port_category(Port, Cat).

eligible(t, Port, Artifact) :-
    eligible_initial_artifact(t, Port, Artifact).

eligible(t, Port, Artifact) :-
    eligible_produced_output(t, Port, Artifact).

% Port binding: exactly one eligible item per port
1 { occurs(t, bind(Tool, Port, WF)) : eligible(t-1, Port, WF) } 1 :-
    occurs(t, use_variant(Tool, TI_ID)),
    input_port(TI_ID, Port).

% Symmetry breaking for same-signature ports using direct term order.
:- occurs(t, use_variant(Tool, TI_ID)),
   not full_workflow_input_witnesses,
   variant_sym_ports(TI_ID, PortHigh, PortLow),
   occurs(t, bind(Tool, PortLow, ArtifactLow)),
   occurs(t, bind(Tool, PortHigh, ArtifactHigh)),
   ArtifactHigh < ArtifactLow.

% If symmetric same-signature ports both consume the same workflow input, use a
% different equivalent workflow input when one is available.
:- occurs(t, use_variant(Tool, TI_ID)),
   not full_workflow_input_witnesses,
   variant_sym_ports(TI_ID, PortHigh, PortLow),
   occurs(t, bind(Tool, PortLow, WF)),
   occurs(t, bind(Tool, PortHigh, WF)),
   workflow_input_slot(WF, Rep, _),
   eligible(t-1, PortHigh, WFAlt),
   workflow_input_slot(WFAlt, Rep, _),
   WFAlt != WF.

% Within one step, interchangeable slots of the same workflow-input class are
% used in increasing Rank order.
:- occurs(t, bind(Tool, Port, SlotHigh)),
   not full_workflow_input_witnesses,
   workflow_input_slot(SlotHigh, Rep, RankHigh),
   workflow_input_slot(SlotLow, Rep, RankLow),
   RankLow < RankHigh,
   eligible(t-1, Port, SlotLow),
   not occurs(t, bind(_, _, SlotLow)).

% Carry forward workflow-input lineage only; input-usage checks do not need the
% full produced-artifact transitive closure.
workflow_input_lineage(t, Artifact, WF) :-
    workflow_input_lineage(t-1, Artifact, WF).

workflow_input_lineage(t, out(t, Tool, PortOut), WF) :-
    occurs(t, output(Tool, _, PortOut)),
    occurs(t, bind(Tool, _, Artifact)),
    workflow_input_lineage(t-1, Artifact, WF).


% ----------------------------------------------------------------
% From output_production.lp — output port selection and dimensions
% ----------------------------------------------------------------

% Output selection: exactly one port per tool_output group
1 { occurs(t, output(Tool, TOut_ID, Port)) : output_port(TOut_ID, Port) } 1 :-
    occurs(t, run(Tool)),
    tool_has_output(Tool),
    tool_output(Tool, TOut_ID).

% Output availability
holds(t, avail(out(t, Tool, Port))) :-
    occurs(t, output(Tool, _, Port)).

% Monotonic bridge for symmetry breakers
available(out(t, Tool, Port)) :- holds(t, avail(out(t, Tool, Port))).
artifact_dim(out(t, Tool, Port), V, Cat) :- holds(t, dim(out(t, Tool, Port), V, Cat)).
artifact_unit(out(t, Tool, Port), U)   :- holds(t, unit(out(t, Tool, Port), U)).

% Output dimensions — each output category picks one retained terminal value.
% When Python precompute emitted compressed representatives, branch only over
% that reduced domain; otherwise fall back to the dense terminal expansion.
1 { occurs(t, output_dim(Tool, Port, Value, Cat)) :
    precomputed_output_port_candidate(Port, Value, Cat) } 1 :-
    occurs(t, output(Tool, _, Port)),
    precomputed_output_port_has_category(Port, Cat),
    not precomputed_output_port_singleton_terminal(Port, _, Cat).

1 { occurs(t, output_dim(Tool, Port, Value, Cat)) :
    output_port_terminal_candidate(Port, Value, Cat) } 1 :-
    occurs(t, output(Tool, _, Port)),
    not precomputed_output_port_has_category(Port, Cat),
    output_port_has_category(Port, Cat),
    not output_port_singleton_terminal(Port, _, Cat).

holds(t, dim(out(t, Tool, Port), Value, Cat)) :-
    occurs(t, output(Tool, _, Port)),
    precomputed_output_port_singleton_terminal(Port, Value, Cat).

holds(t, dim(out(t, Tool, Port), Value, Cat)) :-
    occurs(t, output(Tool, _, Port)),
    not precomputed_output_port_has_category(Port, Cat),
    output_port_singleton_terminal(Port, Value, Cat).

holds(t, dim(out(t, Tool, Port), Value, Cat)) :-
    occurs(t, output_dim(Tool, Port, Value, Cat)).

% ----------------------------------------------------------------
% From unit_reasoning.lp — unit production from tool outputs
% ----------------------------------------------------------------

% Case 1: explicit port_unit declaration
holds(t, unit(out(t, Tool, Port), U)) :-
    occurs(t, output(Tool, _, Port)),
    port_unit(Port, U).

% Case 2: strict mode — no declaration → unit is none
holds(t, unit(out(t, Tool, Port), none)) :-
    unit_mode(strict),
    occurs(t, output(Tool, _, Port)),
    not port_unit(Port, _).

% Case 3: agnostic mode — inherit from homogeneous inputs
input_root(t, Tool, Root) :-
    occurs(t, bind(Tool, _, WF)),
    artifact_unit(WF, U),
    U != none,
    unit_root(U, Root).

single_input_root(t, Tool) :-
    occurs(t, run(Tool)),
    #count { Root : input_root(t, Tool, Root) } = 1.

holds(t, unit(out(t, Tool, Port), U)) :-
    unit_mode(agnostic),
    occurs(t, output(Tool, _, Port)),
    not port_unit(Port, _),
    single_input_root(t, Tool),
    occurs(t, bind(Tool, _, WF)),
    artifact_unit(WF, U),
    U != none.

% Unit binding compatibility constraints
:- occurs(t, bind(Tool, Port, WF)),
   port_unit(Port, UPort),
   artifact_unit(WF, UData),
   UData != none,
   not units_compatible(UPort, UData).

:- unit_mode(strict),
   occurs(t, bind(Tool, Port, WF)),
   port_unit(Port, _),
   not artifact_unit(WF, _).

% ----------------------------------------------------------------
% From reachability.lp
% ----------------------------------------------------------------

consumes_output(t, Tool) :-
    occurs(t, bind(Tool, _, out(_, _, _))).

output_used_by(TPrev, ToolPrev, Port, t, Tool, ConsumerPort) :-
    occurs(t, bind(Tool, ConsumerPort, out(TPrev, ToolPrev, Port))),
    TPrev < t.

% Root: no-input tools
is_root(t, Tool) :-
    occurs(t, run(Tool)),
    not tool_has_input(Tool).

% Root: has inputs but none are tool outputs
is_root(t, Tool) :-
    occurs(t, run(Tool)),
    tool_has_input(Tool),
    not consumes_output(t, Tool).

reachable(t, Tool) :- is_root(t, Tool).

reachable(t, Tool2) :-
    reachable(T1, Tool1),
    output_used_by(T1, Tool1, Port, t, Tool2, ConsumerPort),
    T1 < t.

:- occurs(t, run(Tool)), not reachable(t, Tool).

% ----------------------------------------------------------------
% From goal.lp — derivation only (hard constraint is in check.lp)
% Atoms are timestamped with t to avoid redefinition across groundings.
% ----------------------------------------------------------------

goal_dim_match(t, GoalID, WF, GoalV, Cat) :-
    goal_output(GoalID, GoalV, Cat),
    holds(t, avail(WF)),
    artifact_dim(WF, ActualV, Cat),
    goal_requirement_accepts(goalreq(GoalID, GoalV, Cat), ActualV, Cat).

goal_dim_missing(t, GoalID, WF) :-
    goal_output(GoalID, GoalV, Cat),
    holds(t, avail(WF)),
    not goal_dim_match(t, GoalID, WF, GoalV, Cat).

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

% Goal inertia: once reached, persists
holds(t, goal) :- holds(t-1, goal).

% ----------------------------------------------------------------
% From properties.lp
% pipeline_has(Prop) is omitted here — it is non-timestamped and would
% cause redefinition errors in incremental solving. Use pipeline_has_at(t, Prop)
% for per-step tracking; a global accumulator can be added in user constraints.
% ----------------------------------------------------------------

pipeline_has_at(t, Prop) :-
    tool_property(Tool, Prop),
    occurs(t, run(Tool)).

pipeline_has_at(t, Prop) :-
    port_property(Tool, Port, _, Cat, Prop),
    occurs(t, bind(Tool, Port, WF)),
    artifact_dim(WF, ActualV, Cat),
    port_property_accepts(Tool, Port, ActualV, Cat, Prop).

% ----------------------------------------------------------------
% From requirements.lp — constraints only (static derivations in base.lp)
% ----------------------------------------------------------------

:- requirement_filtering_active, occurs(t, run(Tool)), not tool_req_compatible(Tool).
:- requirement_filtering_active, occurs(t, bind(Tool, Port, _)), not port_env_compatible(Tool, Port).

% ----------------------------------------------------------------
% Sequential workflow: at most one tool runs per timestep
% (mirrors the SAT automaton which allocates exactly one tool slot per step)
% ----------------------------------------------------------------

:- 2 { occurs(t, run(_)) }.

% Contiguous execution: no gaps in the plan — prevents shorter pipelines from
% reappearing at later horizons with empty intermediate steps, which would produce
% C(H,k) step-assignment duplicates per unique k-step sequence.
:- t > 1, occurs(t, run(_)), not occurs(t-1, run(_)).
