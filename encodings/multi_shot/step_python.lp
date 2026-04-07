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
{ occurs(t, run(Tool)) } :- precomputed_tool_allowed_at_step(Tool, t), not tool_has_input(Tool).
{ occurs(t, run(Tool)) } :- precomputed_tool_allowed_at_step(Tool, t), tool_has_input(Tool).

% Variant selection: exactly one input group per run
1 { occurs(t, use_variant(Tool, TI_ID)) : tool_input_variant(Tool, TI_ID) } 1 :-
    occurs(t, run(Tool)),
    tool_has_input(Tool).

occurs(t, run(Tool)) :- occurs(t, use_variant(Tool, _)).

% Port satisfaction per category
port_cat_ok(t, Port, WF, Cat) :-
    precomputed_base_artifact_bindable(Port, WF),
    holds(t, avail(WF)),
    port_requires(Port, ReqV, Cat),
    holds(t, dim(WF, V, Cat)),
    compatible(V, ReqV).

port_cat_ok(t, Port, out(TPrev, ToolPrev, ProducerPort), Cat) :-
    precomputed_output_port_bindable(Port, ProducerPort),
    holds(t, avail(out(TPrev, ToolPrev, ProducerPort))),
    port_requires(Port, ReqV, Cat),
    holds(t, dim(out(TPrev, ToolPrev, ProducerPort), V, Cat)),
    compatible(V, ReqV).

% Eligible item: all categories satisfied
eligible(t, Port, WF) :-
    precomputed_base_artifact_bindable(Port, WF),
    holds(t, avail(WF)),
    port_cat_ok(t, Port, WF, Cat) : port_requires(Port, _, Cat).

eligible(t, Port, out(TPrev, ToolPrev, ProducerPort)) :-
    precomputed_output_port_bindable(Port, ProducerPort),
    holds(t, avail(out(TPrev, ToolPrev, ProducerPort))),
    port_cat_ok(t, Port, out(TPrev, ToolPrev, ProducerPort), Cat) : port_requires(Port, _, Cat).

% Port binding: exactly one eligible item per port
1 { occurs(t, bind(Tool, Port, WF)) : eligible(t-1, Port, WF) } 1 :-
    occurs(t, use_variant(Tool, TI_ID)),
    input_port(TI_ID, Port).

% Canonical artifact order for same-signature ports. This removes duplicate
% witnesses where identical ports are filled by the same artifacts in a
% different order, including mixed workflow-input / produced-output swaps.
artifact_precedes(t, WF1, WF2) :-
    workflow_input(WF1),
    workflow_input(WF2),
    WF1 < WF2.

artifact_precedes(t, WF, Slot) :-
    workflow_input(WF),
    workflow_input_slot(Slot, Rep, _),
    WF < Rep.

artifact_precedes(t, Slot1, Slot2) :-
    workflow_input_slot(Slot1, Rep1, Rank1),
    workflow_input_slot(Slot2, Rep2, Rank2),
    Rep1 < Rep2.

artifact_precedes(t, Slot, WF) :-
    workflow_input_slot(Slot, Rep, _),
    workflow_input(WF),
    Rep < WF.

artifact_precedes(t, Slot1, Slot2) :-
    workflow_input_slot(Slot1, Rep, Rank1),
    workflow_input_slot(Slot2, Rep, Rank2),
    Rank1 < Rank2.

artifact_precedes(t, Slot, out(T, Tool, Port)) :-
    workflow_input_slot(Slot, _, _),
    occurs(T, output(Tool, _, Port)).

artifact_precedes(t, WF, out(T, Tool, Port)) :-
    workflow_input(WF),
    occurs(T, output(Tool, _, Port)).

artifact_precedes(t, out(T1, Tool1, Port1), out(T2, Tool2, Port2)) :-
    occurs(T1, output(Tool1, _, Port1)),
    occurs(T2, output(Tool2, _, Port2)),
    T1 < T2.

artifact_precedes(t, out(T, Tool, Port1), out(T, Tool, Port2)) :-
    occurs(T, output(Tool, _, Port1)),
    occurs(T, output(Tool, _, Port2)),
    Port1 < Port2,
    occurs(T, run(Tool)).

:- occurs(t, use_variant(Tool, TI_ID)),
   variant_sym_ports(TI_ID, PortHigh, PortLow),
   occurs(t, bind(Tool, PortLow, ArtifactLow)),
   occurs(t, bind(Tool, PortHigh, ArtifactHigh)),
   artifact_precedes(t, ArtifactHigh, ArtifactLow).

% If symmetric same-signature ports both consume the same workflow input, use a
% different equivalent workflow input when one is available.
:- occurs(t, use_variant(Tool, TI_ID)),
   variant_sym_ports(TI_ID, PortHigh, PortLow),
   occurs(t, bind(Tool, PortLow, WF)),
   occurs(t, bind(Tool, PortHigh, WF)),
   workflow_input_slot(WF, Rep, _),
   eligible(t-1, PortHigh, WFAlt),
   workflow_input_slot(WFAlt, Rep, _),
   WFAlt != WF.

% Within one step, interchangeable slots of the same workflow-input class are
% assigned canonically to ascending port ids.
:- occurs(t, use_variant(Tool, TI_ID)),
   occurs(t, bind(Tool, Port1, Slot1)),
   occurs(t, bind(Tool, Port2, Slot2)),
   Port1 < Port2,
   workflow_input_slot(Slot1, Rep, Rank1),
   workflow_input_slot(Slot2, Rep, Rank2),
   Rank1 > Rank2.

slot_used_before(t, Rep, Rank) :-
    workflow_input_slot(Slot, Rep, Rank),
    occurs(TPrev, bind(_, _, Slot)),
    TPrev < t.

slot_used_at_step(t, Rep, Rank) :-
    workflow_input_slot(Slot, Rep, Rank),
    occurs(t, bind(_, _, Slot)).

% Across steps, use slots in prefix order within each equivalence class.
:- slot_used_at_step(t, Rep, Rank),
   workflow_input_slot(_, Rep, SmallerRank),
   SmallerRank < Rank,
   not slot_used_before(t, Rep, SmallerRank),
   not slot_used_at_step(t, Rep, SmallerRank).

% If the immediately previous step already consumed a workflow input on a
% matching-signature port and produced an eligible output for the current port,
% prefer that fresh output over reusing the original workflow input directly.
immediate_chain_output_preferred(t, Tool, Port, WF, PrevTool, PrevOutPort) :-
    occurs(t, bind(Tool, Port, WF)),
    workflow_input(WF),
    occurs(t-1, output(PrevTool, _, PrevOutPort)),
    eligible(t-1, Port, out(t-1, PrevTool, PrevOutPort)),
    occurs(t-1, bind(PrevTool, PrevPort, WF)),
    same_input_signature(PrevPort, Port).

:- immediate_chain_output_preferred(t, Tool, Port, WF, PrevTool, PrevOutPort).

% If a tool already consumes the previous step's output on one port and that
% same fresh output is also eligible on another port, prefer it over the
% original workflow input that was used to produce it.
immediate_output_dominates_source_input(t, Tool, Port, WF, PrevTool, PrevOutPort) :-
    occurs(t, bind(Tool, Port, WF)),
    workflow_input(WF),
    occurs(t, bind(Tool, OtherPort, out(t-1, PrevTool, PrevOutPort))),
    OtherPort != Port,
    eligible(t-1, Port, out(t-1, PrevTool, PrevOutPort)),
    occurs(t-1, bind(PrevTool, _, WF)).

:- immediate_output_dominates_source_input(t, Tool, Port, WF, PrevTool, PrevOutPort).

% If the immediately previous step already consumed a workflow-input slot on a
% matching-signature port and produced an eligible output for the current port,
% prefer that fresh output over reusing a later-rank equivalent slot directly.
reused_equivalent_slot_after_immediate_output(t, Tool, Slot, Rep, Rank) :-
    occurs(t, bind(Tool, _, Slot)),
    workflow_input_slot(Slot, Rep, Rank),
    Rank > 1,
    occurs(t, bind(Tool, _, out(t-1, _, _))),
    slot_used_before(t, Rep, Rank).

:- reused_equivalent_slot_after_immediate_output(t, Tool, Slot, Rep, Rank),
   workflow_input_slot(SmallerSlot, Rep, SmallerRank),
   SmallerRank < Rank,
   eligible(t-1, Port, SmallerSlot),
   occurs(t, bind(Tool, Port, Slot)),
   slot_used_before(t, Rep, SmallerRank).

% Artifact ancestry over committed runs.
artifact_parent(out(t, Tool, PortOut), WF) :-
    occurs(t, output(Tool, _, PortOut)),
    occurs(t, bind(Tool, _, WF)).

% Carry forward ancestors from previous steps; the artifact graph is monotone.
artifact_ancestor(t, Artifact, Ancestor) :-
    artifact_ancestor(t-1, Artifact, Ancestor).

% Only compute fresh ancestry for artifacts produced at THIS step.
artifact_ancestor(t, out(t, Tool, PortOut), Ancestor) :-
    artifact_parent(out(t, Tool, PortOut), Ancestor).

artifact_ancestor(t, out(t, Tool, PortOut), Ancestor) :-
    artifact_parent(out(t, Tool, PortOut), Mid),
    artifact_ancestor(t, Mid, Ancestor).

% When the same variant is used in a chain and the later run consumes the
% earlier output, keep workflow-input bindings on the repeated ports ordered by
% input id across time. This collapses branch-swapped repeated-tool chains.
same_variant_chain(t, TPrev, Tool, TI_ID) :-
    TPrev < t,
    occurs(TPrev, use_variant(Tool, TI_ID)),
    occurs(t, use_variant(Tool, TI_ID)),
    occurs(t, bind(Tool, _, out(TPrev, Tool, _))).

:- same_variant_chain(t, TPrev, Tool, TI_ID),
   input_port(TI_ID, Port),
   occurs(TPrev, bind(Tool, Port, PrevArtifact)),
   occurs(t, bind(Tool, Port, Artifact)),
   artifact_precedes(t, Artifact, PrevArtifact).

% In a three-step same-variant chain, if the middle run still consumes a raw
% workflow input even though the step before it already produced an eligible
% output from that same input, keep only the version where the middle run uses
% that fresh output. This targets the remaining repeated-chain swap family
% without broad ancestor-based pruning.
same_variant_chain_middle_prefers_previous_output(
    t, Tool, TI_ID, Port, WF, EarlierTool, EarlierOutPort
) :-
    same_variant_chain(t, t-1, Tool, TI_ID),
    occurs(t-1, bind(Tool, Port, WF)),
    workflow_input(WF),
    occurs(t-2, output(EarlierTool, _, EarlierOutPort)),
    eligible(t-2, Port, out(t-2, EarlierTool, EarlierOutPort)),
    occurs(t-2, bind(EarlierTool, _, WF)).

:- same_variant_chain_middle_prefers_previous_output(
       t, Tool, TI_ID, Port, WF, EarlierTool, EarlierOutPort).

% If the current step consumes the previous step's output, and that previous
% step still consumed a raw workflow input even though the step before it had
% already produced an eligible output from that same workflow input, prefer the
% tighter chain through the earlier output.
middle_step_prefers_earlier_output_in_consumed_chain(
    t, MidTool, MidPort, WF, EarlierTool, EarlierOutPort
) :-
    t > 2,
    occurs(t, bind(_, _, out(t-1, MidTool, _))),
    occurs(t-1, bind(MidTool, MidPort, WF)),
    workflow_input(WF),
    occurs(t-2, output(EarlierTool, _, EarlierOutPort)),
    eligible(t-2, MidPort, out(t-2, EarlierTool, EarlierOutPort)),
    occurs(t-2, bind(EarlierTool, _, WF)).

:- middle_step_prefers_earlier_output_in_consumed_chain(
       t, MidTool, MidPort, WF, EarlierTool, EarlierOutPort).

% When two single-input producer runs with the same input signature later feed a
% symmetric join, keep the earlier producer on the more-derived source and the
% later producer on the older source. This removes the extra branch-swapped
% variants while preserving the SAT-aligned one.
paired_symmetric_producers(t, T1, Tool1, TI1, Port1, WF1, T2, Tool2, TI2, Port2, WF2) :-
    T1 < T2,
    occurs(T1, use_variant(Tool1, TI1)),
    occurs(T2, use_variant(Tool2, TI2)),
    variant_single_input(TI1),
    variant_single_input(TI2),
    occurs(T1, bind(Tool1, Port1, WF1)),
    occurs(T2, bind(Tool2, Port2, WF2)),
    same_input_signature(Port1, Port2),
    occurs(TJ, use_variant(JoinTool, JoinTI)),
    variant_sym_ports(JoinTI, JoinPort1, JoinPort2),
    occurs(TJ, bind(JoinTool, JoinPort1, out(T1, Tool1, _))),
    occurs(TJ, bind(JoinTool, JoinPort2, out(T2, Tool2, _))),
    TJ > T2,
    JoinPort2 < JoinPort1.

:- paired_symmetric_producers(t, T1, Tool1, TI1, Port1, WF, T2, Tool2, TI2, Port2, WF).

:- paired_symmetric_producers(t, T1, Tool1, TI1, Port1, WF1, T2, Tool2, TI2, Port2, WF2),
   artifact_ancestor(t, WF1, WF2).

% When repeated runs of the same single-input variant later feed the two
% symmetric inputs of the same join, order the producer inputs canonically.
repeated_same_variant_symmetric_join_input_swap(
    t, T1, Tool, TI_ID, WF1, T2, WF2, JoinTool, JoinTI, JoinPortHigh, JoinPortLow
) :-
    T1 < T2,
    occurs(T1, use_variant(Tool, TI_ID)),
    occurs(T2, use_variant(Tool, TI_ID)),
    variant_single_input(TI_ID),
    occurs(T1, bind(Tool, Port1, WF1)),
    occurs(T2, bind(Tool, Port2, WF2)),
    occurs(TJ, use_variant(JoinTool, JoinTI)),
    variant_sym_ports(JoinTI, JoinPortHigh, JoinPortLow),
    occurs(TJ, bind(JoinTool, JoinPortLow, out(T1, Tool, _))),
    occurs(TJ, bind(JoinTool, JoinPortHigh, out(T2, Tool, _))),
    artifact_precedes(t, WF2, WF1).

:- repeated_same_variant_symmetric_join_input_swap(
       t, T1, Tool, TI_ID, WF1, T2, WF2, JoinTool, JoinTI, JoinPortHigh, JoinPortLow).

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

% Output dimensions — Case A: terminal value
holds(t, dim(out(t, Tool, Port), V, Cat)) :-
    occurs(t, output(Tool, _, Port)),
    dimension(Port, (V, Cat)),
    is_terminal(V).

% Output dimensions — Case B: non-terminal, planner picks terminal descendant
1 { occurs(t, output_dim(Tool, Port, Descendant, Cat)) :
    compatible(Descendant, V),
    is_terminal(Descendant) } 1 :-
    occurs(t, output(Tool, _, Port)),
    dimension(Port, (V, Cat)),
    not is_terminal(V).

holds(t, dim(out(t, Tool, Port), Descendant, Cat)) :-
    occurs(t, output_dim(Tool, Port, Descendant, Cat)).

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
    holds(t, unit(WF, U)),
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
    holds(t, unit(WF, U)),
    U != none.

% Unit binding compatibility constraints
:- occurs(t, bind(Tool, Port, WF)),
   port_unit(Port, UPort),
   holds(t, unit(WF, UData)),
   UData != none,
   not units_compatible(UPort, UData).

:- unit_mode(strict),
   occurs(t, bind(Tool, Port, WF)),
   port_unit(Port, _),
   not holds(t, unit(WF, _)).

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
    holds(t, dim(WF, ActualV, Cat)),
    compatible(ActualV, GoalV).

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
    port_property(Tool, Port, ReqV, Cat, Prop),
    occurs(t, bind(Tool, Port, WF)),
    holds(t, dim(WF, ActualV, Cat)),
    compatible(ActualV, ReqV).


% ----------------------------------------------------------------
% From requirements.lp — constraints only (static derivations in base.lp)
% ----------------------------------------------------------------

:- requirement_filtering_active, occurs(t, run(Tool)), not tool_req_compatible(Tool).
:- requirement_filtering_active, occurs(t, bind(Tool, Port, _)), not port_env_compatible(Tool, Port).

% ----------------------------------------------------------------
% From constraints.lp
% ----------------------------------------------------------------

% Contiguous execution: no gaps in the plan
:- t > 1, occurs(t, run(_)), not occurs(t-1, run(_)).

% SAT mode does not forbid repeated tool occurrences during synthesis. The
% config-level tool_seq_repeat flag is handled later as a solution-distinction
% policy, so repeated runs remain legal here.

% ----------------------------------------------------------------
% Sequential workflow: at most one tool runs per timestep
% (mirrors the SAT automaton which allocates exactly one tool slot per step)
% ----------------------------------------------------------------

:- 2 { occurs(t, run(_)) }.
