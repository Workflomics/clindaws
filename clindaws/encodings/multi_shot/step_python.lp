#program step(t).

% ================================================================
% STEP — per-timestep planning rules
% ================================================================

% Mark this timestep
time(t).

% ----------------------------------------------------------------
% Transition firing for stateful artifacts (non-monotonic)
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
% Tool choice, variant selection, port binding
% ----------------------------------------------------------------

% Run choice
{ occurs(t, run(Tool)) } :- precomputed_tool_allowed_at_step(Tool, t), not tool_has_input(Tool).
{ occurs(t, run(Tool)) } :- precomputed_tool_allowed_at_step(Tool, t), tool_has_input(Tool).

% Heuristic: Branch on earlier time steps first to solidify the pipeline prefix
#heuristic occurs(t, run(Tool)). [100 - t, true]


% Variant selection: exactly one input group per run
1 { occurs(t, use_variant(Tool, TI_ID)) : tool_input_variant(Tool, TI_ID) } 1 :-
    occurs(t, run(Tool)),
    tool_has_input(Tool).

occurs(t, run(Tool)) :- occurs(t, use_variant(Tool, _)).

% Signature-level eligibility
eligible_signature(t, Signature, WF) :-
    direct_signature_initial_artifact(Signature, WF),
    workflow_input(WF),
    available(WF).

eligible_signature(t, Signature, Slot) :-
    direct_signature_initial_artifact(Signature, Slot),
    workflow_input_slot(Slot, _, _),
    available(Slot).

output_signature_candidate(t, Signature, out(TPrev, ToolPrev, ProducerPort)) :-
    direct_signature_support_class(Signature, SupportClass),
    direct_output_port_profile(ProducerPort, Profile),
    direct_support_class_profile(SupportClass, Profile),
    available(out(TPrev, ToolPrev, ProducerPort)).

signature_cat_ok(t, Signature, out(TPrev, ToolPrev, ProducerPort), Cat) :-
    output_signature_candidate(t, Signature, out(TPrev, ToolPrev, ProducerPort)),
    direct_signature_requires(Signature, ReqV, Cat),
    artifact_dim(out(TPrev, ToolPrev, ProducerPort), V, Cat),
    compatible(V, ReqV).

eligible_signature(t, Signature, out(TPrev, ToolPrev, ProducerPort)) :-
    output_signature_candidate(t, Signature, out(TPrev, ToolPrev, ProducerPort)),
    signature_cat_ok(t, Signature, out(TPrev, ToolPrev, ProducerPort), Cat) :
        direct_signature_category(Signature, Cat).

eligible(t, Port, Artifact) :-
    direct_port_signature(Port, Signature),
    eligible_signature(t, Signature, Artifact).

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
% assigned canonically to ascending port ids.
:- occurs(t, use_variant(Tool, TI_ID)),
   not full_workflow_input_witnesses,
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
   not full_workflow_input_witnesses,
   workflow_input_slot(_, Rep, SmallerRank),
   SmallerRank < Rank,
   not slot_used_before(t, Rep, SmallerRank),
   not slot_used_at_step(t, Rep, SmallerRank).

% Artifact ancestry over committed runs.
artifact_parent(out(t, Tool, PortOut), WF) :-
    occurs(t, output(Tool, _, PortOut)),
    occurs(t, bind(Tool, _, WF)).

% Carry forward ancestors from previous steps; the artifact graph is monotone.
artifact_ancestor(out(t, Tool, PortOut), Ancestor) :-
    artifact_parent(out(t, Tool, PortOut), Ancestor).

artifact_ancestor(out(t, Tool, PortOut), Ancestor) :-
    artifact_parent(out(t, Tool, PortOut), Mid),
    artifact_ancestor(Mid, Ancestor).

% Direct-only chain-pruning rules that tried to canonicalize repeated-input and
% repeated-tool witnesses were removed here. They reduced duplicates, but they
% can also prune valid workflows that the compressed-candidate backend and APE
% still admit. GeoGMT E0 is one concrete regression case. Keep only the sound
% local symmetric-port canonicalization above.

% ----------------------------------------------------------------
% From output_production.lp — output port selection and dimensions
% ----------------------------------------------------------------

% Output selection: exactly one port per tool_output group
1 { occurs(t, output(Tool, TOut_ID, Port)) : output_port(TOut_ID, Port) } 1 :-
    occurs(t, run(Tool)),
    tool_has_output(Tool),
    tool_output(Tool, TOut_ID).

% Output availability (monotonic)
available(out(t, Tool, Port)) :-
    occurs(t, output(Tool, _, Port)).

% Output dimensions — Case A: terminal value
artifact_dim(out(t, Tool, Port), V, Cat) :-
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


artifact_dim(out(t, Tool, Port), Descendant, Cat) :-
    occurs(t, output_dim(Tool, Port, Descendant, Cat)).

% ----------------------------------------------------------------
% From unit_reasoning.lp — unit production from tool outputs
% ----------------------------------------------------------------

% Case 1: explicit port_unit declaration
artifact_unit(out(t, Tool, Port), U) :-
    occurs(t, output(Tool, _, Port)),
    port_unit(Port, U).

% Case 2: strict mode — no declaration → unit is none
artifact_unit(out(t, Tool, Port), none) :-
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

artifact_unit(out(t, Tool, Port), U) :-
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
    available(WF),
    artifact_dim(WF, ActualV, Cat),
    compatible(ActualV, GoalV).

goal_dim_missing(t, GoalID, WF) :-
    goal_output(GoalID, GoalV, Cat),
    available(WF),
    not goal_dim_match(t, GoalID, WF, GoalV, Cat).

goal_satisfied_by(t, GoalID, WF) :-
    available(WF),
    goal_output(GoalID, _, _),
    goal_dim_match(t, GoalID, WF, _, _),
    not goal_dim_missing(t, GoalID, WF).

goal_satisfied_by_output(t, GoalID, WF) :-
    goal_satisfied_by(t, GoalID, WF),
    not planner_workflow_input(WF).

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
    artifact_dim(WF, ActualV, Cat),
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
