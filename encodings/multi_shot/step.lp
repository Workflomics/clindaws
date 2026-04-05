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

% Variant selection: exactly one input group per run
1 { occurs(t, use_variant(Tool, TI_ID)) : tool_input_variant(Tool, TI_ID) } 1 :-
    occurs(t, run(Tool)),
    tool_has_input(Tool).

occurs(t, run(Tool)) :- occurs(t, use_variant(Tool, _)).

% Port satisfaction per category
port_cat_ok(t, Port, WF, Cat) :-
    holds(t, avail(WF)),
    port_requires(Port, ReqV, Cat),
    holds(t, dim(WF, V, Cat)),
    compatible(V, ReqV).

% Eligible item: all categories satisfied
eligible(t, Port, WF) :-
    holds(t, avail(WF)),
    input_port(_, Port),
    port_cat_ok(t, Port, WF, Cat) : port_requires(Port, _, Cat).

% Port binding: exactly one eligible item per port
1 { occurs(t, bind(Tool, Port, WF)) : eligible(t-1, Port, WF) } 1 :-
    occurs(t, use_variant(Tool, TI_ID)),
    input_port(TI_ID, Port).

% Canonicalize equivalent workflow-input bindings for same-signature ports.
:- occurs(t, use_variant(Tool, TI_ID)),
   occurs(t, bind(Tool, Port1, WF1)),
   occurs(t, bind(Tool, Port2, WF2)),
   Port1 < Port2,
   input_port(TI_ID, Port1),
   input_port(TI_ID, Port2),
   same_input_signature(Port1, Port2),
   equivalent_workflow_inputs(WF2, WF1).

% Artifact ancestry over committed runs.
artifact_parent(out(t, Tool, PortOut), WF) :-
    occurs(t, output(Tool, _, PortOut)),
    occurs(t, bind(Tool, _, WF)).

artifact_ancestor(t, Artifact, Ancestor) :-
    artifact_parent(Artifact, Ancestor).

artifact_ancestor(t, Artifact, Ancestor) :-
    artifact_parent(Artifact, Mid),
    artifact_ancestor(t, Mid, Ancestor).

% Symmetry breaking for duplicate input signatures on the same join tool.
:- occurs(t, bind(Tool, Port1, out(T1, Tool1, _))),
   occurs(t, bind(Tool, Port2, out(T2, Tool2, _))),
   Port1 < Port2,
   same_input_signature(Port1, Port2),
   T1 < T2.

% When two single-input producer runs with the same input signature later feed a
% symmetric join, keep the earlier producer on the more-derived source and the
% later producer on the older source. This removes the extra branch-swapped
% variants while preserving the SAT-aligned one.
paired_symmetric_producers(t, T1, Tool1, Port1, WF1, T2, Tool2, Port2, WF2) :-
    T1 < T2,
    tool_single_input(Tool1),
    tool_single_input(Tool2),
    occurs(T1, bind(Tool1, Port1, WF1)),
    occurs(T2, bind(Tool2, Port2, WF2)),
    same_input_signature(Port1, Port2),
    occurs(TJ, bind(JoinTool, JoinPort1, out(T1, Tool1, _))),
    occurs(TJ, bind(JoinTool, JoinPort2, out(T2, Tool2, _))),
    TJ > T2,
    JoinPort2 < JoinPort1,
    same_input_signature(JoinPort1, JoinPort2).

:- paired_symmetric_producers(t, T1, Tool1, Port1, WF, T2, Tool2, Port2, WF).

:- paired_symmetric_producers(t, T1, Tool1, Port1, WF1, T2, Tool2, Port2, WF2),
   artifact_ancestor(t, WF1, WF2).

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
    occurs(T1, output(Tool1, _, Port)),
    occurs(t, bind(Tool2, _, out(T1, Tool1, Port))),
    t > T1.

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

% No runs after goal is reached
:- occurs(t, run(_)), holds(t-1, goal).

% SAT mode does not forbid repeated tool occurrences during synthesis. The
% config-level tool_seq_repeat flag is handled later as a solution-distinction
% policy, so repeated runs remain legal here.

% ----------------------------------------------------------------
% Sequential workflow: at most one tool runs per timestep
% (mirrors the SAT automaton which allocates exactly one tool slot per step)
% ----------------------------------------------------------------

:- 2 { occurs(t, run(_)) }.
