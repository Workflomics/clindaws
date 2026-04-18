#program step_initial(t).

% ================================================================
% STEP_INITIAL — specialized horizon-1 expansion
% ================================================================
% The first step can use workflow-input-only bindability facts and, in exact
% horizon mode, an even smaller goal-oriented filter over initial candidates and
% retained output values.

time(t).

initial_goal_filter_active(t) :-
    t = 1,
    query(t),
    exact_horizon_mode.

1 { occurs(t, run(Tool)) : dynamic_initial_goal_tool(Tool) } 1 :-
    initial_goal_filter_active(t).

1 { occurs(t, run(Tool)) : dynamic_step_allowed_tool(Tool, t) } 1 :-
    not initial_goal_filter_active(t).
:- initial_goal_filter_active(t), not occurs(t, run(_)).

:- dynamic_must_run_tool_at_step(ToolMust, t),
   occurs(t, run(Tool)),
   Tool != ToolMust.

1 { occurs(t, use_dynamic_candidate(Tool, Candidate)) :
    dynamic_tool_candidate(Candidate, Tool),
    dynamic_initial_goal_candidate(Candidate),
    dynamic_candidate_allowed_at_step(Candidate, t) } 1 :-
    initial_goal_filter_active(t),
    occurs(t, run(Tool)).

1 {
    occurs(t, use_dynamic_candidate(Tool, Candidate)) :
        dynamic_tool_candidate(Candidate, Tool),
        dynamic_candidate_allowed_at_step(Candidate, t)
} 1 :-
    not initial_goal_filter_active(t),
    occurs(t, run(Tool)).

:- dynamic_must_run_candidate_at_step(CandidateMust, t),
   occurs(t, use_dynamic_candidate(_, Candidate)),
   Candidate != CandidateMust.

occurs(t, run(Tool)) :- occurs(t, use_dynamic_candidate(Tool, _)).

:- initial_goal_filter_active(t),
   occurs(t, use_dynamic_candidate(_, Candidate)),
   enable_all_outputs_consumed,
   dynamic_candidate_total_output_multiplicity(Candidate, NP),
   NG = #count { GoalID : goal_id(GoalID) },
   NP != NG.

1 { occurs(t, bind(Tool, Port, WF)) : dynamic_initial_bindable(Candidate, Port, WF) } 1 :-
    occurs(t, use_dynamic_candidate(Tool, Candidate)),
    dynamic_candidate_input_port(Candidate, Port).

% Carry workflow-input lineage only; input-usage checks do not need full
% produced-artifact transitive ancestry in the optimized backend.
workflow_input_lineage(t, out(t, Tool, PortOut), WF) :-
    occurs(t, output(Tool, _, PortOut)),
    occurs(t, bind(Tool, _, Artifact)),
    workflow_input_lineage(0, Artifact, WF).

:- occurs(t, use_dynamic_candidate(Tool, Candidate)),
   occurs(t, bind(Tool, Port1, WF1)),
   occurs(t, bind(Tool, Port2, WF2)),
   Port1 < Port2,
   same_candidate_input_signature(Candidate, Port1, Candidate, Port2),
   equivalent_workflow_inputs(WF2, WF1).

occurs(t, output(Tool, Candidate, Port)) :-
    occurs(t, use_dynamic_candidate(Tool, Candidate)),
    dynamic_candidate_output_port(Candidate, Port).

available(out(t, Tool, Port)) :-
    occurs(t, output(Tool, _, Port)).


1 { occurs(t, output_dim(Tool, Port, V, Category)) :
    dynamic_initial_goal_output_value(Candidate, Port, V, Category) } 1 :-
    initial_goal_filter_active(t),
    occurs(t, output(Tool, Candidate, Port)),
    dynamic_initial_goal_output_value(Candidate, Port, _, Category).

1 { occurs(t, output_dim(Tool, Port, V, Category)) :
    dynamic_candidate_output_choice_value(Candidate, Port, V, Category) } 1 :-
    occurs(t, output(Tool, Candidate, Port)),
    dynamic_candidate_output_choice_value(Candidate, Port, _, Category),
    not initial_goal_filter_active(t).

artifact_dim(out(t, Tool, Port), V, Category) :-
    occurs(t, output_dim(Tool, Port, V, Category)).

artifact_dim(out(t, Tool, Port), V, Category) :-
    occurs(t, output(Tool, Candidate, Port)),
    dynamic_candidate_output_singleton(Candidate, Port, V, Category).

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
