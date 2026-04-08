% ================================================================
% INPUT USAGE CONSTRAINTS (multi-shot)
% #program check(t)
%
% Activation: enable_all_inputs_used. / enable_some_input_used.
%             enable_inputs_consumed. in config / extra_lp
% Docs: DOCS/multi_shot/constraints/input_usage.md
%
% Unlike the single-shot version, the intermediate input_used(WF)
% atom is not used here — it would cause redefinition errors when
% a new binding for WF first appears at step T > 1 (the atom was
% committed as false after earlier steps). Aggregate counts in the
% constraint body avoid this.
% ================================================================

#program check(t).

% ----------------------------------------------------------------
% APE-style workflow-input usage:
% an initial artifact counts as used when some later used state references an
% artifact that is the input itself or a descendant produced from it.
% This mirrors APE's MEM_TYPE_REFERENCE-based check more closely than a pure
% direct-bind count.
% ----------------------------------------------------------------

workflow_input_reference_check(t, Artifact, WF) :-
   query(t),
   planner_workflow_input(WF),
   Artifact = WF.

workflow_input_reference_check(t, Artifact, WF) :-
   query(t),
   planner_workflow_input(WF),
   artifact_ancestor(t, Artifact, WF).

workflow_input_used_check(t, WF) :-
   query(t),
   planner_workflow_input(WF),
   occurs(T, bind(Tool, Port, Artifact)),
   workflow_input_reference_check(t, Artifact, WF).

workflow_input_used_check(t, WF) :-
   query(t),
   planner_workflow_input(WF),
   chosen_goal_output(t, _, Artifact),
   workflow_input_reference_check(t, Artifact, WF).

% ----------------------------------------------------------------
% enable_all_inputs_used — every initial ingredient must be referenced by some
% later tool input or chosen workflow output.
% ----------------------------------------------------------------

:- query(t),
   enable_all_inputs_used,
   planner_workflow_input(WF),
   not workflow_input_used_check(t, WF).

% ----------------------------------------------------------------
% enable_some_input_used — at least one initial ingredient is referenced by
% some later tool input or chosen workflow output.
% ----------------------------------------------------------------

:- query(t),
   enable_some_input_used,
   #count { WF : workflow_input_used_check(t, WF) } = 0.

% ----------------------------------------------------------------
% enable_inputs_consumed — synonym for used_once in multi-shot
% (consumed semantics: ingredient used at most once, same constraint)
% ----------------------------------------------------------------

:- query(t),
   enable_inputs_consumed,
   planner_workflow_input(WF),
   #count { T, Tool, Port : occurs(T, bind(Tool, Port, WF)) } > 1.
