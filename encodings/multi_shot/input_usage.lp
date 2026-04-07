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
% enable_all_inputs_used — every initial ingredient must be bound
% ----------------------------------------------------------------

:- query(t),
   enable_all_inputs_used,
   workflow_input_slot(WF, _, _),
   #count { T, Tool, Port : occurs(T, bind(Tool, Port, WF)) } = 0.

% ----------------------------------------------------------------
% enable_some_input_used — at least one initial ingredient is bound
% ----------------------------------------------------------------

:- query(t),
   enable_some_input_used,
   #count { WF : workflow_input_slot(WF, _, _),
                 occurs(T, bind(Tool, Port, WF)) } = 0.

% ----------------------------------------------------------------
% enable_inputs_consumed — synonym for used_once in multi-shot
% (consumed semantics: ingredient used at most once, same constraint)
% ----------------------------------------------------------------

:- query(t),
   enable_inputs_consumed,
   workflow_input_slot(WF, _, _),
   #count { T, Tool, Port : occurs(T, bind(Tool, Port, WF)) } > 1.
