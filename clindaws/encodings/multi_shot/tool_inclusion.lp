% ================================================================
% TOOL INCLUSION / EXCLUSION CONSTRAINTS (multi-shot)
% #program check(t)
%
% Activation: must_use(Tool). / must_not_use(Tool). in config / extra_lp
% Docs: DOCS/multi_shot/constraints/tool_inclusion.md
%
% All constraints live in #program check(t) with a query(t) guard.
% Using occurs(T, run(Tool)) ranges over ALL steps committed so far;
% the query(t) guard namespaces the constraint to the current horizon.
% ================================================================

#program check(t).

% ----------------------------------------------------------------
% must_use(Tool) — concrete tool must run at least once
% ----------------------------------------------------------------

:- query(t),
   constraint_must_use(Sel),
   not constraint_selected_any(t, Sel).

% ----------------------------------------------------------------
% must_not_use(Tool) — concrete tool must not run at all
% ----------------------------------------------------------------

:- query(t),
   constraint_must_not_use(Sel),
   constraint_selected_any(t, Sel).
