#program check(t).

% TOOL INCLUSION / EXCLUSION (optimized multi-shot)
% Selector matches are pre-resolved incrementally in constraints.lp; this file
% only enforces the final must-use / must-not-use checks at the query horizon.

:- query(t),
   constraint_must_use(Sel),
   not constraint_selected_any(t, Sel).

:- query(t),
   constraint_must_not_use(Sel),
   constraint_selected_any(t, Sel).
