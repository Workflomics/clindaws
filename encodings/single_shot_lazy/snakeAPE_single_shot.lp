% ================================================================
% snakeAPE — single shot planning modules
% ================================================================
%
% Include this file to load the full single-shot planning stack
% plus all user constraint enforcement.
% User declarations (must_use, before, require_property, etc.)
% go in user_constraints.lp, which is included here.

% --- Core planning modules ---
#include "show.lp".
#include "pre_compute.lp".
#include "propagation.lp".
#include "tool_choice.lp".
#include "output_production.lp".
#include "reachability.lp".
#include "goal.lp".
#include "usefulness.lp".
#include "constraints.lp".
#include "tool_taxonomy_logic.lp".

% --- User constraint declarations ---
#include "user_constraints.lp".

% --- Constraint enforcement ---
#include "plan_constraints.lp".
#include "temporal_constraint.lp".
#include "tool_inclusion_constraints.lp".
#include "tool_dependency_constraints.lp".
#include "tool_repetition_constraints.lp".
