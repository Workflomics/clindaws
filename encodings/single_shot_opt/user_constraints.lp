% ================================================================
% USER CONSTRAINTS
% ================================================================
%
% This file contains optional constraints that prune the solution
% space according to user preferences. Each constraint is controlled
% by a single fact that enables it. To activate a constraint, simply
% uncomment the corresponding enable fact.
%
% SECTIONS:
%   1. Plan length constraints        → plan_constraints.lp
%   2. Temporal constraints           → temporal_constraint.lp
%   3. Tool inclusion/exclusion       → tool_inclusion_constraints.lp
%   5. Input usage constraints        → input_usage_constraints.lp
%   6. Tool repetition limit          → tool_repetition_constraints.lp
%   8. Minimization / Optimization


% ================================================================
% SECTION 1: PLAN LENGTH CONSTRAINTS
% ================================================================
%
% Restricts solutions to the shortest valid plan length.
% Enforcement logic lives in plan_constraints.lp (included below).
%
% TRADE-OFF: You lose all longer valid pipelines. Enable only if you
%            want a single canonical shortest plan per structure.

% Uncomment to enable:
% enable_shortest_plan.


% ================================================================
% SECTION 2: TEMPORAL CONSTRAINTS
% ================================================================
%
% Declare ordering facts here. Enforcement lives in temporal_constraint.lp.
%
% PREDICATES:
%   before(A, B)            — A at T, B at T+1 (directly adjacent)
%   after(A, B)             — A at T, B at T-1 (directly adjacent)
%   eventually_before(A, B) — A at some T1 < T2 where B runs (any gap)
%   eventually_after(A, B)  — A at some T1 > T2 where B runs (any gap)
%
% SELECTORS (usable as either argument):
%   tool_name                               — concrete tool
%   abstract_class_name                     — any member of that class
%   with_input_type(Tool, TypeVal)          — tool bound to input of that type
%   with_input_state(Tool, State)           — tool bound to input in that state
%   with_output_type(Tool, TypeVal)         — tool producing output of that type
%   with_output_state(Tool, State)          — tool producing output in that state
%   with_input_unit(Tool, Unit)             — tool bound to input with that unit
%   with_property(Tool, Prop)               — tool carrying that tool_property
%   with_requirement(Tool, ReqType, ReqVal) — tool with that capability requirement
%   and(SelA, SelB)                         — both conditions at the same timestep
%   named_selector                          — N conditions without nesting (see below)
%
% COMBINING SELECTORS:
%   % Inline (two conditions):
%   before(and(with_input_type(tool_a, value_x), with_property(tool_a, prop_y)), tool_b).
%
%   % Named (N conditions, no nesting):
%   selector(my_sel).
%   selector_cond(my_sel, with_input_type(tool_a, value_x)).
%   selector_cond(my_sel, with_property(tool_a, prop_y)).
%   eventually_before(my_sel, tool_b).

% before(tool_a, tool_b).
% eventually_before(with_input_type(tool_a, value_x), tool_b).


% ================================================================
% SECTION 3: TOOL INCLUSION / EXCLUSION CONSTRAINTS
% ================================================================
%
% Force specific tools to appear or not appear in every plan.
% Abstract class names from the tool taxonomy are also accepted.
% Enforcement logic lives in tool_inclusion_constraints.lp.
%
% PATTERN:
%   must_use(tool)     — this tool must appear in every valid plan.
%   must_not_use(tool) — this tool must never appear.
%
% EXAMPLES:
%   % must_use(let_rise).
%   % must_not_use(clean_oven).
%   % must_use(baking).       % abstract class — any baking tool must run
%   % must_not_use(maintenance). % abstract class — no maintenance tool




% ================================================================
% SECTION 5: INPUT USAGE CONSTRAINTS
% ================================================================
%
% Controls how initial workflow inputs are treated. Three
% independent constraints, each activated by an enable fact.
% Enforcement logic lives in input_usage_constraints.lp.
%
% Uncomment to enable:
%   % enable_all_inputs_used.   % every input must be bound somewhere
%   % enable_inputs_used_once.  % each input bound to at most one port
%   % enable_inputs_consumed.   % binding consumes the item (destructive)


% ================================================================
% SECTION 6: TOOL REPETITION LIMIT
% ================================================================
%
% By default every tool may only run once per plan.
% Enforcement logic lives in tool_repetition_constraints.lp.
%
% To allow a tool to run more than once, declare it here:
%   % multi_run(add_extra_topping).
%   % multi_run(slice_toppings).





% ----------------------------------------------------------------
% SECTION 8: MINIMIZATION
% ----------------------------------------------------------------
% Minimize the amount of workflow candidates and find the optimal ones

% #minimize { T, Tool : occurs(T, run(Tool)) }.

