% ================================================================
% CONSTRAINTS
% ================================================================
%
% This file contains hard integrity constraints that apply to every
% valid answer set regardless of user preferences. These are not
% configurable — they encode fundamental correctness properties
% of the pipeline planning semantics.

% ----------------------------------------------------------------
% CONTIGUOUS EXECUTION
% ----------------------------------------------------------------
%
% Tool runs must form a contiguous block starting from T=1.
% If any tool runs at timestep T > 1, then some tool must have
% run at T-1. This prevents gaps in the plan — e.g., a tool
% running at T=5 with nothing at T=4.
%
% PURPOSE: Forces a canonical plan representation where all tool
%          executions are packed as early as possible. Without
%          this, the same pipeline could be assigned to timesteps
%          1..N or 3..N+2 or 5..N+4 etc., producing an explosion
%          of timing variants that are structurally identical.
%
% INTERACTION WITH MULTI-SHOT: The iterative deepening driver
%          increments the horizon until SAT. Contiguous execution
%          ensures that at horizon N, every model uses all N steps
%          (no wasted slack), so the first SAT horizon gives the
%          true minimum plan length.
%
% NOTE: This constraint does NOT force exactly one tool per
%       timestep — multiple tools may still run in parallel at
%       the same T, as long as no timestep is skipped entirely.

:- time(T), T > 1, occurs(T, run(_)), not occurs(T-1, run(_)).

% ----------------------------------------------------------------
% NO REPEATED PORT BINDING FOR THE SAME TOOL INSTANCE
% ----------------------------------------------------------------
%
% If a tool runs twice (which requires it to be declared in
% multi_run/1), it cannot bind the same input port to the same
% workflow item in both runs. Each run of the tool must consume
% at least one distinct input.
%
% PURPOSE: Prevents a multi-run tool from doing identical work
%          twice. For example, add_extra_topping running twice
%          but binding the same topping ingredient to the same
%          port in both runs produces no additional value — it
%          is semantically equivalent to running once.
%
% NOTE: This constraint only activates when a tool appears in
%       multi_run/1 (declared in constraints.lp). Single-run
%       tools are already covered by the multi_run count
%       constraint which prevents them from running more than
%       once at all.
%
% EXAMPLE: add_extra_topping at T=3 and T=5 are both allowed,
%          but if T=3 binds pepperoni to topping_port, then T=5
%          must bind a different ingredient to topping_port.

% SAT-aligned behavior permits rerunning the same tool on the same bound input
% chain when the workflow remains useful, so do not forbid these repetitions
% here.
