% LAZY CONSTRAINTS (multi-shot-lazy)
% Constraint facts are emitted by the Python translator into the base facts:
%   constraint_selector(SelID, SelectorName)
%   constraint_first(SelID)
%   constraint_not_consecutive(SelID)
%   constraint_tool_at_step(Step, SelID)
%   constraint_max_uses(SelID, Limit)
%   constraint_unique_inputs(SelID)
%   constraint_implies_future(SelA_ID, SelB_ID)
%   constraint_depends_prior(SelA_ID, SelB_ID)
%   constraint_forbid_later(SelA_ID, SelB_ID)
%   constraint_next(SelA_ID, SelB_ID)
%   constraint_used_iff_used(SelA_ID, SelB_ID)
%   constraint_mutex(SelA_ID, SelB_ID)
%   constraint_connected(SelA_ID, SelB_ID)
%   must_use(Selector)      -- handled by tool_inclusion.lp
%   must_not_use(Selector)  -- handled by tool_inclusion.lp

% Per-step selector resolution.
% Grounded once per horizon; constraint_selected_tool(T, Sel, Tool) and
% constraint_selected_time(T, Sel) accumulate across all horizons so that
% check(t) can reason over the full workflow built so far.
#program constraint_step(t).

constraint_selected_tool(t, Sel, Tool) :-
    constraint_selector(Sel, Tool),
    occurs(t, run(Tool)).

constraint_selected_tool(t, Sel, Tool) :-
    constraint_selector(Sel, Class),
    tool_abstract_class(Class),
    occurs(t, run(Tool)),
    tool_is_a(Tool, Class).

constraint_selected_time(t, Sel) :-
    constraint_selected_tool(t, Sel, _).

% Integrity constraints checked at each horizon.
% All rules are guarded by query(t) so they only fire when the solver
% tests whether the workflow of length t is a valid solution.
% use_m / nuse_m are intentionally absent here — tool_inclusion.lp
% already enforces must_use / must_not_use in its own check(t) block.
#program check(t).

% first_m: selector must appear at T=1 and nowhere else.
:- query(t),
   constraint_first(Sel),
   not constraint_selected_time(1, Sel).

:- query(t),
   constraint_first(Sel),
   constraint_selected_time(T, Sel),
   T > 1.

% at_step: selector must appear at the given step.
:- query(t),
   constraint_tool_at_step(Step, Sel),
   not constraint_selected_time(Step, Sel).

% ite_m: if A appears, some B must appear at a later step.
:- query(t),
   constraint_implies_future(SelA, SelB),
   constraint_selected_time(T1, SelA),
   #count { T2 : constraint_selected_time(T2, SelB), T2 > T1 } = 0.

% depend_m: if A appears, some B must have appeared at an earlier step.
:- query(t),
   constraint_depends_prior(SelA, SelB),
   constraint_selected_time(T1, SelA),
   #count { T2 : constraint_selected_time(T2, SelB), T2 < T1 } = 0.

% itn_m: if A appears, B cannot appear at any later step.
:- query(t),
   constraint_forbid_later(SelA, SelB),
   constraint_selected_time(T1, SelA),
   constraint_selected_time(T2, SelB),
   T2 > T1.

% next_m: if A appears at T, B must appear at T+1.
:- query(t),
   constraint_next(SelA, SelB),
   constraint_selected_time(T1, SelA),
   #count { T2 : constraint_selected_time(T2, SelB), T2 = T1 + 1 } = 0.

% connected_op: some later B must bind an artifact produced by some earlier A.
:- query(t),
   constraint_connected(SelA, SelB),
   #count {
       T1, T2 :
           constraint_selected_tool(T1, SelA, ToolA),
           constraint_selected_tool(T2, SelB, ToolB),
           T1 < T2,
           occurs(T1, output(ToolA, _, Port)),
           occurs(T2, bind(ToolB, _, out(T1, ToolA, Port)))
   } = 0.

% used_iff_used: A appears iff B appears.
:- query(t),
   constraint_used_iff_used(SelA, SelB),
   constraint_selected_time(_, SelA),
   not constraint_selected_time(_, SelB).

:- query(t),
   constraint_used_iff_used(SelA, SelB),
   constraint_selected_time(_, SelB),
   not constraint_selected_time(_, SelA).

% max_uses: selector may appear at most N times.
:- query(t),
   constraint_max_uses(Sel, Max),
   #count { T : constraint_selected_time(T, Sel) } > Max.

% mutex_tools: A and B may not both appear in the same workflow.
:- query(t),
   constraint_mutex(SelA, SelB),
   constraint_selected_time(_, SelA),
   constraint_selected_time(_, SelB).

% not_consecutive: selector may not appear in adjacent steps.
:- query(t),
   constraint_not_consecutive(Sel),
   constraint_selected_time(T, Sel),
   Next = T + 1,
   constraint_selected_time(Next, Sel).

% unique_inputs: a matching tool run may not bind the same workflow artifact
% to two different input ports.
:- query(t),
   constraint_unique_inputs(Sel),
   constraint_selected_tool(T, Sel, Tool),
   occurs(T, bind(Tool, P1, WF)),
   occurs(T, bind(Tool, P2, WF)),
   P1 != P2.
