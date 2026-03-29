% TEMPORAL CONSTRAINTS
% Activation: before/after/eventually_before/eventually_after facts in user_constraints.lp
% Docs: DOCS/constraints/02_temporal.md

% ----------------------------------------------------------------
% SELECTOR RESOLUTION
% tool_run_at(T, Selector) maps each selector to the timestep(s)
% where it is satisfied. See docs for all selector forms.
% ----------------------------------------------------------------

% Concrete tool name
tool_run_at(T, Tool) :-
    tool(Tool),
    occurs(T, run(Tool)).

% Abstract tool class — matches any member that runs
tool_run_at(T, Class) :-
    tool_abstract_class(Class),
    occurs(T, run(Tool)),
    tool_is_a(Tool, Class).

% Tool with a specific input type
tool_run_at(T, with_input_type(Tool, TypeVal)) :-
    occurs(T, run(Tool)),
    occurs(T, bind(Tool, _, WF)),
    holds(T-1, dim(WF, TypeVal, type)).

% Tool with a specific input state
tool_run_at(T, with_input_state(Tool, State)) :-
    occurs(T, run(Tool)),
    occurs(T, bind(Tool, _, WF)),
    holds(T-1, dim(WF, State, state)).

% Tool with a specific output type
tool_run_at(T, with_output_type(Tool, TypeVal)) :-
    occurs(T, run(Tool)),
    occurs(T, output(Tool, _, Port)),
    holds(T, dim(out(T, Tool, Port), TypeVal, type)).

% Tool with a specific output state
tool_run_at(T, with_output_state(Tool, State)) :-
    occurs(T, run(Tool)),
    occurs(T, output(Tool, _, Port)),
    holds(T, dim(out(T, Tool, Port), State, state)).

% Tool with a specific input unit
tool_run_at(T, with_input_unit(Tool, Unit)) :-
    occurs(T, run(Tool)),
    occurs(T, bind(Tool, _, WF)),
    holds(T-1, unit(WF, Unit)).

% Tool with a given tool_property annotation
tool_run_at(T, with_property(Tool, Prop)) :-
    occurs(T, run(Tool)),
    tool_property(Tool, Prop).

% Tool with a specific requirement (type + value)
tool_run_at(T, with_requirement(Tool, ReqType, ReqVal)) :-
    occurs(T, run(Tool)),
    tool_requirement(Tool, (ReqType, ReqVal)).

% Collect selectors actually referenced in temporal constraints,
% and propagate into sub-selectors of and(...) terms.
% This prevents the and-rule from causing infinite grounding.
sel_used(Sel) :- before(Sel, _).
sel_used(Sel) :- before(_, Sel).
sel_used(Sel) :- after(Sel, _).
sel_used(Sel) :- after(_, Sel).
sel_used(Sel) :- eventually_before(Sel, _).
sel_used(Sel) :- eventually_before(_, Sel).
sel_used(Sel) :- eventually_after(Sel, _).
sel_used(Sel) :- eventually_after(_, Sel).
sel_used(SelA) :- sel_used(and(SelA, _)).
sel_used(SelB) :- sel_used(and(_, SelB)).

% Inline conjunction — only fires for and-selectors actually used in constraints
tool_run_at(T, and(SelA, SelB)) :-
    sel_used(and(SelA, SelB)),
    tool_run_at(T, SelA),
    tool_run_at(T, SelB).

% Named compound selector — every selector_cond must match at T
% occurs(T, run(_)) grounds T to timesteps where any tool runs,
% making T safe for use inside the conditional literal.
tool_run_at(T, Name) :-
    selector(Name),
    selector_cond(Name, _),
    occurs(T, run(_)),
    tool_run_at(T, Cond) : selector_cond(Name, Cond).

% ----------------------------------------------------------------
% ENFORCEMENT
% ----------------------------------------------------------------

% SelA at T, SelB at T+1 (directly adjacent)
:- before(SelA, SelB),
   tool_run_at(T1, SelA),
   tool_run_at(T2, SelB),
   T2 != T1 + 1.

% SelA at T, SelB at T-1 (directly adjacent)
:- after(SelA, SelB),
   tool_run_at(T1, SelA),
   tool_run_at(T2, SelB),
   T1 != T2 + 1.

% SelA at T1 strictly before SelB at T2 (any gap)
:- eventually_before(SelA, SelB),
   tool_run_at(T1, SelA),
   tool_run_at(T2, SelB),
   T1 >= T2.

% SelA at T1 strictly after SelB at T2 (any gap)
:- eventually_after(SelA, SelB),
   tool_run_at(T1, SelA),
   tool_run_at(T2, SelB),
   T1 <= T2.
