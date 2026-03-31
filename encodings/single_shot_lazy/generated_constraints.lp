% GENERATED CONSTRAINTS
% Activation: translated constraint facts emitted into translation.lp
% This file is loaded only for single-shot-lazy runs.

% Resolve translated selectors to concrete tool runs.
constraint_selected_tool(T, Sel, Tool) :-
    constraint_selector(Sel, Tool),
    occurs(T, run(Tool)).

constraint_selected_tool(T, Sel, Tool) :-
    constraint_selector(Sel, Class),
    tool_abstract_class(Class),
    occurs(T, run(Tool)),
    tool_is_a(Tool, Class).

constraint_selected_time(T, Sel) :-
    constraint_selected_tool(T, Sel, _).

constraint_used(Sel) :-
    constraint_selected_time(_, Sel).

constraint_connected_pair(SelA, SelB, T1, T2) :-
    constraint_connected(SelA, SelB),
    constraint_selected_tool(T1, SelA, ToolA),
    constraint_selected_tool(T2, SelB, ToolB),
    T1 < T2,
    occurs(T1, output(ToolA, _, Port)),
    occurs(T2, bind(ToolB, _, out(T1, ToolA, Port))).

% first_m: selector must be present at the first timestep and nowhere later.
:- constraint_first(Sel),
   not constraint_selected_time(1, Sel).

:- constraint_first(Sel),
   constraint_selected_time(T, Sel),
   T > 1.

% tool_has_to_be: selector must be present at the specified timestep.
:- constraint_tool_at_step(Step, Sel),
   not constraint_selected_time(Step, Sel).

% ite_m: if selector A occurs, selector B must occur later.
:- constraint_implies_future(SelA, SelB),
   constraint_selected_time(T1, SelA),
   #count { T2 : constraint_selected_time(T2, SelB), T2 > T1 } = 0.

% depend_m: if selector A occurs, selector B must have occurred earlier.
:- constraint_depends_prior(SelA, SelB),
   constraint_selected_time(T1, SelA),
   #count { T2 : constraint_selected_time(T2, SelB), T2 < T1 } = 0.

% itn_m: if selector A occurs, selector B cannot occur later.
:- constraint_forbid_later(SelA, SelB),
   constraint_selected_time(T1, SelA),
   constraint_selected_time(T2, SelB),
   T2 > T1.

% next_m: if selector A occurs, selector B must occur at the next timestep.
:- constraint_next(SelA, SelB),
   constraint_selected_time(T1, SelA),
   #count { T2 : constraint_selected_time(T2, SelB), T2 = T1 + 1 } = 0.

% connected_op: some later matching consumer must bind an artifact produced by
% an earlier matching producer.
:- constraint_connected(SelA, SelB),
   #count { T1, T2 : constraint_connected_pair(SelA, SelB, T1, T2) } = 0.

% used_iff_used: both selectors either appear together or stay absent.
:- constraint_used_iff_used(SelA, SelB),
   constraint_used(SelA),
   not constraint_used(SelB).

:- constraint_used_iff_used(SelA, SelB),
   constraint_used(SelB),
   not constraint_used(SelA).

% max_uses: selector cannot appear more than the configured number of times.
:- constraint_max_uses(Sel, Max),
   #count { T : constraint_selected_time(T, Sel) } > Max.

% mutex_tools: selectors cannot both appear anywhere in the workflow.
:- constraint_mutex(SelA, SelB),
   constraint_used(SelA),
   constraint_used(SelB).

% not_consecutive: selector cannot appear in adjacent steps.
:- constraint_not_consecutive(Sel),
   constraint_selected_time(T, Sel),
   Next = T + 1,
   constraint_selected_time(Next, Sel).

% unique_inputs: a matching tool run cannot bind the same workflow item twice.
:- constraint_unique_inputs(Sel),
   constraint_selected_tool(T, Sel, Tool),
   occurs(T, bind(Tool, P1, WF)),
   occurs(T, bind(Tool, P2, WF)),
   P1 != P2.
