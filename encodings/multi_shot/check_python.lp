#program check(t).
#external query(t).

% ================================================================
% CHECK — volatile goal and usefulness check at horizon t
% Python-precomputed direct mode uses compact signature requirements here to
% avoid rebuilding consumer-port joins for repeated identical signatures.
% ================================================================

goal_id(GoalID) :- goal_output(GoalID, _, _).

% Hard goal constraint
:- query(t), not holds(t, goal).

% Only accept workflows at the first horizon where the goal is reached.
:- query(t), t > 1, holds(t-1, goal).

1 { chosen_goal_output(t, GoalID, WF) : goal_satisfied_by_output(t, GoalID, WF) } 1 :-
    query(t),
    goal_id(GoalID).

% ----------------------------------------------------------------
% Output-dimension canonicalization
% Keep the smallest terminal value that still satisfies all actual downstream
% consumers and any goal the produced output satisfies at horizon t.
% ----------------------------------------------------------------

output_dim_violates_consumer(t, TProd, ToolProd, Port, V, Cat) :-
    query(t),
    occurs(TProd, output(ToolProd, _, Port)),
    dimension(Port, (BaseV, Cat)),
    not is_terminal(BaseV),
    compatible(V, BaseV),
    is_terminal(V),
    output_used_by(TProd, ToolProd, Port, TUse, ToolUse, PortIn),
    direct_port_signature(PortIn, Signature),
    direct_signature_requires(Signature, ReqV, Cat),
    not compatible(V, ReqV).

chosen_output_goal_requirement(t, TProd, ToolProd, Port, GoalID, GoalV, Cat) :-
    query(t),
    chosen_goal_output(t, GoalID, out(TProd, ToolProd, Port)),
    goal_output(GoalID, GoalV, Cat).

output_dim_violates_goal(t, TProd, ToolProd, Port, V, Cat) :-
    query(t),
    occurs(TProd, output(ToolProd, _, Port)),
    dimension(Port, (BaseV, Cat)),
    not is_terminal(BaseV),
    compatible(V, BaseV),
    is_terminal(V),
    chosen_output_goal_requirement(t, TProd, ToolProd, Port, GoalID, GoalV, Cat),
    not compatible(V, GoalV).

output_dim_supported(t, TProd, ToolProd, Port, V, Cat) :-
    query(t),
    occurs(TProd, output(ToolProd, _, Port)),
    dimension(Port, (BaseV, Cat)),
    not is_terminal(BaseV),
    compatible(V, BaseV),
    is_terminal(V),
    not output_dim_violates_consumer(t, TProd, ToolProd, Port, V, Cat),
    not output_dim_violates_goal(t, TProd, ToolProd, Port, V, Cat).

output_dim_smaller_supported(t, TProd, ToolProd, Port, V, Cat) :-
    output_dim_supported(t, TProd, ToolProd, Port, V, Cat),
    output_dim_supported(t, TProd, ToolProd, Port, V2, Cat),
    V2 < V.

:- query(t),
   occurs(TProd, output_dim(ToolProd, Port, V, Cat)),
   output_dim_supported(t, TProd, ToolProd, Port, V, Cat),
   output_dim_smaller_supported(t, TProd, ToolProd, Port, V, Cat).

% ----------------------------------------------------------------
% Usefulness: full backward propagation at check time
% Namespaced as useful_check(t, T, Tool) to avoid stale atoms
% from previous check iterations polluting the model.
% ----------------------------------------------------------------

% Direct — output satisfies goal at current horizon
useful_check(t, T, Tool) :-
    query(t),
    occurs(T, run(Tool)),
    occurs(T, output(Tool, _, Port)),
    chosen_goal_output(t, _, out(T, Tool, Port)).

% Backward — output consumed by a useful tool
useful_check(t, T1, Tool1) :-
    query(t),
    occurs(T1, run(Tool1)),
    output_used_by(T1, Tool1, Port, T2, Tool2, ConsumerPort),
    useful_check(t, T2, Tool2),
    T2 > T1.

% Trivial — tools with no output
useful_check(t, T, Tool) :-
    query(t),
    occurs(T, run(Tool)),
    not tool_has_output(Tool).

% Constraint: every tool that runs must be useful
:- query(t),
   enable_usefulness_pruning,
   occurs(T, run(Tool)),
   not useful_check(t, T, Tool).

% SAT permits reusing outputs from earlier runs of the same tool, so do not
% forbid these chains here.
