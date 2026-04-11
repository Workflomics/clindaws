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

1 { chosen_goal_output(t, GoalID, WF) : goal_satisfied_by_output(t, GoalID, WF) } 1 :-
    query(t),
    goal_id(GoalID).

% ----------------------------------------------------------------
% Output-dimension canonicalization
% Keep the smallest terminal value that still satisfies all actual downstream
% consumers and any goal the produced output satisfies at horizon t.
% Restrict the volatile check to profile/category pairs whose retained
% optimized candidates remain distinguishable for some consumer or goal.
% ----------------------------------------------------------------

output_dim_check_candidate(t, TProd, ToolProd, Port, V, Cat) :-
    query(t),
    occurs(TProd, output(ToolProd, _, Port)),
    direct_output_port_profile(Port, Profile),
    direct_output_profile_category_requires_check(Profile, Cat),
    direct_output_profile_candidate_requires_check(Profile, V, Cat).

output_dim_violates_consumer(t, TProd, ToolProd, Port, V, Cat) :-
    output_dim_check_candidate(t, TProd, ToolProd, Port, V, Cat),
    direct_output_port_profile(Port, Profile),
    output_used_by(TProd, ToolProd, Port, TUse, ToolUse, PortIn),
    direct_port_signature(PortIn, Signature),
    direct_signature_category(Signature, Cat),
    not direct_output_profile_candidate_supports_signature(Profile, Signature, V, Cat).

output_dim_violates_goal(t, TProd, ToolProd, Port, V, Cat) :-
    output_dim_check_candidate(t, TProd, ToolProd, Port, V, Cat),
    direct_output_port_profile(Port, Profile),
    chosen_goal_output(t, GoalID, out(TProd, ToolProd, Port)),
    direct_goal_category(GoalID, Cat),
    not direct_output_profile_candidate_supports_goal(Profile, GoalID, V, Cat).

output_dim_supported(t, TProd, ToolProd, Port, V, Cat) :-
    output_dim_check_candidate(t, TProd, ToolProd, Port, V, Cat),
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
