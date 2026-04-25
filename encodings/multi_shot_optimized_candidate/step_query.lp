#program step_query(t).

% STEP_QUERY — exact-horizon query-time output filtering
% When exact horizon mode is active, the current horizon can restrict itself to
% the subset of tools/candidates/output values that can still satisfy the goal.

query_output_filter_active(t) :-
    query(t),
    exact_horizon_mode,
    not enable_all_inputs_used,
    not enable_all_outputs_consumed.

exact_query_support_horizon(t, H) :-
    query_output_filter_active(t),
    exact_target_horizon(H).

exact_query_support_present(t) :-
    exact_query_support_horizon(t, _).

query_current_output_category_needed(t, Candidate, Port, Category) :-
    query_output_filter_active(t),
    dynamic_check_relevant_output_category_at_horizon(Candidate, Port, Category, t).

query_current_output_category_needed(t, Candidate, Port, Category) :-
    query_output_filter_active(t),
    goal_output(_, _, Category),
    dynamic_candidate_output_goal_match(Candidate, Port, _, Category).

query_current_output_category_needed(t, Candidate, Port, Category) :-
    query_output_filter_active(t),
    constraint_data_selector_category(_, Category),
    dynamic_candidate_output_category(Candidate, Port, Category).

query_current_output_category_needed(t, Candidate, Port, Category) :-
    query_output_filter_active(t),
    constraint_data_selector(Sel, Wanted),
    not constraint_data_selector_category(Sel, _),
    dynamic_candidate_output_singleton(Candidate, Port, Actual, Category),
    Actual = Wanted.

query_current_output_category_needed(t, Candidate, Port, Category) :-
    query_output_filter_active(t),
    constraint_data_selector(Sel, Wanted),
    not constraint_data_selector_category(Sel, _),
    dynamic_candidate_output_singleton(Candidate, Port, Actual, Category),
    ancestor(Actual, Wanted).

query_current_output_category_needed(t, Candidate, Port, Category) :-
    query_output_filter_active(t),
    constraint_data_selector(Sel, Wanted),
    not constraint_data_selector_category(Sel, _),
    dynamic_candidate_output_choice_value(Candidate, Port, Actual, Category),
    Actual = Wanted.

query_current_output_category_needed(t, Candidate, Port, Category) :-
    query_output_filter_active(t),
    constraint_data_selector(Sel, Wanted),
    not constraint_data_selector_category(Sel, _),
    dynamic_candidate_output_choice_value(Candidate, Port, Actual, Category),
    ancestor(Actual, Wanted).

dynamic_query_output_has_goal_value(Candidate, Port, Category) :-
    query_output_filter_active(t),
    query_current_output_category_needed(t, Candidate, Port, Category),
    dynamic_candidate_output_choice_value(Candidate, Port, V, Category),
    goal_output(_, GoalV, Category),
    compatible(V, GoalV).

dynamic_query_output_preferred_value(Candidate, Port, V, Category) :-
    query_output_filter_active(t),
    query_current_output_category_needed(t, Candidate, Port, Category),
    dynamic_candidate_output_choice_value(Candidate, Port, V, Category),
    dynamic_query_output_has_goal_value(Candidate, Port, Category),
    goal_output(_, GoalV, Category),
    compatible(V, GoalV),
    not dynamic_query_output_smaller_goal_value(Candidate, Port, V, Category).

dynamic_query_output_smaller_goal_value(Candidate, Port, V, Category) :-
    query_output_filter_active(t),
    query_current_output_category_needed(t, Candidate, Port, Category),
    dynamic_candidate_output_choice_value(Candidate, Port, V, Category),
    dynamic_candidate_output_choice_value(Candidate, Port, V2, Category),
    goal_output(_, GoalV, Category),
    compatible(V, GoalV),
    compatible(V2, GoalV),
    V2 < V.

dynamic_query_output_preferred_value(Candidate, Port, V, Category) :-
    query_output_filter_active(t),
    query_current_output_category_needed(t, Candidate, Port, Category),
    dynamic_candidate_output_choice_value(Candidate, Port, V, Category),
    not dynamic_query_output_has_goal_value(Candidate, Port, Category),
    not dynamic_query_output_smaller_value(Candidate, Port, V, Category).

dynamic_query_output_smaller_value(Candidate, Port, V, Category) :-
    query_output_filter_active(t),
    query_current_output_category_needed(t, Candidate, Port, Category),
    dynamic_candidate_output_choice_value(Candidate, Port, V, Category),
    dynamic_candidate_output_choice_value(Candidate, Port, V2, Category),
    V2 < V.

dynamic_query_output_value(Candidate, Port, V, Category) :-
    query_output_filter_active(t),
    dynamic_query_output_preferred_value(Candidate, Port, V, Category).

:- exact_query_support_horizon(t, H),
   occurs(t, run(Tool)),
   not dynamic_goal_support_tool_at_horizon(Tool, H).

:- query_output_filter_active(t),
   not exact_query_support_present(t),
   occurs(t, run(Tool)),
   not dynamic_query_goal_tool(Tool).

:- exact_query_support_horizon(t, H),
   occurs(t, use_dynamic_candidate(_, Candidate)),
   not dynamic_structurally_supportable_candidate_at_horizon(Candidate, H).

:- query_output_filter_active(t),
   not exact_query_support_present(t),
   occurs(t, use_dynamic_candidate(_, Candidate)),
   not dynamic_query_goal_candidate(Candidate).

1 { occurs(t, output_dim(Tool, Port, V, Category)) :
    dynamic_query_output_value(Candidate, Port, V, Category) } 1 :-
    query_output_filter_active(t),
    occurs(t, output(Tool, Candidate, Port)),
    query_current_output_category_needed(t, Candidate, Port, Category),
    dynamic_query_output_value(Candidate, Port, _, Category).
