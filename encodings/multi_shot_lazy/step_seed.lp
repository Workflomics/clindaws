#program step_seed(t).

% Seed eligible(t, ...) after the specialized first step has been applied.
% This is grounded only when the next horizon is requested, so horizon-1-only
% runs keep the fast lazy_initial_bindable path from step_initial.lp.
lazy_port_category_match(t, Candidate, Port, WF, Category) :-
    holds(t, avail(WF)),
    lazy_candidate_input_value(Candidate, Port, ReqV, Category),
    holds(t, dim(WF, ActualV, Category)),
    compatible(ReqV, ActualV).

eligible(t, Candidate, Port, WF) :-
    holds(t, avail(WF)),
    lazy_candidate_input_port(Candidate, Port),
    lazy_port_category_match(t, Candidate, Port, WF, Category) :
        lazy_candidate_input_category(Candidate, Port, Category).
