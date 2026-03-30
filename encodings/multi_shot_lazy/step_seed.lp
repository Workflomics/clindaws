#program step_seed(t).

% Seed eligibility only for artifacts introduced at horizon t. Artifact
% compatibility is monotonic, so once an artifact is known to satisfy a lazy
% candidate port it stays eligible forever and does not need to be recomputed
% against the full carried-forward state at later horizons.
lazy_seed_port_category_match(t, Candidate, Port, out(t, Tool, PortOut), Category) :-
    occurs(t, output(Tool, _, PortOut)),
    lazy_candidate_input_value(Candidate, Port, ReqV, Category),
    artifact_dim(out(t, Tool, PortOut), ActualV, Category),
    compatible(ReqV, ActualV).

eligible(Candidate, Port, out(t, Tool, PortOut)) :-
    occurs(t, output(Tool, _, PortOut)),
    lazy_candidate_input_port(Candidate, Port),
    lazy_seed_port_category_match(t, Candidate, Port, out(t, Tool, PortOut), Category) :
        lazy_candidate_input_category(Candidate, Port, Category).
