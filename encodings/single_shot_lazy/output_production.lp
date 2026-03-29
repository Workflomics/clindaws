% OUTPUT PRODUCTION — lazy-schema single-shot outputs

occurs(T, output(Tool, Candidate, Port)) :-
    occurs(T, use_lazy_candidate(Tool, Candidate)),
    lazy_candidate_output_port(Candidate, Port).

holds(T, avail(out(T, Tool, Port))) :-
    occurs(T, output(Tool, _, Port)).

1 { occurs(T, output_dim(Tool, Port, V, Category)) :
    lazy_candidate_output_choice_value(Candidate, Port, V, Category) } 1 :-
    occurs(T, output(Tool, Candidate, Port)),
    lazy_candidate_output_choice_value(Candidate, Port, _, Category).

holds(T, dim(out(T, Tool, Port), V, Category)) :-
    occurs(T, output_dim(Tool, Port, V, Category)).

holds(T, dim(out(T, Tool, Port), V, Category)) :-
    occurs(T, output(Tool, Candidate, Port)),
    lazy_candidate_output_singleton(Candidate, Port, V, Category).
