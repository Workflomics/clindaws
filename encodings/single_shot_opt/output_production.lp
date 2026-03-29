% OUTPUT PRODUCTION — candidate-schema single-shot outputs

occurs(T, output(Tool, TI_ID, Port)) :-
    occurs(T, use_variant(Tool, TI_ID)),
    candidate_output_port(TI_ID, Port).

holds(T, avail(out(T, Tool, Port))) :-
    occurs(T, output(Tool, _, Port)).

holds(T, dim(out(T, Tool, Port), V, Cat)) :-
    occurs(T, output(Tool, TI_ID, Port)),
    candidate_output_value(TI_ID, Port, V, Cat).
