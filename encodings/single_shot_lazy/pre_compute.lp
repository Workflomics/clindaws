% PRE-COMPUTE — lazy-schema single-shot static derivations

ancestor(Child, Parent) :- taxonomy(_, _, (_, Child, Parent)).
ancestor(Child, Ancestor) :- ancestor(Child, Mid), ancestor(Mid, Ancestor).

is_terminal(V) :- taxonomy(_, _, (_, V, _)), not taxonomy(_, _, (_, _, V)).

compatible(V, V)        :- is_terminal(V).
compatible(V, Ancestor) :- is_terminal(V), ancestor(V, Ancestor).

tool_has_input(Tool) :-
    lazy_tool_candidate(Candidate, Tool),
    lazy_candidate_input_port(Candidate, _).

tool_has_output(Tool) :-
    lazy_tool_candidate(Candidate, Tool),
    lazy_candidate_output_port(Candidate, _).

lazy_candidate_input_category(Candidate, Port, Category) :-
    lazy_candidate_input_value(Candidate, Port, _, Category).

eligible(0, Candidate, Port, WF) :-
    lazy_initial_bindable(Candidate, Port, WF).
