% TOOL CHOICE — lazy-schema single-shot run choice and binding

{ occurs(T, run(Tool)) } :- time(T), tool(Tool), not tool_has_input(Tool).
{ occurs(T, run(Tool)) } :- time(T), tool(Tool), tool_has_input(Tool).

1 { occurs(T, use_lazy_candidate(Tool, Candidate)) : lazy_tool_candidate(Candidate, Tool) } 1 :-
    occurs(T, run(Tool)).

occurs(T, run(Tool)) :- occurs(T, use_lazy_candidate(Tool, _)).

lazy_port_category_match(T, Candidate, Port, WF, Category) :-
    holds(T, avail(WF)),
    lazy_candidate_input_value(Candidate, Port, ReqV, Category),
    holds(T, dim(WF, ReqV, Category)).

eligible(T, Candidate, Port, WF) :-
    holds(T, avail(WF)),
    lazy_candidate_input_port(Candidate, Port),
    lazy_port_category_match(T, Candidate, Port, WF, Category) :
        lazy_candidate_input_category(Candidate, Port, Category).

1 { occurs(T, bind(Tool, Port, WF)) : eligible(T-1, Candidate, Port, WF) } 1 :-
    occurs(T, use_lazy_candidate(Tool, Candidate)),
    lazy_candidate_input_port(Candidate, Port).
