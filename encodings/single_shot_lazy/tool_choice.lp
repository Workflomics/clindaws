% TOOL CHOICE — lazy-schema single-shot run choice and binding

{ occurs(T, run(Tool)) } :- time(T), tool(Tool), not tool_has_input(Tool).
{ occurs(T, run(Tool)) } :- time(T), tool(Tool), tool_has_input(Tool).

1 { occurs(T, use_lazy_candidate(Tool, Candidate)) : lazy_tool_candidate(Candidate, Tool) } 1 :-
    occurs(T, run(Tool)).

occurs(T, run(Tool)) :- occurs(T, use_lazy_candidate(Tool, _)).

% First-step binding comes from eligible(0, ...) in pre_compute.lp.
% Later runs still depend on eligible(T-1), so generic eligibility must be
% available starting at T=1 to support runs at T=2 and beyond.
lazy_port_category_match(T, Candidate, Port, WF, Category) :-
    holds(T, avail(WF)),
    lazy_candidate_input_value(Candidate, Port, ReqV, Category),
    holds(T, dim(WF, ActualV, Category)),
    compatible(ReqV, ActualV).

eligible(T, Candidate, Port, WF) :-
    holds(T, avail(WF)),
    lazy_candidate_input_port(Candidate, Port),
    lazy_port_category_match(T, Candidate, Port, WF, Category) :
        lazy_candidate_input_category(Candidate, Port, Category).

1 { occurs(T, bind(Tool, Port, WF)) : eligible(T-1, Candidate, Port, WF) } 1 :-
    occurs(T, use_lazy_candidate(Tool, Candidate)),
    lazy_candidate_input_port(Candidate, Port).

artifact_parent(out(T, Tool, PortOut), WF) :-
    occurs(T, output(Tool, _, PortOut)),
    occurs(T, bind(Tool, _, WF)).

artifact_ancestor(Artifact, Ancestor) :-
    artifact_parent(Artifact, Ancestor).

artifact_ancestor(Artifact, Ancestor) :-
    artifact_parent(Artifact, Mid),
    artifact_ancestor(Mid, Ancestor).

:- occurs(T, use_lazy_candidate(Tool, Candidate)),
   occurs(T, bind(Tool, Port1, out(T1, Tool1, _))),
   occurs(T, bind(Tool, Port2, out(T2, Tool2, _))),
   Port1 < Port2,
   same_candidate_input_signature(Candidate, Port1, Candidate, Port2),
   T1 < T2.

paired_symmetric_producers(T1, Tool1, Candidate1, Port1, WF1, T2, Tool2, Candidate2, Port2, WF2) :-
    T1 < T2,
    occurs(T1, use_lazy_candidate(Tool1, Candidate1)),
    occurs(T2, use_lazy_candidate(Tool2, Candidate2)),
    candidate_single_input(Candidate1),
    candidate_single_input(Candidate2),
    occurs(T1, bind(Tool1, Port1, WF1)),
    occurs(T2, bind(Tool2, Port2, WF2)),
    same_candidate_input_signature(Candidate1, Port1, Candidate2, Port2),
    occurs(TJ, use_lazy_candidate(JoinTool, JoinCandidate)),
    occurs(TJ, bind(JoinTool, JoinPort1, out(T1, Tool1, _))),
    occurs(TJ, bind(JoinTool, JoinPort2, out(T2, Tool2, _))),
    TJ > T2,
    JoinPort2 < JoinPort1,
    same_candidate_input_signature(JoinCandidate, JoinPort1, JoinCandidate, JoinPort2).

:- paired_symmetric_producers(T1, Tool1, Candidate1, Port1, WF, T2, Tool2, Candidate2, Port2, WF).

:- paired_symmetric_producers(T1, Tool1, Candidate1, Port1, WF1, T2, Tool2, Candidate2, Port2, WF2),
   artifact_ancestor(WF1, WF2).
