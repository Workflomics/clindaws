% INPUT USAGE CONSTRAINTS
% Activation: enable_all_inputs_used. / enable_some_input_used. / enable_inputs_consumed.
% Docs: DOCS/constraints/05_input_usage.md

input_used(WF) :-
    holds(0, avail(WF)),
    occurs(_, bind(_, _, WF)).

:- enable_all_inputs_used,
   holds(0, avail(WF)),
   not input_used(WF).

:- enable_some_input_used,
   #count { WF : holds(0, avail(WF)), input_used(WF) } = 0.

consumed(T, WF) :-
    enable_inputs_consumed,
    holds(0, avail(WF)),
    occurs(T, bind(_, _, WF)).

:- enable_inputs_consumed,
   consumed(T1, WF),
   occurs(T2, bind(_, _, WF)),
   T2 > T1.
