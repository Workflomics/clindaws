#program init.

% ================================================================
% INIT — timestep-0 derivations grounded once for multi-shot
% ================================================================

port_cat_ok(0, Port, WF, Cat) :-
    holds(0, avail(WF)),
    port_requires(Port, ReqV, Cat),
    holds(0, dim(WF, V, Cat)),
    compatible(V, ReqV).

eligible(0, Port, WF) :-
    holds(0, avail(WF)),
    input_port(_, Port),
    port_cat_ok(0, Port, WF, Cat) : port_requires(Port, _, Cat).
