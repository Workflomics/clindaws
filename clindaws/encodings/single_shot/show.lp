% SHOW — output display configuration
% Docs: DOCS/encoding/show.md

% #show is_terminal/1.
% #show compatible/2.
% #show tool_input_variant/2.
% #show port_requires/3.

% #show holds/2.
% #show occurs/2.
%
% stateholds(T, state(A,S)) :- holds(T,state(A,S)).
% #show stateholds/2.
% #show stateholds/2.(T, state(A,S)) : holds(T, state(A,S)).
% #show useful/2.
tool_at_time(T, Tool) :- occurs(T, run(Tool)).
#project tool_at_time/2.
#show tool_at_time/2.
% #show rdf_triple/3.
