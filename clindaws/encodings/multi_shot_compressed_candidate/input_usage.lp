#program check(t).

workflow_input_reference_check(t, Artifact, WF) :-
   query(t),
   holds(0, avail(WF)),
   Artifact = WF.

workflow_input_reference_check(t, Artifact, WF) :-
   query(t),
   holds(0, avail(WF)),
   workflow_input_lineage(t, Artifact, WF).

workflow_input_used_check(t, WF) :-
   query(t),
   holds(0, avail(WF)),
   occurs(T, bind(Tool, Port, Artifact)),
   workflow_input_reference_check(t, Artifact, WF).

workflow_input_used_check(t, WF) :-
   query(t),
   holds(0, avail(WF)),
   chosen_goal_output(t, _, Artifact),
   workflow_input_reference_check(t, Artifact, WF).

:- query(t),
   enable_all_inputs_used,
   holds(0, avail(WF)),
   not workflow_input_used_check(t, WF).

:- query(t),
   enable_some_input_used,
   #count { WF : workflow_input_used_check(t, WF) } = 0.

:- query(t),
   enable_inputs_consumed,
   holds(0, avail(WF)),
   #count { T, Tool, Port : occurs(T, bind(Tool, Port, WF)) } > 1.
