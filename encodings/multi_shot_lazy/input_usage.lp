#program check(t).

:- query(t),
   enable_all_inputs_used,
   holds(0, avail(WF)),
   #count { T, Tool, Port : occurs(T, bind(Tool, Port, WF)) } = 0.

:- query(t),
   enable_inputs_used_once,
   holds(0, avail(WF)),
   #count { T, Tool, Port : occurs(T, bind(Tool, Port, WF)) } > 1.

:- query(t),
   enable_inputs_consumed,
   holds(0, avail(WF)),
   #count { T, Tool, Port : occurs(T, bind(Tool, Port, WF)) } > 1.
