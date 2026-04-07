#program check(t).

:- query(t),
   constraint_must_use(Sel),
   not constraint_selected_any(t, Sel).

:- query(t),
   constraint_must_not_use(Sel),
   constraint_selected_any(t, Sel).
