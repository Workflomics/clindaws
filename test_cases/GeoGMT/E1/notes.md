`operation_input` constraints were removed from the SAT file because the Java
APE version in use rejects that constraint id in JSON.

The SAT failure was:

`java -jar APE-2.6.1-executable.jar synthesis test_cases/GeoGMT/E1/config_SAT_GeoGMT_E1_WC.json --sat --benchmark`

with:

`ConstraintFormatException ... constraint ID: operation_input`

ASP was updated to track the same effective constraint set as SAT:

- the broken ASP config path to the old missing
  `constraints_ASP_e1_SLTLx_e1.1-3.json` file was replaced
- the ASP case now uses `constraints_ASP_e1.json`
- the SAT-side `SLTLx` eventual-use formulas were mirrored as native
  `use_m(...)` constraints for:
  - `Draw_water`
  - `Draw_land`
  - `Draw_political_borders`
  - `Draw_lines`
  - `Draw_boundary_frame`
  - `Write_title`
  - `Draw_time_stamp_logo`

This keeps the E1 SAT and ASP workflow restrictions aligned without keeping the
old unsupported `operation_input` requirements on the ASP side.
