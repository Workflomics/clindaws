`operation_input` constraints were removed from the SAT file because the Java
APE version in use rejects that constraint id in JSON.

The SAT failure was:

`java -jar APE-2.6.1-executable.jar synthesis test_cases/GeoGMT/E1/config_SAT_GeoGMT_E1_WC.json --sat --benchmark`

with:

`ConstraintFormatException ... constraint ID: operation_input`

ASP was updated to match the revised SAT constraint set exactly. The following
native ASP constraints were removed from `constraints_ASP_e4.json`:

- `operation_input(Draw_lines, cities)`
- `operation_input(Draw_points, birds)`
- `operation_input(Draw_points, cities)`

All remaining GeoGMT E4 workflow constraints stay aligned between SAT and ASP.
