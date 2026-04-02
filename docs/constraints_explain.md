# Constraints In `snakeAPE`

`snakeAPE` currently uses a single config entry for constraints:

- `constraints_path`

`constraints_path` is currently wired for the benchmark-targeted modes:

- `single-shot`
- `single-shot-lazy`
- `multi-shot`
- `multi-shot-lazy`

The file behind `constraints_path` can be one of two shapes:

- APE-style objects from a `constraints.json` file
- Native atom strings from a `constraint_asp.json` file

One file must use one shape only. Mixing native strings and APE-style objects in the same `constraints` list is rejected.

## How Constraint Selection Works

Constraint selectors refer to tools or tool classes from the ontology and tool annotations.

- Concrete tool class example: `CreateProject`
- Abstract tool class example: `RunDFT`
- Native examples:
  - `at_step(CreateProject, 1)`
  - `connected_op(CreateStructureBulk, RunDFT)`
  - `mutex_tools(RunVasp, RunSphinx)`

Selectors are matched against the taxonomy-aware `constraint_selected_tool/3` relation in the lazy ASP layer, so a class selector can match any concrete tool implementation below it.

## APE-Style Templates From `constraints.json`

These are read from `constraints_path` and translated into internal ASP facts.

- `use_m(A)`: selector `A` must appear somewhere
- `nuse_m(A)`: selector `A` must not appear
- `ite_m(A, B)`: if `A` appears, some `B` must appear later
- `depend_m(A, B)`: if `A` appears, some `B` must appear earlier
- `itn_m(A, B)`: if `A` appears, `B` cannot appear later
- `next_m(A, B)`: if `A` appears, some `B` must appear at the next timestep
- `use_t(X)`: some workflow artifact must carry data selector `X`
- `unique_inputs(A)`: a matching tool run may not bind the same workflow artifact to two different input ports
- `first_m(A)`: `A` must occur at timestep 1 and nowhere later
- `connected_op(A, B)`: some later matching `B` must bind an artifact produced by some earlier matching `A`
- `operationInput(A, X)`: some matching `A` run must bind an artifact carrying data selector `X`

Current limitations:

- General `SLTLx` formulas are not parsed
- Unsupported template IDs are skipped with a translation comment

## Native Atoms Via `constraints_path`

If `constraints_path` points at a native file, the file shape is:

```json
{
  "constraints": [
    "at_step(CreateProject, 1)",
    "itn_m(CreateProject, CreateProject)",
    "connected_op(CreateStructureBulk, RunDFT)"
  ]
}
```

Supported native atoms:

- `at_step(A, Step)`: selector `A` must occur at exactly `Step`
- `use_m(A)`: selector `A` must appear somewhere
- `nuse_m(A)`: selector `A` must not appear
- `ite_m(A, B)`: if `A` appears, some `B` must appear later
- `depend_m(A, B)`: if `A` appears, some `B` must appear earlier
- `itn_m(A, B)`: if `A` appears, `B` cannot appear later
- `next_m(A, B)`: if `A` appears, some `B` must appear at the next timestep
- `use_t(X)`: some workflow artifact must carry data selector `X`
- `first_m(A)`: `A` must occur at timestep 1 and nowhere later
- `unique_inputs(A)`: matching tool runs cannot reuse one workflow artifact on two input ports
- `connected_op(A, B)`: some later matching `B` must bind an artifact produced by some earlier matching `A`
- `operation_input(A, X)`: some matching `A` run must bind an artifact carrying data selector `X`
- `used_iff_used(A, B)`: `A` appears iff `B` appears
- `max_uses(A, N)`: selector `A` may appear at most `N` times
- `mutex_tools(A, B)`: `A` and `B` may not both appear in the same workflow
- `not_consecutive(A)`: `A` may not appear in adjacent timesteps

Notes:

- `at_step(A, 1)` only constrains the required timestep. If you also need “and never later”, combine it with another constraint such as `itn_m(A, A)` or use `first_m(A)`.
- Native atoms are not arbitrary ASP rules. They are parsed as restricted atoms and lowered by the translator.
- Data selectors are matched ontology-aware when the value exists in the ontology, and by exact string equality otherwise. This exact-match fallback is what makes `APE_label` values such as `cities` and `birds` usable in GeoGMT constraints.

## What Is Implemented Today

Implemented in the current benchmark-targeted paths:

- APE-style templates listed above
- Native atoms listed above
- Taxonomy-aware selector matching for concrete tools and abstract tool classes
- Data-selector constraints over workflow artifacts, including non-ontology labels such as `APE_label`
- Native `connected_op` with actual data-flow semantics, not just ordering

Not implemented:

- General `SLTLx` parsing
- Arbitrary raw ASP snippets inside JSON
- Guaranteed support for `single-shot-opt` or `multi-shot-opt`

## Defect Concentration Example

The current defect example uses:

- `constraints_path = ./defect_concentration/constraint_asp.json`

The important native entries are:

- `at_step(CreateProject, 1)`
- `itn_m(CreateProject, CreateProject)`
- `connected_op(CreateStructureBulk, RunDFT)`
- `used_iff_used(CreateAntisite, CalcChemicalPotentialB)`
- `mutex_tools(RunVasp, RunSphinx)`

This setup matches the APE SAT baseline for `defect_concentration` in `single-shot-lazy` at horizons 9 and 10.
