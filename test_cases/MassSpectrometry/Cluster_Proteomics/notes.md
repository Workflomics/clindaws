# Cluster_Proteomics — Known Issues

## This test case is currently UNSAT

Synthesis produces zero solutions at all horizons (5–10). Three stacked root causes:

---

### Root Cause 1 (definitive): `phyml` output has no consumer

`phyml` outputs `data_0872` (phylogenetic tree / `format_1910` Newick). No tool in
`bio.tools_proteomics_domain.json` accepts `data_0872` as input.

With `"use_all_generated_data": "one"`, every tool output must be consumed by a
downstream step. Since `phyml`'s output can never be consumed, the `use_m(phyml)`
constraint alone makes the problem unsatisfiable regardless of workflow length.

**Fix options:**
- Remove `use_m(phyml)` from constraints (if phylogenetic analysis is not required).
- Add tools to the annotation set that consume `data_0872` and eventually produce
  `data_0943` (mass spectrum), or relax `use_all_generated_data`.

---

### Root Cause 2: `ipk_blast` → `MAFFT` format mismatch

- `ipk_blast` outputs `data_0863` / `format_1921` (BLAST tabular)
- `MAFFT` requires `data_0863` / `format_1929` (FASTA)

In the EDAM hierarchy `format_1929` is a *subtype* of `format_1921`, so the output
type is strictly more general than what `MAFFT` expects — no direct connection is
possible. No format-converter tool in the current annotation set bridges this gap.

**Fix options:**
- Annotate `ipk_blast` to output `format_1929` (if its actual output supports FASTA).
- Add a converter tool (e.g. blast_formatter) that takes `data_0863`/`format_1921`
  and emits `data_0863`/`format_1929`.

---

### Root Cause 3: Required output type unreachable from constrained tools

The workflow output is `data_0943` / `format_3712` (mass spectrum, Thermo RAW).
`phyml` outputs `data_0872` (phylogenetic tree) and `data_0943` / `data_0872` sit in
completely separate branches of the EDAM data hierarchy. No tool in the proteomics
domain set converts phylogenetic-tree data into mass spectrum data. Only `msconvert`
produces `format_3712`, and it requires `data_0943` as input — it cannot be reached
from the `ipk_blast`→`MAFFT`→`phyml` chain.

---

### Additional fix applied (2026-04-12)

The four tools named in the constraints were missing from `bio.tools_proteomics_domain.json`
(upstream bug — identical in https://github.com/sanctuuary/APE_UseCases), causing APE to
abort with "tool not recognized" before synthesis even started. They were added from
`full_bio.tools.json`:

| Tool | Operation |
|------|-----------|
| `ipk_blast` | `operation_0346` (Sequence similarity search) |
| `MAFFT` | `operation_0492` (Multiple sequence alignment) |
| `phyml` | `operation_0547` (Phylogenetic tree construction) |
| `hmmer3_op4` | `operation_0336` / `operation_3434` (Format validation / Conversion) |

This unblocks APE from crashing, but synthesis remains UNSAT for the reasons above.
