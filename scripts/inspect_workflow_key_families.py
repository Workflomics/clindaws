#!/usr/bin/env python3
"""Inspect structural-vs-parity workflow key families from saved answer sets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import clingo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clindaws.core.config import load_config
from clindaws.core.workflow import extract_workflow_key_bundle, workflow_signature_length
from clindaws.execution.runner import _tool_output_dims_lookup, _workflow_input_dims_from_config
from clindaws.core.tool_annotations import (
    load_candidate_tool_annotations,
    load_direct_tool_annotations,
)
from clindaws.translators.signatures import _tool_input_signatures


def _tool_input_signatures_with_label_aliases(tools: tuple) -> dict[str, tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]]:
    tool_input_signatures = _tool_input_signatures(tools)
    signatures_by_label: dict[str, set[tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]]] = defaultdict(set)
    for tool in tools:
        signatures_by_label[str(tool.label)].add(tool_input_signatures[tool.mode_id])
    for label, signatures in signatures_by_label.items():
        if len(signatures) == 1:
            tool_input_signatures.setdefault(label, next(iter(signatures)))
    return tool_input_signatures


def _load_answer_sets(path: Path) -> list[tuple[clingo.Symbol, ...]]:
    answer_sets: list[tuple[clingo.Symbol, ...]] = []
    current_terms: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if current_terms:
                answer_sets.append(tuple(clingo.parse_term(term) for term in current_terms))
                current_terms = []
            continue
        if line.startswith("Answer Set "):
            if current_terms:
                answer_sets.append(tuple(clingo.parse_term(term) for term in current_terms))
                current_terms = []
            continue
        current_terms.extend(line.split())

    if current_terms:
        answer_sets.append(tuple(clingo.parse_term(term) for term in current_terms))
    return answer_sets


def _format_key(key: tuple[object, ...], *, limit: int = 220) -> str:
    text = repr(key)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="APE config json used for the run")
    parser.add_argument("--answer-sets", required=True, type=Path, help="Saved answer_sets__*.txt artifact")
    parser.add_argument("--optimized-annotations", action="store_true", help="Load optimized/candidate tool annotations instead of direct annotations")
    parser.add_argument("--horizon", type=int, help="Only inspect answer sets with this workflow length")
    parser.add_argument("--top", type=int, default=10, help="How many split/merge families to print")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.optimized_annotations:
        tools = load_candidate_tool_annotations(config.tool_annotations_path, config.ontology_prefix)
    else:
        tools = load_direct_tool_annotations(config.tool_annotations_path, config.ontology_prefix)

    tool_input_signatures = _tool_input_signatures_with_label_aliases(tools)
    workflow_input_dims = _workflow_input_dims_from_config(config)
    tool_output_dims = _tool_output_dims_lookup(tools)

    structural_to_parity: dict[tuple[object, ...], set[tuple[object, ...]]] = defaultdict(set)
    parity_to_structural: dict[tuple[object, ...], set[tuple[object, ...]]] = defaultdict(set)
    structural_counts: dict[tuple[object, ...], int] = defaultdict(int)
    parity_counts: dict[tuple[object, ...], int] = defaultdict(int)

    total = 0
    kept = 0
    for answer_set in _load_answer_sets(args.answer_sets):
        total += 1
        keys = extract_workflow_key_bundle(
            answer_set,
            tool_input_signatures,
            workflow_input_dims,
            tool_output_dims,
        )
        if args.horizon is not None and workflow_signature_length(keys.tool_sequence_key) != args.horizon:
            continue
        kept += 1
        structural_to_parity[keys.structural_workflow_key].add(keys.parity_workflow_key)
        parity_to_structural[keys.parity_workflow_key].add(keys.structural_workflow_key)
        structural_counts[keys.structural_workflow_key] += 1
        parity_counts[keys.parity_workflow_key] += 1

    print(f"answer sets loaded: {total}")
    if args.horizon is not None:
        print(f"answer sets kept at horizon {args.horizon}: {kept}")
    else:
        print(f"answer sets kept: {kept}")
    print(f"unique structural keys: {len(structural_counts)}")
    print(f"unique parity keys:     {len(parity_counts)}")

    parity_collapses = [
        (parity_key, structural_keys)
        for parity_key, structural_keys in parity_to_structural.items()
        if len(structural_keys) > 1
    ]
    structural_splits = [
        (structural_key, parity_keys)
        for structural_key, parity_keys in structural_to_parity.items()
        if len(parity_keys) > 1
    ]

    print(f"parity-collapsed families: {len(parity_collapses)}")
    print(f"structural keys split by parity: {len(structural_splits)}")

    if parity_collapses:
        print("\nTop parity-collapsed families:")
        for parity_key, structural_keys in sorted(
            parity_collapses,
            key=lambda item: (-len(item[1]), -parity_counts[item[0]], _format_key(item[0], limit=120)),
        )[: args.top]:
            print(
                f"- parity_count={parity_counts[parity_key]} structural_variants={len(structural_keys)} "
                f"parity_key={_format_key(parity_key)}"
            )
            for structural_key in sorted(structural_keys, key=lambda key: (-structural_counts[key], _format_key(key, limit=120)))[: args.top]:
                print(
                    f"  structural_count={structural_counts[structural_key]} "
                    f"structural_key={_format_key(structural_key)}"
                )

    if structural_splits:
        print("\nTop structural-to-parity splits:")
        for structural_key, parity_keys in sorted(
            structural_splits,
            key=lambda item: (-len(item[1]), -structural_counts[item[0]], _format_key(item[0], limit=120)),
        )[: args.top]:
            print(
                f"- structural_count={structural_counts[structural_key]} parity_variants={len(parity_keys)} "
                f"structural_key={_format_key(structural_key)}"
            )
            for parity_key in sorted(parity_keys, key=lambda key: (-parity_counts[key], _format_key(key, limit=120)))[: args.top]:
                print(
                    f"  parity_count={parity_counts[parity_key]} "
                    f"parity_key={_format_key(parity_key)}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
