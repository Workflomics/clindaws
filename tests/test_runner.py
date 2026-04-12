import unittest
from types import SimpleNamespace
from unittest.mock import patch

from clindaws.core.models import FactBundle, HorizonRecord
from clindaws.execution.runner import (
    COMPRESSED_CANDIDATE_TRANSLATION_BUILDER,
    _mode_config,
    _select_fact_bundle,
    _solve_callback_profile_payload,
)


def _fact_bundle(*, internal_schema: str = "", internal_solver_mode: str = "") -> FactBundle:
    return FactBundle(
        facts="",
        fact_count=1,
        tool_labels={},
        tool_input_signatures={},
        workflow_input_ids=(),
        goal_count=0,
        predicate_counts={},
        tool_stats=(),
        cache_stats={},
        emit_stats={},
        internal_schema=internal_schema,
        internal_solver_mode=internal_solver_mode,
    )


class RunnerSummaryTests(unittest.TestCase):
    def test_solve_callback_profile_aggregates_horizon_timings(self) -> None:
        payload = _solve_callback_profile_payload(
            (
                HorizonRecord(
                    horizon=1,
                    grounding_sec=0.1,
                    solving_sec=1.0,
                    peak_rss_mb=10.0,
                    satisfiable=False,
                    models_seen=2,
                    models_stored=0,
                    unique_workflows_seen=1,
                    unique_workflows_stored=0,
                    model_callback_sec=0.3,
                    shown_symbols_sec=0.1,
                    workflow_signature_key_sec=0.05,
                    canonicalization_sec=0.1,
                ),
                HorizonRecord(
                    horizon=2,
                    grounding_sec=0.2,
                    solving_sec=2.0,
                    peak_rss_mb=11.0,
                    satisfiable=True,
                    models_seen=3,
                    models_stored=1,
                    unique_workflows_seen=2,
                    unique_workflows_stored=1,
                    model_callback_sec=0.5,
                    shown_symbols_sec=0.2,
                    workflow_signature_key_sec=0.1,
                    canonicalization_sec=0.15,
                ),
            ),
            solving_sec=3.0,
        )

        assert payload is not None
        self.assertAlmostEqual(payload["model_callback_sec"], 0.8)
        self.assertAlmostEqual(payload["shown_symbols_sec"], 0.3)
        self.assertAlmostEqual(payload["workflow_signature_key_sec"], 0.15)
        self.assertAlmostEqual(payload["canonicalization_sec"], 0.25)
        self.assertAlmostEqual(payload["other_callback_sec"], 0.1)
        self.assertAlmostEqual(payload["model_callback_share_of_solving_sec"], 0.8 / 3.0)

    def test_select_fact_bundle_uses_compressed_candidate_for_optimized_multi_shot(self) -> None:
        config = SimpleNamespace(
            tool_annotations_path="/tmp/tools.json",
            ontology_prefix="ont:",
        )
        ontology = object()
        messages: list[str] = []
        direct_bundle = _fact_bundle(
            internal_schema="legacy_direct",
            internal_solver_mode="multi-shot",
        )
        compressed_bundle = _fact_bundle(
            internal_schema="compressed_candidate_optimized",
            internal_solver_mode="multi-shot-compressed-candidate",
        )

        with (
            patch("clindaws.execution.runner._legacy_direct_bundle", return_value=direct_bundle) as legacy_bundle,
            patch(
                "clindaws.execution.runner.load_candidate_tool_annotations",
                return_value=("candidate_tool",),
            ) as load_candidates,
            patch(
                "clindaws.execution.runner._compressed_candidate_internal_bundle",
                return_value=compressed_bundle,
            ) as compressed_candidate_bundle,
        ):
            fact_bundle, resolved_translation_builder = _select_fact_bundle(
                mode_config=_mode_config("multi-shot"),
                mode="multi-shot",
                config=config,
                ontology=ontology,
                tools=("direct_tool",),
                optimized=True,
                effective_translation_strategy="ape_clingo_legacy",
                progress_callback=messages.append,
                max_workers=7,
            )

        legacy_bundle.assert_called_once_with(config, ontology, ("direct_tool",))
        load_candidates.assert_called_once_with("/tmp/tools.json", "ont:")
        compressed_candidate_bundle.assert_called_once_with(
            config,
            ontology,
            ("candidate_tool",),
            max_workers=7,
        )
        self.assertIs(fact_bundle, compressed_bundle)
        self.assertEqual(
            resolved_translation_builder,
            COMPRESSED_CANDIDATE_TRANSLATION_BUILDER,
        )
        self.assertEqual(
            messages,
            [
                "Step 1b: compressed-candidate optimization started.",
                "Step 1b complete: selected compressed-candidate optimized schema with 1 facts.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
