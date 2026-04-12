import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clindaws.core.models import FactBundle, HorizonRecord
from clindaws.execution.runner import (
    COMPRESSED_CANDIDATE_TRANSLATION_BUILDER,
    RunContext,
    _effective_parallel_mode,
    _load_tools_for_mode,
    _mode_config,
    _select_fact_bundle,
    _solve_callback_profile_payload,
    run_ground_only,
    run_once,
)
from clindaws.execution.solver import program_paths_for_mode


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
    def test_load_tools_for_plain_multi_shot_preserves_duplicate_outputs(self) -> None:
        config = SimpleNamespace(
            tool_annotations_path="/tmp/tools.json",
            ontology_prefix="ont:",
        )

        with (
            patch(
                "clindaws.execution.runner.load_multi_shot_tool_annotations",
                return_value=("multi_shot_tool",),
            ) as load_multi_shot,
            patch("clindaws.execution.runner.load_direct_tool_annotations") as load_direct,
            patch("clindaws.execution.runner.load_candidate_tool_annotations") as load_candidate,
        ):
            tools = _load_tools_for_mode(config, "ape_multi_shot")

        load_multi_shot.assert_called_once_with("/tmp/tools.json", "ont:")
        load_direct.assert_not_called()
        load_candidate.assert_not_called()
        self.assertEqual(tools, ("multi_shot_tool",))

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
            patch(
                "clindaws.execution.runner._ape_multi_shot_direct_bundle",
                return_value=direct_bundle,
            ) as legacy_bundle,
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

        legacy_bundle.assert_called_once_with(
            config,
            ontology,
            ("direct_tool",),
            internal_solver_mode="multi-shot",
        )
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

    def test_program_paths_map_optimized_multi_shot_to_compressed_candidate(self) -> None:
        paths = program_paths_for_mode("multi-shot", optimized=True)

        self.assertTrue(paths)
        self.assertTrue(all("multi_shot_compressed_candidate" in str(path) for path in paths))

    def test_program_paths_map_plain_multi_shot_to_direct_backend(self) -> None:
        paths = program_paths_for_mode("multi-shot", optimized=False)

        self.assertTrue(paths)
        self.assertTrue(all("multi_shot_compressed_candidate" not in str(path) for path in paths))

    def test_program_paths_map_single_shot_to_plain_multi_shot_backend(self) -> None:
        paths = program_paths_for_mode("single-shot", optimized=False)

        self.assertTrue(paths)
        self.assertTrue(all("clindaws/encodings/multi_shot" in str(path) for path in paths))
        self.assertTrue(all("multi_shot_compressed_candidate" not in str(path) for path in paths))

    def test_single_shot_uses_ape_multi_shot_translation_pathway(self) -> None:
        self.assertEqual(_mode_config("single-shot").translation_pathway, "ape_multi_shot")

    def test_select_fact_bundle_uses_multi_shot_direct_builder_for_single_shot(self) -> None:
        config = SimpleNamespace(
            tool_annotations_path="/tmp/tools.json",
            ontology_prefix="ont:",
        )
        ontology = object()
        single_shot_bundle = _fact_bundle(
            internal_schema="legacy_direct",
            internal_solver_mode="single-shot",
        )

        with (
            patch(
                "clindaws.execution.runner.build_fact_bundle_ape_multi_shot",
                return_value=single_shot_bundle,
            ) as build_bundle,
            patch(
                "clindaws.execution.runner.apply_precompute",
                return_value=single_shot_bundle,
            ) as apply_precompute,
        ):
            fact_bundle, resolved_translation_builder = _select_fact_bundle(
                mode_config=_mode_config("single-shot"),
                mode="single-shot",
                config=config,
                ontology=ontology,
                tools=("multi_shot_tool",),
                optimized=False,
                effective_translation_strategy="ape_clingo_legacy",
                progress_callback=None,
            )

        build_bundle.assert_called_once_with(config, ontology, ("multi_shot_tool",))
        apply_precompute.assert_called_once()
        self.assertEqual(fact_bundle.internal_solver_mode, "single-shot")
        self.assertEqual(resolved_translation_builder, "runtime_legacy")

    def test_run_ground_only_supports_single_shot(self) -> None:
        config = SimpleNamespace(
            solutions_dir_path=Path("/tmp/single-shot-ground-only"),
            solution_length_max=4,
            debug_mode=False,
            config_path=Path("/tmp/dummy-config.json"),
            ontology_path=Path("/tmp/dummy-ontology.owl"),
            tool_annotations_path=Path("/tmp/dummy-tools.json"),
        )
        ctx = RunContext(
            config=config,
            solution_dir=Path("/tmp/single-shot-ground-only"),
            fact_bundle=_fact_bundle(
                internal_schema="legacy_direct",
                internal_solver_mode="single-shot",
            ),
            translation_sec=0.1,
            translation_peak_rss_mb=12.0,
            effective_translation_strategy="hybrid",
            resolved_translation_builder="runtime_legacy",
            run_metadata={
                "config_path": str(config.config_path),
                "ontology_used": str(config.ontology_path),
                "ontology_entry_count": 0,
                "tool_annotation_used": str(config.tool_annotations_path),
                "tool_count": 0,
                "constraints_used": None,
                "constraint_count": 0,
            },
        )
        grounding_output = SimpleNamespace(
            grounded_horizons=(4,),
            horizon_records=(),
            base_grounding_peak_rss_mb=0.0,
            base_grounding_sec=0.2,
            grounding_sec=0.3,
        )

        with (
            patch("clindaws.execution.runner._prepare_run_context", return_value=ctx),
            patch(
                "clindaws.execution.runner._write_translation_summary",
                return_value=(Path("/tmp/translation_summary.json"), {}),
            ),
            patch(
                "clindaws.execution.runner._GROUNDER_DISPATCH",
                {"single-shot": lambda *args, **kwargs: grounding_output},
            ),
        ):
            result = run_ground_only(
                "dummy-config.json",
                mode="single-shot",
                grounding_strategy="hybrid",
                stage="full",
            )

        self.assertEqual(result.mode, "single-shot")
        self.assertEqual(result.grounded_horizons, (4,))

    def test_run_once_timeout_returns_without_writing_run_artifacts(self) -> None:
        config = SimpleNamespace(
            timeout_sec=10.0,
            solutions_dir_path=Path("/tmp/unused-solutions-dir"),
            number_of_generated_graphs=0,
            debug_mode=False,
        )
        ctx = RunContext(
            config=config,
            solution_dir=Path("/tmp/unused-output-dir"),
            fact_bundle=_fact_bundle(
                internal_schema="compressed_candidate_optimized",
                internal_solver_mode="multi-shot-compressed-candidate",
            ),
            translation_sec=0.1,
            translation_peak_rss_mb=12.0,
            effective_translation_strategy="hybrid",
            resolved_translation_builder=COMPRESSED_CANDIDATE_TRANSLATION_BUILDER,
            run_metadata={},
        )

        with (
            patch("clindaws.execution.runner._prepare_run_context", return_value=ctx),
            patch(
                "clindaws.execution.runner._run_solve_in_worker",
                return_value=(SimpleNamespace(
                    raw_solutions=(),
                    solutions=(),
                    base_grounding_peak_rss_mb=0.0,
                    base_grounding_sec=0.0,
                    grounding_sec=0.0,
                    solving_sec=0.0,
                    horizon_records=(),
                ), True),
            ),
        ):
            result = run_once(
                "dummy-config.json",
                mode="multi-shot",
                grounding_strategy="hybrid",
                optimized=True,
                render_graphs=False,
            )

        self.assertTrue(result.timed_out)
        self.assertEqual(result.completed_stage, "run_timeout")
        self.assertIsNone(result.run_log_path)
        self.assertIsNone(result.run_summary_path)
        self.assertIsNone(result.answer_set_path)
        self.assertIsNone(result.workflow_signature_path)

    def test_effective_parallel_mode_defaults_to_six_compete_for_large_optimized_multi_shot(self) -> None:
        fact_bundle = _fact_bundle(internal_schema="compressed_candidate_optimized")
        fact_bundle = FactBundle(
            **{
                **fact_bundle.__dict__,
                "tool_labels": {f"tool_{index}": f"Tool {index}" for index in range(200)},
            }
        )

        with patch("clindaws.execution.runner.os.cpu_count", return_value=12):
            effective_parallel_mode = _effective_parallel_mode("multi-shot", None, fact_bundle)

        self.assertEqual(effective_parallel_mode, "6,compete")


if __name__ == "__main__":
    unittest.main()
