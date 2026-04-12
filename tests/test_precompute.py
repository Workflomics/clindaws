import unittest

from clindaws.core.workflow_input_compression import (
    build_workflow_input_compression_plan,
    workflow_input_compression_stats,
)
from clindaws.execution.precompute import (
    _compress_plain_output_port_candidates,
    _emit_multi_shot_workflow_input_facts,
    _emit_direct_output_profile_check_facts,
)
from clindaws.translators.fact_writer import _FactWriter


class PrecomputeTests(unittest.TestCase):
    def test_workflow_input_compression_stats_capture_slot_domain(self) -> None:
        plan = build_workflow_input_compression_plan(
            {
                "wf_input_0": {"cat_a": ("value_a",)},
                "wf_input_1": {"cat_a": ("value_a",)},
                "wf_input_2": {"cat_a": ("value_b",)},
            }
        )

        self.assertEqual(
            workflow_input_compression_stats(plan),
            {
                "workflow_input_equivalence_classes": 2,
                "workflow_input_repeated_equivalence_classes": 1,
                "workflow_input_collapsed_members": 1,
                "workflow_input_slots": 2,
                "workflow_input_planner_visible_count_uncompressed": 3,
                "workflow_input_planner_visible_count_compressed": 3,
                "workflow_input_planner_visible_reduction_if_compressed": 0,
            },
        )

    def test_multi_shot_workflow_input_facts_keep_raw_inputs_and_slots(self) -> None:
        writer = _FactWriter()

        planner_artifact_profiles, stats = _emit_multi_shot_workflow_input_facts(
            writer,
            parsed=type(
                "Parsed",
                (),
                {
                    "workflow_input_ids": ("wf_input_0", "wf_input_1", "wf_input_2"),
                    "workflow_input_dims": {
                        "wf_input_0": {"cat_a": ("value_a",)},
                        "wf_input_1": {"cat_a": ("value_a",)},
                        "wf_input_2": {"cat_a": ("value_b",)},
                    },
                    "workflow_input_units": {},
                },
            )(),
        )

        self.assertIn("wf_input_0", planner_artifact_profiles)
        self.assertIn("wf_input_1", planner_artifact_profiles)
        self.assertIn('slot("wf_input_0", "wf_input_0")', planner_artifact_profiles)
        self.assertIn('slot("wf_input_0", "wf_input_1")', planner_artifact_profiles)
        self.assertNotIn("planner_workflow_input", writer.text())
        self.assertEqual(
            stats,
            {
                "precompute_workflow_input_equivalence_classes": 2,
                "precompute_workflow_input_repeated_equivalence_classes": 1,
                "precompute_workflow_input_classes": 2,
                "precompute_repeated_workflow_input_classes": 1,
                "precompute_workflow_input_collapsed_members": 1,
                "precompute_workflow_input_slots": 2,
                "precompute_workflow_input_planner_visible_count_uncompressed": 3,
                "precompute_workflow_input_planner_visible_count_compressed": 3,
                "precompute_workflow_input_planner_visible_reduction_if_compressed": 0,
            },
        )

    def test_plain_output_port_compression_keeps_one_equivalent_representative(self) -> None:
        compressed, stats = _compress_plain_output_port_candidates(
            output_port_terminal_sets={
                "port_1": {
                    "cat_a": frozenset({"value_a", "value_b"}),
                },
            },
            port_requirements={
                "consumer_1": {
                    "cat_a": (frozenset({"value_a", "value_b"}),),
                },
            },
            goal_requirements={},
        )

        self.assertEqual(
            compressed,
            {
                "port_1": {
                    "cat_a": ("value_a",),
                },
            },
        )
        self.assertEqual(
            stats,
            {
                "precompute_plain_output_port_candidate_categories_retained": 1,
                "precompute_plain_output_port_candidate_categories_requiring_check": 0,
                "precompute_plain_output_port_candidates_dense": 2,
                "precompute_plain_output_port_candidates": 1,
                "precompute_plain_output_port_candidates_dropped": 1,
                "precompute_plain_output_port_candidate_categories_compressed": 1,
                "precompute_plain_output_port_candidates_requiring_check": 0,
            },
        )

    def test_plain_output_port_compression_keeps_distinguishable_representatives(self) -> None:
        compressed, stats = _compress_plain_output_port_candidates(
            output_port_terminal_sets={
                "port_1": {
                    "cat_a": frozenset({"value_a", "value_b"}),
                },
            },
            port_requirements={
                "consumer_1": {
                    "cat_a": (frozenset({"value_a"}),),
                },
            },
            goal_requirements={},
        )

        self.assertEqual(
            compressed,
            {
                "port_1": {
                    "cat_a": ("value_a", "value_b"),
                },
            },
        )
        self.assertEqual(
            stats,
            {
                "precompute_plain_output_port_candidate_categories_retained": 1,
                "precompute_plain_output_port_candidate_categories_requiring_check": 1,
                "precompute_plain_output_port_candidates_dense": 2,
                "precompute_plain_output_port_candidates": 2,
                "precompute_plain_output_port_candidates_dropped": 0,
                "precompute_plain_output_port_candidate_categories_compressed": 0,
                "precompute_plain_output_port_candidates_requiring_check": 2,
            },
        )

    def test_plain_output_port_compression_distinguishes_goal_support_profiles(self) -> None:
        compressed, stats = _compress_plain_output_port_candidates(
            output_port_terminal_sets={
                "port_1": {
                    "cat_a": frozenset({"value_a", "value_b", "value_c"}),
                },
            },
            port_requirements={},
            goal_requirements={
                "goal_1": {
                    "cat_a": (frozenset({"value_a", "value_c"}),),
                },
                "goal_2": {
                    "cat_a": (frozenset({"value_c"}),),
                },
            },
        )

        self.assertEqual(
            compressed,
            {
                "port_1": {
                    "cat_a": ("value_a", "value_b", "value_c"),
                },
            },
        )
        self.assertEqual(
            stats,
            {
                "precompute_plain_output_port_candidate_categories_retained": 1,
                "precompute_plain_output_port_candidate_categories_requiring_check": 1,
                "precompute_plain_output_port_candidates_dense": 3,
                "precompute_plain_output_port_candidates": 3,
                "precompute_plain_output_port_candidates_dropped": 0,
                "precompute_plain_output_port_candidate_categories_compressed": 0,
                "precompute_plain_output_port_candidates_requiring_check": 3,
            },
        )

    def test_identical_support_profiles_do_not_require_check(self) -> None:
        writer = _FactWriter()

        stats = _emit_direct_output_profile_check_facts(
            writer,
            compressed_output_profile_candidates={
                "profile_1": {
                    "cat_a": ("value_a", "value_b"),
                },
            },
            candidate_signature_supports={
                ("profile_1", "value_a", "cat_a"): frozenset({"sig_1"}),
                ("profile_1", "value_b", "cat_a"): frozenset({"sig_1"}),
            },
            candidate_goal_supports={},
        )

        self.assertEqual(
            stats,
            {
                "precompute_output_profile_candidate_categories_retained": 1,
                "precompute_output_profile_candidate_categories_requiring_check": 0,
                "precompute_output_profile_candidates_requiring_check": 0,
            },
        )
        self.assertEqual(writer.text(), "")

    def test_distinct_support_profiles_require_check(self) -> None:
        writer = _FactWriter()

        stats = _emit_direct_output_profile_check_facts(
            writer,
            compressed_output_profile_candidates={
                "profile_1": {
                    "cat_a": ("value_a", "value_b"),
                },
            },
            candidate_signature_supports={
                ("profile_1", "value_a", "cat_a"): frozenset({"sig_1"}),
                ("profile_1", "value_b", "cat_a"): frozenset(),
            },
            candidate_goal_supports={},
        )

        self.assertEqual(
            stats,
            {
                "precompute_output_profile_candidate_categories_retained": 1,
                "precompute_output_profile_candidate_categories_requiring_check": 1,
                "precompute_output_profile_candidates_requiring_check": 2,
            },
        )
        self.assertEqual(
            writer.text().splitlines(),
            [
                'direct_output_profile_category_requires_check("profile_1", "cat_a").',
                'direct_output_profile_candidate_requires_check("profile_1", "value_a", "cat_a").',
                'direct_output_profile_candidate_requires_check("profile_1", "value_b", "cat_a").',
            ],
        )


if __name__ == "__main__":
    unittest.main()
