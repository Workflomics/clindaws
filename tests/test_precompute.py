import unittest

from clindaws.execution.precompute import (
    _compress_plain_output_port_candidates,
    _emit_direct_output_profile_check_facts,
)
from clindaws.translators.fact_writer import _FactWriter


class PrecomputeTests(unittest.TestCase):
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
