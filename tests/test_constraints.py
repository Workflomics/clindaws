import re
import unittest
from pathlib import Path

from snakeAPE.config import load_config
from snakeAPE.ontology import Ontology
from snakeAPE.tool_annotations import load_direct_tool_annotations
from snakeAPE.translator import build_fact_bundle_ape_multi_shot


ROOT = Path(__file__).resolve().parents[2] / "ape_asp_benchmarks" / "test_cases"


class ConstraintTranslationTests(unittest.TestCase):
    def _translated_facts(self, config_name: str) -> str:
        config = load_config(ROOT / "QuAnGIS" / "No1" / config_name)
        ontology = Ontology.from_file(config.ontology_path, config.ontology_prefix)
        tools = load_direct_tool_annotations(config.tool_annotations_path, config.ontology_prefix)
        return build_fact_bundle_ape_multi_shot(config, ontology, tools).facts

    def test_quangis_no1_sat_use_t_keeps_all_selector_dimensions(self) -> None:
        facts = self._translated_facts("config_SAT_QuAnGIS_no1_WC.json")

        selector_match = re.search(r'constraint_use_data\("([^"]+)"\)\.', facts)
        self.assertIsNotNone(selector_match)
        selector_id = selector_match.group(1)

        self.assertIn(
            f'constraint_data_selector_category("{selector_id}", "CoreConceptQ").',
            facts,
        )
        self.assertIn(
            f'constraint_data_selector_category("{selector_id}", "LayerA").',
            facts,
        )
        self.assertIn(
            f'constraint_data_selector_category("{selector_id}", "NominalA").',
            facts,
        )
        self.assertIn(
            f'constraint_data_selector_dim("{selector_id}", "CoreConceptQ", "MatrixQ").',
            facts,
        )
        self.assertIn(
            f'constraint_data_selector_dim("{selector_id}", "LayerA", "RegionA").',
            facts,
        )
        self.assertIn(
            f'constraint_data_selector_dim("{selector_id}", "NominalA", "ERA").',
            facts,
        )

    def test_quangis_no1_asp_constraints_match_sat_selector_shape(self) -> None:
        asp_facts = self._translated_facts("config_ASP_QuAnGIS_no1_WC.json")
        sat_facts = self._translated_facts("config_SAT_QuAnGIS_no1_WC.json")

        def relevant(facts: str) -> set[str]:
            return {
                line
                for line in facts.splitlines()
                if line.startswith("constraint_use_data(")
                or line.startswith("constraint_data_selector_category(")
                or line.startswith("constraint_data_selector_dim(")
            }

        self.assertEqual(relevant(asp_facts), relevant(sat_facts))


if __name__ == "__main__":
    unittest.main()
