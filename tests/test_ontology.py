import unittest
from pathlib import Path

from snakeAPE.config import load_config
from snakeAPE.ontology import Ontology


ROOT = Path(__file__).resolve().parents[2] / "ape_asp_benchmarks" / "test_cases"


class OntologyParsingTests(unittest.TestCase):
    def test_geogmt_intersection_ancestors_are_preserved(self) -> None:
        config = load_config(
            ROOT / "GeoGMT" / "E0" / "config_ASP_GeoGMT_E0_WC.json"
        )
        ontology = Ontology.from_file(config.ontology_path, config.ontology_prefix)

        self.assertTrue(
            {
                "pscoast_S",
                "Draw_water_mass",
                "Draw_water",
                "Basemaps",
                "Plot_creation",
                "ToolsTaxonomy",
            }.issubset(ontology.ancestors_of("pscoast_S"))
        )
        self.assertTrue(
            {"pscoast_W", "Draw_water_borders", "Draw_water"}.issubset(
                ontology.ancestors_of("pscoast_W")
            )
        )
        self.assertTrue(
            {"pscoast_I", "Draw_rivers", "Draw_water"}.issubset(
                ontology.ancestors_of("pscoast_I")
            )
        )

    def test_defect_concentration_ontology_remains_stable(self) -> None:
        config = load_config(
            ROOT
            / "defect_concentration"
            / "config_ASP_defect_2000_8_WC.json"
        )
        ontology = Ontology.from_file(config.ontology_path, config.ontology_prefix)

        self.assertIn("Bulk", ontology.ancestors_of("Bulk"))
        self.assertIn("Structure", ontology.ancestors_of("Bulk"))
        self.assertIn("Tool", ontology.ancestors_of("CreateProject"))


if __name__ == "__main__":
    unittest.main()
