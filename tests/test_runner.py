import unittest

from clindaws.core.models import HorizonRecord
from clindaws.execution.runner import _solve_callback_profile_payload


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


if __name__ == "__main__":
    unittest.main()
