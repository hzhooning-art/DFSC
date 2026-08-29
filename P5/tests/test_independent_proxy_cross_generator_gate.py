import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_independent_proxy_cross_generator_gate import (  # noqa: E402
    EXTERNAL_MECHANISM_STRENGTHS,
    make_external_case,
    run_external_transfer_map,
)


class IndependentProxyCrossGeneratorGateTests(unittest.TestCase):
    def test_external_generators_exclude_calibration_family(self):
        self.assertNotIn("nonlinear_feedback", EXTERNAL_MECHANISM_STRENGTHS)
        self.assertEqual(
            set(EXTERNAL_MECHANISM_STRENGTHS),
            {"rate_drift", "stretched_exponential"},
        )

    def test_proxy_and_project_traces_use_independent_seeds(self):
        proxy = make_external_case(256, "rate_drift", 1.5, 0.0, 0, role="proxy")
        project = make_external_case(256, "rate_drift", 1.5, 0.30, 0, role="project")
        self.assertNotEqual(proxy["seed"], project["seed"])
        self.assertEqual(proxy["d"], 0.0)
        self.assertEqual(project["d"], 0.30)

    def test_small_matrix_reuses_only_independent_proxy_across_memory_orders(self):
        result = run_external_transfer_map(
            prefix_lengths=(256,),
            memory_orders=(0.0, 0.30),
            mechanism_strengths={
                "rate_drift": (1.5,),
                "stretched_exponential": (0.70,),
            },
            repeats=1,
            calibration_draws=4,
        )
        self.assertEqual(len(result["records"]), 4)
        self.assertTrue(result["protocol"]["independent_proxy_trace"])
        self.assertTrue(result["protocol"]["cross_generator_transfer"])
        self.assertFalse(result["protocol"]["project_observation_used_for_proxy"])
        self.assertTrue(
            result["protocol"]["deterministic_trend_statistic_is_diagnostic_only"]
        )
        self.assertTrue(
            all("deterministic_trend_statistic" in item for item in result["records"])
        )
        for mechanism in ("rate_drift", "stretched_exponential"):
            records = [
                item for item in result["records"] if item["mechanism"] == mechanism
            ]
            self.assertEqual(len({item["proxy_seed"] for item in records}), 1)
            self.assertEqual(len({item["curvature_proxy"] for item in records}), 1)
            self.assertEqual(len({item["threshold"] for item in records}), 1)


if __name__ == "__main__":
    unittest.main()
