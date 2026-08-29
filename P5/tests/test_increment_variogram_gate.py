import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_increment_variogram_gate import (  # noqa: E402
    increment_memory_statistic,
    increment_null_threshold,
    run_increment_feasibility_map,
)
from probe_long_memory_mismatch_gate import fractional_noise  # noqa: E402


class IncrementVariogramGateTests(unittest.TestCase):
    def test_statistic_annihilates_linear_trend(self):
        rng = np.random.default_rng(17)
        noise = rng.normal(size=(256, 3))
        trend = np.linspace(-2.0, 3.0, 256)[:, None]
        baseline = increment_memory_statistic(noise)
        shifted = increment_memory_statistic(noise + trend)
        self.assertAlmostEqual(baseline, shifted, places=12)

    def test_strong_fractional_memory_increases_statistic(self):
        iid = fractional_noise(
            np.random.default_rng(29), 512, 3, marginal_scale=1.0, d=0.0
        )
        memory = fractional_noise(
            np.random.default_rng(29), 512, 3, marginal_scale=1.0, d=0.45
        )
        self.assertGreater(
            increment_memory_statistic(memory),
            increment_memory_statistic(iid),
        )

    def test_conditional_threshold_is_deterministic(self):
        mean = np.column_stack(
            [np.linspace(0.0, 1.0, 256), np.linspace(1.0, 0.0, 256)]
        )
        first = increment_null_threshold(mean, draws=16, seed=902, quantile=0.99)
        second = increment_null_threshold(mean, draws=16, seed=902, quantile=0.99)
        self.assertEqual(first, second)

    def test_small_matrix_is_complete(self):
        result = run_increment_feasibility_map(
            prefix_lengths=(256,),
            memory_orders=(0.0, 0.30),
            strengths=(0.085,),
            repeats=1,
            calibration_draws=8,
        )
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(result["assessment_by_length"]["256"]["expected_count"], 2)
        self.assertTrue(result["protocol"]["linear_trend_annihilation"])
        self.assertTrue(result["protocol"]["strength_conditional_threshold"])
        self.assertTrue(result["protocol"]["decision_frozen_before_project_run"])


if __name__ == "__main__":
    unittest.main()
