import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_curvature_normalized_increment_gate import (  # noqa: E402
    CALIBRATION_STRENGTHS,
    build_curvature_calibration,
    curvature_proxy,
    run_curvature_normalized_map,
)
from probe_spectral_long_memory_gate import STRENGTHS  # noqa: E402


class CurvatureNormalizedIncrementGateTests(unittest.TestCase):
    def test_calibration_bank_excludes_project_strengths(self):
        self.assertTrue(set(CALIBRATION_STRENGTHS).isdisjoint(STRENGTHS))

    def test_curvature_proxy_is_finite_and_linear_trend_invariant(self):
        rng = np.random.default_rng(51)
        values = rng.normal(scale=0.01, size=(256, 3))
        trend = np.linspace(-1.0, 2.0, 256)[:, None]
        baseline = curvature_proxy(values)
        shifted = curvature_proxy(values + trend)
        self.assertTrue(np.isfinite(baseline))
        self.assertAlmostEqual(baseline, shifted, places=7)

    def test_calibration_uses_only_observable_proxy(self):
        calibration = build_curvature_calibration(256, draws=4)
        self.assertEqual(len(calibration["bank"]), len(CALIBRATION_STRENGTHS))
        self.assertNotIn("strength", calibration["model"])
        self.assertEqual(calibration["model"]["input"], "curvature_proxy")

    def test_small_curvature_matrix_is_complete(self):
        result = run_curvature_normalized_map(
            prefix_lengths=(256,),
            memory_orders=(0.0, 0.30),
            strengths=(0.05, 0.085, 0.2),
            repeats=1,
            calibration_draws=4,
        )
        self.assertEqual(len(result["records"]), 6)
        self.assertEqual(result["assessment_by_length"]["256"]["expected_count"], 6)
        self.assertTrue(result["protocol"]["observable_curvature_normalization"])
        self.assertFalse(result["protocol"]["project_strength_used_by_gate"])


if __name__ == "__main__":
    unittest.main()
