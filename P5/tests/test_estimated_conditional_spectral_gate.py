import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_estimated_conditional_spectral_gate import (  # noqa: E402
    estimate_smooth_mean,
    estimated_conditional_threshold,
    make_independent_control,
    run_estimated_conditional_map,
)


class EstimatedConditionalSpectralGateTests(unittest.TestCase):
    def test_smooth_mean_recovers_control_trajectory(self):
        case = make_independent_control(256, strength=0.085)
        estimate = estimate_smooth_mean(case["observations"])
        rmse = float(np.sqrt(np.mean((estimate - case["clean"]) ** 2)))
        self.assertEqual(estimate.shape, case["clean"].shape)
        self.assertLess(rmse, 8.0e-4)

    def test_refitted_conditional_threshold_is_deterministic(self):
        case = make_independent_control(256, strength=0.085)
        first = estimated_conditional_threshold(
            case["observations"], draws=16, seed=2201, quantile=0.99
        )
        second = estimated_conditional_threshold(
            case["observations"], draws=16, seed=2201, quantile=0.99
        )
        self.assertEqual(first, second)

    def test_small_estimated_matrix_is_complete(self):
        result = run_estimated_conditional_map(
            prefix_lengths=(256,),
            memory_orders=(0.0, 0.30),
            strengths=(0.085,),
            repeats=1,
            calibration_draws=8,
        )
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(result["assessment_by_length"]["256"]["expected_count"], 2)
        self.assertTrue(result["protocol"]["independent_control_trajectory"])
        self.assertTrue(result["protocol"]["mean_refit_in_bootstrap"])


if __name__ == "__main__":
    unittest.main()
