import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_long_memory_mismatch_gate import (  # noqa: E402
    assess_prescreen,
    ar1_whiteness_diagnostic,
    fractional_noise,
    long_memory_aware_refusal,
    make_dataset,
)
from probe_correlated_noise_gate import estimate_ar1_noise  # noqa: E402


class LongMemoryMismatchGateTests(unittest.TestCase):
    def test_fractional_noise_has_requested_scale_and_long_lag_dependence(self):
        iid = fractional_noise(
            np.random.default_rng(701), 20_000, 2, marginal_scale=0.8, d=0.0
        )
        long_memory = fractional_noise(
            np.random.default_rng(702), 20_000, 2, marginal_scale=0.8, d=0.35
        )
        self.assertLess(abs(np.std(iid) - 0.8), 0.04)
        self.assertLess(abs(np.std(long_memory) - 0.8), 0.04)
        iid_lag20 = np.corrcoef(iid[:-20, 0], iid[20:, 0])[0, 1]
        memory_lag20 = np.corrcoef(
            long_memory[:-20, 0], long_memory[20:, 0]
        )[0, 1]
        self.assertGreater(memory_lag20 - iid_lag20, 0.08)

    def test_whiteness_diagnostic_accepts_iid_control(self):
        sample = fractional_noise(
            np.random.default_rng(801), 512, 3, marginal_scale=8.0e-4, d=0.0
        )
        estimate = estimate_ar1_noise(sample)
        diagnostic = ar1_whiteness_diagnostic(
            sample, rho=estimate["rho"], degree=8, max_lag=8
        )
        self.assertTrue(diagnostic["adequate"])
        self.assertGreaterEqual(diagnostic["p_value"], 0.01)

    def test_whiteness_diagnostic_rejects_strong_long_memory(self):
        sample = fractional_noise(
            np.random.default_rng(802), 512, 3, marginal_scale=8.0e-4, d=0.4
        )
        estimate = estimate_ar1_noise(sample)
        diagnostic = ar1_whiteness_diagnostic(
            sample, rho=estimate["rho"], degree=8, max_lag=8
        )
        self.assertFalse(diagnostic["adequate"])
        self.assertLess(diagnostic["p_value"], 0.01)

    def test_inadequate_ar1_model_forces_refusal(self):
        self.assertTrue(
            long_memory_aware_refusal(
                train_rmse=1.0e-4,
                estimated_scale=1.0,
                condition=1.0,
                identifiable=True,
                ar1_adequate=False,
            )
        )

    def test_project_dataset_is_deterministic_and_records_long_memory_order(self):
        first = make_dataset(0.3, 0.085, 0)
        second = make_dataset(0.3, 0.085, 0)
        self.assertEqual(first["d"], 0.3)
        self.assertEqual(first["seed"], second["seed"])
        np.testing.assert_array_equal(
            first["observations"].cpu().numpy(),
            second["observations"].cpu().numpy(),
        )
        self.assertIn("ar1_whiteness", first)

    def test_prescreen_fails_when_controls_and_strong_memory_are_not_separated(self):
        records = []
        for index in range(6):
            records.append(
                {"d": 0.0, "ar1_model_adequate": index < 2,
                 "long_memory_mismatch_detected": index >= 2}
            )
        for index in range(12):
            records.append(
                {"d": 0.3, "ar1_model_adequate": index >= 7,
                 "long_memory_mismatch_detected": index < 7}
            )
        assessment = assess_prescreen(records, expected_count=18)
        self.assertFalse(assessment["route_pass"])
        self.assertAlmostEqual(assessment["control_adequacy_rate"], 2 / 6)
        self.assertAlmostEqual(
            assessment["strong_long_memory_detection_rate"], 7 / 12
        )


if __name__ == "__main__":
    unittest.main()
