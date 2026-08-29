import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_correlated_noise_gate import (  # noqa: E402
    correlation_aware_refusal,
    estimate_ar1_noise,
    make_dataset,
    stationary_ar1,
)


class CorrelatedNoiseGateTests(unittest.TestCase):
    def test_stationary_ar1_has_requested_marginal_scale_and_correlation(self):
        rng = np.random.default_rng(1234)
        sample = stationary_ar1(rng, 20_000, 2, marginal_scale=0.8, rho=0.6)
        self.assertLess(abs(np.std(sample) - 0.8), 0.04)
        empirical_rho = np.corrcoef(sample[:-1, 0], sample[1:, 0])[0, 1]
        self.assertLess(abs(empirical_rho - 0.6), 0.04)

    def test_multilag_estimator_recovers_pure_ar1_noise(self):
        for rho in (0.0, 0.3, 0.6, 0.85):
            rng = np.random.default_rng(7000 + int(100 * rho))
            sample = stationary_ar1(
                rng, 4096, 3, marginal_scale=8.0e-4, rho=rho
            )
            estimate = estimate_ar1_noise(sample)
            self.assertLess(abs(estimate["rho"] - rho), 0.08)
            self.assertLess(abs(estimate["marginal_scale"] / 8.0e-4 - 1.0), 0.15)

    def test_unidentifiable_correlation_forces_refusal(self):
        self.assertTrue(
            correlation_aware_refusal(
                train_rmse=1.0e-4,
                estimated_scale=1.0,
                condition=1.0,
                identifiable=False,
            )
        )

    def test_short_prefix_is_not_declared_identifiable(self):
        rng = np.random.default_rng(99)
        sample = stationary_ar1(rng, 24, 3, marginal_scale=8.0e-4, rho=0.6)
        estimate = estimate_ar1_noise(sample)
        self.assertFalse(estimate["identifiable"])
        self.assertIn("rho_half_width", estimate)

    def test_estimator_separates_smooth_trend_from_correlated_noise(self):
        rng = np.random.default_rng(2026)
        times = np.linspace(0.0, 9.5, 78)
        rates = np.asarray([0.12, 0.21, 0.34])
        smooth = np.exp(-times[:, None] * rates[None, :])
        noise = stationary_ar1(
            rng, 78, 3, marginal_scale=8.0e-4, rho=0.6
        )
        estimate = estimate_ar1_noise(smooth + noise)
        self.assertLess(abs(estimate["rho"] - 0.6), 0.25)
        self.assertLess(
            abs(estimate["marginal_scale"] / 8.0e-4 - 1.0), 0.25
        )

    def test_estimator_handles_project_nonlinear_calibration_prefix(self):
        estimate = make_dataset(0.6, 0.085, 0)["aware_estimate"]
        self.assertLess(abs(estimate["rho"] - 0.6), 0.25)
        self.assertLess(
            abs(estimate["marginal_scale"] / 8.0e-4 - 1.0), 0.35
        )


if __name__ == "__main__":
    unittest.main()
