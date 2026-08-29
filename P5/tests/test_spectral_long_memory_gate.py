import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_long_memory_mismatch_gate import fractional_noise  # noqa: E402
from probe_spectral_long_memory_gate import (  # noqa: E402
    assess_length,
    calibrated_null_threshold,
    local_whittle_d,
    make_prefix_case,
    spectral_memory_statistic,
)


class SpectralLongMemoryGateTests(unittest.TestCase):
    def test_local_whittle_separates_iid_and_long_memory(self):
        iid = fractional_noise(
            np.random.default_rng(1201), 4096, 3, marginal_scale=1.0, d=0.0
        )
        memory = fractional_noise(
            np.random.default_rng(1202), 4096, 3, marginal_scale=1.0, d=0.35
        )
        self.assertLess(local_whittle_d(iid, bandwidth=128), 0.12)
        self.assertGreater(local_whittle_d(memory, bandwidth=128), 0.22)

    def test_monte_carlo_threshold_is_deterministic(self):
        first = calibrated_null_threshold(
            length=256, channels=3, draws=32, seed=1301, quantile=0.99
        )
        second = calibrated_null_threshold(
            length=256, channels=3, draws=32, seed=1301, quantile=0.99
        )
        self.assertEqual(first, second)

    def test_calibrated_gate_accepts_control_and_detects_strong_memory(self):
        threshold = calibrated_null_threshold(
            length=512, channels=3, draws=128, seed=1401, quantile=0.99
        )
        control = fractional_noise(
            np.random.default_rng(1402), 512, 3, marginal_scale=8.0e-4, d=0.0
        )
        memory = fractional_noise(
            np.random.default_rng(1403), 512, 3, marginal_scale=8.0e-4, d=0.4
        )
        self.assertLessEqual(spectral_memory_statistic(control), threshold)
        self.assertGreater(spectral_memory_statistic(memory), threshold)

    def test_project_prefix_case_is_deterministic(self):
        first = make_prefix_case(256, d=0.3, strength=0.085, repeat=0)
        second = make_prefix_case(256, d=0.3, strength=0.085, repeat=0)
        self.assertEqual(first["seed"], second["seed"])
        np.testing.assert_array_equal(first["observations"], second["observations"])

    def test_length_assessment_requires_both_control_and_detection_rates(self):
        records = []
        for index in range(6):
            records.append({"d": 0.0, "mismatch_detected": index >= 5})
        for index in range(12):
            records.append({"d": 0.3, "mismatch_detected": index < 9})
        assessment = assess_length(records, expected_count=18)
        self.assertTrue(assessment["route_pass"])
        records[0]["mismatch_detected"] = True
        self.assertFalse(assess_length(records, expected_count=18)["route_pass"])


if __name__ == "__main__":
    unittest.main()
