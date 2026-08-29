import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p5_memory_protocol import CurveRecord, ar1_profile_bic, fit, residual_ar1_diagnostics  # noqa: E402


class CorrelationAndConstraintTests(unittest.TestCase):
    def test_ar1_diagnostic_detects_positive_dependence(self):
        rng = np.random.default_rng(12)
        rows = []
        for _ in range(20):
            noise = np.zeros(80)
            for index in range(1, len(noise)):
                noise[index] = 0.75 * noise[index - 1] + rng.normal(scale=0.2)
            rows.append(noise)
        diagnostic = residual_ar1_diagnostics(np.asarray(rows))
        self.assertGreater(diagnostic["rho_ar1"], 0.60)
        self.assertLess(diagnostic["effective_sample_size"], diagnostic["n_observations"])
        self.assertTrue(np.isfinite(ar1_profile_bic(np.asarray(rows), 4)["ar1_bic"]))

    def test_nonnegative_fit_is_available_and_audited(self):
        time = np.linspace(0.0, 8.0, 32)
        curves = [
            CurveRecord(str(i), f"g{i % 2}", "y", time, 0.1 + (0.5 + i / 10) * np.exp(-0.4 * time))
            for i in range(4)
        ]
        result = fit(curves, 1, starts=2, nonnegative_amplitudes=True)
        self.assertEqual(result["amplitude_constraint"], "nonnegative")
        self.assertIn("ar1_bic", result)


if __name__ == "__main__":
    unittest.main()
