import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))

from p4_matrix_ood_conditioning import observation_diagnostics  # noqa: E402


class MatrixOODDiagnosticsTests(unittest.TestCase):
    def test_diagonal_generator_has_finite_observation_sensitivity_and_zero_nonnormality(self):
        params = torch.tensor([[-0.3, 0.0, 0.0, -0.7]], dtype=torch.float64)
        y0 = torch.tensor([[1.0, 0.8]], dtype=torch.float64)
        times = torch.tensor([0.05, 0.10, 0.15], dtype=torch.float64)

        sigma_min, jacobian_condition, nonnormality = observation_diagnostics(params, y0, times)

        self.assertTrue(torch.isfinite(sigma_min).all())
        self.assertTrue(torch.isfinite(jacobian_condition).all())
        self.assertGreater(float(sigma_min.item()), 0.0)
        self.assertAlmostEqual(float(nonnormality.item()), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
