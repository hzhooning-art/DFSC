import sys
import unittest
from pathlib import Path

import torch


P4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P4_ROOT / "experiments"))
from p4_periodic_heat_2d_audit import analytic_field, coefficients, spectral_heat_primitive


class PeriodicHeat2DAuditTest(unittest.TestCase):
    def test_value_and_parameter_gradient(self):
        torch.set_default_dtype(torch.float64)
        device = torch.device("cpu")
        coeff = coefficients(74101, device, torch.float64)
        kappa = torch.tensor(0.08, dtype=torch.float64, requires_grad=True)
        initial, _ = analytic_field(16, coeff, kappa.detach(), 0.0)
        prediction = spectral_heat_primitive(initial, kappa, 0.5)
        reference, derivative = analytic_field(16, coeff, kappa.detach(), 0.5)
        self.assertLess(float((prediction - reference).abs().max().detach()), 1.0e-12)
        gradient = torch.autograd.grad(prediction.square().mean(), kappa)[0]
        reference_gradient = (2.0 * reference * derivative).mean()
        self.assertLess(float((gradient - reference_gradient).abs().detach()), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
