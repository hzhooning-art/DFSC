import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p5_memory_protocol import CurveRecord, identifiability_certificate  # noqa: E402


class IdentifiabilityCertificateTests(unittest.TestCase):
    @staticmethod
    def curves(rates: tuple[float, float]) -> list[CurveRecord]:
        time = np.linspace(0.0, 12.0, 50)
        output = []
        for index, amplitudes in enumerate(((1.0, 0.7), (0.6, 1.2), (1.3, 0.4), (0.9, 0.9))):
            value = 0.2 + sum(a * np.exp(-rate * time) for a, rate in zip(amplitudes, rates))
            output.append(CurveRecord(str(index), f"g{index % 2}", "signal", time, value))
        return output

    def test_coalescing_rates_reduce_local_boundary_index(self):
        separated_rates = (0.15, 0.80)
        close_rates = (0.45, 0.46)
        separated = identifiability_certificate(
            self.curves(separated_rates), separated_rates, noise_std=1e-3
        )
        close = identifiability_certificate(self.curves(close_rates), close_rates, noise_std=1e-3)
        self.assertGreater(separated["local_boundary_index"], close["local_boundary_index"])
        self.assertTrue(np.isfinite(close["minimum_projected_information_eigenvalue"]))

    def test_rejects_nonpositive_rates(self):
        with self.assertRaises(ValueError):
            identifiability_certificate(self.curves((0.1, 0.2)), (0.0, 0.2))


if __name__ == "__main__":
    unittest.main()
