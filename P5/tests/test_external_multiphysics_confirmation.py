import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_external_multiphysics_confirmation import (  # noqa: E402
    external_backgrounds,
    standardized_curve_residual,
)


class ExternalMultiphysicsConfirmationTests(unittest.TestCase):
    def test_residual_standardization_is_finite_and_non_degenerate(self):
        x = np.linspace(0.0, 10.0, 500)
        residual = standardized_curve_residual(np.exp(-x) + 0.01 * np.sin(17 * x))
        self.assertTrue(np.isfinite(residual).all())
        self.assertGreater(float(np.std(residual)), 0.1)

    def test_external_background_inventory_has_three_sources_and_seven_backgrounds(self):
        backgrounds, provenance = external_backgrounds()
        self.assertEqual(len(backgrounds), 7)
        self.assertEqual(len({row["source"] for row in provenance}), 3)
        self.assertTrue(all(len(values) >= 65 for values in backgrounds.values()))


if __name__ == "__main__":
    unittest.main()
