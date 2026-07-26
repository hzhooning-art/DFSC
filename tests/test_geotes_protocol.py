from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp48", ROOT / "experiments" / "exp48_real_geotes_cross_cycle.py")
EXP48 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXP48)


class GeoTESProtocolTests(unittest.TestCase):
    def test_cycle_split_has_finite_three_channel_responses(self) -> None:
        first, second = EXP48.load_cycles()
        self.assertEqual(first["response"].shape[1], 3)
        self.assertEqual(second["response"].shape[1], 3)
        self.assertTrue(np.isfinite(first["response"]).all())
        self.assertTrue(np.isfinite(second["response"]).all())
        self.assertAlmostEqual(float(first["time"][0]), 0.0)
        self.assertAlmostEqual(float(second["time"][-1]), 1.0)

    def test_forcing_tensor_is_batched_and_boundary_localized(self) -> None:
        times = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
        driver = torch.sin(times).abs()
        quadrature = (torch.arange(6, dtype=torch.float64) + 0.5) / 6.0
        forcing = EXP48.forcing_tensor(times, driver, quadrature)
        self.assertEqual(tuple(forcing.shape), (8, 6, 3))
        self.assertTrue(torch.all(forcing[:, :, 1:] == 0).item())


if __name__ == "__main__":
    unittest.main()
