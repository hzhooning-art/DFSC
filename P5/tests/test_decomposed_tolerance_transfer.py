from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_decomposed_tolerance_transfer import (  # noqa: E402
    MODEL_ALLOWANCE,
    NOISE_ENVELOPE,
    PROXY_SCOPE,
    block_score_matrix,
    build_transfer_observation,
    frozen_total_tolerance,
)


class DecomposedToleranceTransferTests(unittest.TestCase):
    def test_constructions_have_matched_rms_scale(self) -> None:
        first = block_score_matrix("antisymmetric")
        second = block_score_matrix("curved")
        self.assertAlmostEqual(float(np.sqrt(np.mean(first**2))), float(np.sqrt(np.mean(second**2))))
        self.assertFalse(np.allclose(first, second))

    def test_transfer_generator_shapes(self) -> None:
        times, observations, train_idx, val_idx, rates, labels = build_transfer_observation(
            32, 4.0e-4, "curved", 0.05, 7
        )
        self.assertEqual(tuple(observations.shape), (65, 32))
        self.assertEqual(tuple(rates.shape), (32, 2))
        self.assertEqual(labels.shape, (32,))
        self.assertGreater(train_idx.numel(), 0)
        self.assertGreater(val_idx.numel(), 0)
        self.assertEqual(times.numel(), 65)

    def test_frozen_tolerance_uses_stage45_constants(self) -> None:
        proxy = 0.5
        expected = (
            NOISE_ENVELOPE["intercept"]
            + NOISE_ENVELOPE["slope"] * proxy
            + NOISE_ENVELOPE["one_sided_residual"]
            + MODEL_ALLOWANCE
        )
        self.assertAlmostEqual(frozen_total_tolerance(proxy), expected)
        self.assertLess(PROXY_SCOPE[0], proxy)
        self.assertGreater(PROXY_SCOPE[1], proxy)


if __name__ == "__main__":
    unittest.main()
