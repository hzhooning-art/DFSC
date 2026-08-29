import sys
import unittest
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_conditional_contract_transfer import (  # noqa: E402
    build_transfer_observation,
    score_matrix,
)


class ConditionalContractTransferTests(unittest.TestCase):
    def test_rotated_scores_match_reference_rms(self):
        reference = score_matrix("antisymmetric")
        rotated = score_matrix("rotated")
        self.assertAlmostEqual(float(np.sqrt(np.mean(reference**2))), float(np.sqrt(np.mean(rotated**2))), places=12)

    def test_rotated_is_not_antisymmetric_copy(self):
        self.assertFalse(np.allclose(score_matrix("antisymmetric"), score_matrix("rotated")))

    def test_builder_shapes(self):
        times, observations, train, val, labels = build_transfer_observation(48, 8e-4, "rotated", 0.05, 7)
        self.assertEqual(tuple(observations.shape), (65, 48))
        self.assertEqual(len(labels), 48)
        self.assertEqual(len(set(labels.tolist())), 4)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)


if __name__ == "__main__":
    unittest.main()
