import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_increment_transfer_gate import (  # noqa: E402
    leave_one_strength_out_threshold,
    run_increment_transfer_map,
)


class IncrementTransferGateTests(unittest.TestCase):
    def test_held_out_strength_is_not_used_for_calibration(self):
        result = leave_one_strength_out_threshold(
            length=256,
            held_out_strength=0.085,
            draws_per_donor=4,
            seed=441,
        )
        self.assertEqual(result["donor_strengths"], [0.05, 0.2])
        self.assertNotIn(0.085, result["donor_strengths"])

    def test_transfer_threshold_is_deterministic(self):
        first = leave_one_strength_out_threshold(256, 0.085, 4, 991)
        second = leave_one_strength_out_threshold(256, 0.085, 4, 991)
        self.assertEqual(first["threshold"], second["threshold"])

    def test_small_transfer_matrix_is_complete(self):
        result = run_increment_transfer_map(
            prefix_lengths=(256,),
            memory_orders=(0.0, 0.30),
            strengths=(0.05, 0.085, 0.2),
            repeats=1,
            calibration_draws=4,
        )
        self.assertEqual(len(result["records"]), 6)
        self.assertEqual(len(result["assessment_by_length"]["256"]["by_strength"]), 3)
        self.assertTrue(result["protocol"]["leave_one_strength_out"])
        self.assertFalse(result["protocol"]["strength_matched_calibration"])


if __name__ == "__main__":
    unittest.main()
