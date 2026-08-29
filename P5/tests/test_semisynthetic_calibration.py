import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from probe_semisynthetic_acceptance_calibration import simulate  # noqa: E402


class SemisyntheticCalibrationTests(unittest.TestCase):
    def test_declared_grouping_and_finite_values(self):
        curves = simulate(6501, "separated")
        self.assertEqual(len(curves), 12)
        self.assertEqual(len({row.group for row in curves}), 4)
        self.assertTrue(all(len(row.time) == 64 for row in curves))


if __name__ == "__main__":
    unittest.main()
