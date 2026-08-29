import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p5_memory_protocol import holm_adjust


class MultipleComparisonTests(unittest.TestCase):
    def test_holm_adjust_matches_step_down_definition(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.06)

    def test_holm_adjust_is_order_independent_and_not_smaller_than_raw(self):
        raw = {"late": 0.024, "early": 1e-6, "middle": 0.001}
        adjusted = holm_adjust(raw)
        self.assertEqual(set(adjusted), set(raw))
        self.assertTrue(all(adjusted[key] >= value for key, value in raw.items()))
        self.assertLessEqual(adjusted["early"], adjusted["middle"])
        self.assertLessEqual(adjusted["middle"], adjusted["late"])


if __name__ == "__main__":
    unittest.main()
