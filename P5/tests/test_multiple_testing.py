import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p5_memory_protocol import holm_adjust


class HolmAdjustmentTests(unittest.TestCase):
    def test_holm_adjustment_preserves_names_and_step_down_order(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertEqual(set(adjusted), {"a", "b", "c"})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)

    def test_holm_adjustment_validates_probability_range(self):
        with self.assertRaises(ValueError):
            holm_adjust({"bad": 1.1})

    def test_empty_family_is_supported(self):
        self.assertEqual(holm_adjust({}), {})


if __name__ == "__main__":
    unittest.main()

