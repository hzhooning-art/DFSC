import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from p4_testing_strategy_comparison import STRATEGIES, evaluate_strategy  # noqa: E402


class TestingStrategyComparisonTests(unittest.TestCase):
    def test_fault_class_coverage_is_monotone(self):
        rows = [evaluate_strategy(*strategy) for strategy in STRATEGIES]
        coverage = [row["fully_detected_fault_classes"] for row in rows]
        self.assertEqual(coverage, sorted(coverage))
        self.assertEqual(coverage[-1], 10)

    def test_clean_records_are_not_rejected(self):
        rows = [evaluate_strategy(*strategy) for strategy in STRATEGIES]
        self.assertTrue(all(row["false_rejections"] == 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
