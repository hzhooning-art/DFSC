import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from probe_common_budget_order_detection import run_trial, summarize  # noqa: E402


class CommonBudgetOrderDetectionTests(unittest.TestCase):
    def test_clear_two_rate_trial_is_supported(self):
        row = run_trial(2, 16.0, 48, 0.001, "white", 0.32, 99123)
        self.assertEqual(row["methods"]["selective_detector"]["decision"], 2)

    def test_summary_counts_abstention_outside_selective_accuracy(self):
        records = [
            {"true_rank": 1, "methods": {"m": {"decision": 1, "runtime_seconds": 0.1}}},
            {"true_rank": 2, "methods": {"m": {"decision": None, "runtime_seconds": 0.2}}},
        ]
        result = summarize(records)["m"]
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["selective_accuracy"], 1.0)
        self.assertEqual(result["overall_accuracy_abstention_as_error"], 0.5)
        self.assertEqual(result["abstention_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
