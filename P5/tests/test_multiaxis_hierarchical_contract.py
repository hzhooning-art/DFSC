import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_multiaxis_hierarchical_contract import (  # noqa: E402
    decision_metrics,
    hierarchical_decision,
    structural_class,
)


class MultiaxisHierarchicalContractTests(unittest.TestCase):
    def test_cross_axis_conflict_abstains(self):
        decision, reason = hierarchical_decision(True, "RETAIN", "REFUSE")
        self.assertEqual(decision, "INDETERMINATE")
        self.assertEqual(reason, "CROSS_AXIS_CONFLICT")

    def test_numerical_ineligibility_precedes_axis_agreement(self):
        decision, reason = hierarchical_decision(False, "REFUSE", "REFUSE")
        self.assertEqual(decision, "INDETERMINATE")
        self.assertEqual(reason, "NUMERICAL_AXIS_INELIGIBLE")

    def test_structural_class_uses_frozen_bic_limit(self):
        self.assertEqual(structural_class({"group_bic_support": 1.0e9}), "REFUSE")
        self.assertEqual(structural_class({"group_bic_support": -1.0}), "RETAIN")
        self.assertEqual(structural_class({"group_bic_support": None}), "INDETERMINATE")

    def test_selective_accuracy_excludes_abstentions(self):
        pairs = [
            {"truth_class": "RETAIN", "prediction": "RETAIN"},
            {"truth_class": "REFUSE", "prediction": "INDETERMINATE"},
        ]
        metrics = decision_metrics(pairs, "prediction")
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["selective_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
