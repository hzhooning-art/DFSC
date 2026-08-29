import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_asymmetric_evidence_hierarchy import (  # noqa: E402
    asymmetric_decision,
    repeated_exceedance_score,
)


class AsymmetricEvidenceHierarchyTests(unittest.TestCase):
    def test_repeated_exceedance_uses_second_largest_budget_score(self):
        records = [
            {"shared_val_rmse": 1.0, "adjusted_tolerance": 1.0},
            {"shared_val_rmse": 2.0, "adjusted_tolerance": 1.0},
            {"shared_val_rmse": 3.0, "adjusted_tolerance": 1.0},
        ]
        self.assertEqual(repeated_exceedance_score(records), 2.0)

    def test_structural_consensus_can_refuse_without_validation_agreement(self):
        decision, reason = asymmetric_decision(True, "INDETERMINATE", "REFUSE", 2, 1.0, 1.1)
        self.assertEqual(decision, "REFUSE")
        self.assertEqual(reason, "STRUCTURAL_CONSENSUS_REFUSAL")

    def test_strong_validation_requires_structural_support(self):
        unsupported = asymmetric_decision(True, "REFUSE", "RETAIN", 0, 2.0, 1.1)
        supported = asymmetric_decision(True, "REFUSE", "INDETERMINATE", 1, 2.0, 1.1)
        self.assertEqual(unsupported[0], "INDETERMINATE")
        self.assertEqual(supported[0], "REFUSE")

    def test_retention_remains_concordant(self):
        self.assertEqual(asymmetric_decision(True, "RETAIN", "RETAIN", 0, 0.5, 1.1)[0], "RETAIN")
        self.assertEqual(asymmetric_decision(True, "RETAIN", "INDETERMINATE", 1, 0.5, 1.1)[0], "INDETERMINATE")


if __name__ == "__main__":
    unittest.main()
