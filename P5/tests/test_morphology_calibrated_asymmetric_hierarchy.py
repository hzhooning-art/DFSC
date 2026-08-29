import math
import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_morphology_calibrated_asymmetric_hierarchy import (  # noqa: E402
    calibrate_thresholds,
    morphology_decision,
    repeated_structural_score,
)


class MorphologyCalibratedHierarchyTests(unittest.TestCase):
    def test_repeated_structural_score_uses_second_largest(self):
        records = [
            {"group_bic_support": 2.0},
            {"group_bic_support": 5.0},
            {"group_bic_support": 3.0},
        ]
        self.assertEqual(repeated_structural_score(records), 3.0)

    def test_thresholds_are_strictly_above_acceptable_envelope(self):
        pairs = [
            {"truth_class": "RETAIN", "validation_score": 1.0, "structural_score": 2.0},
            {"truth_class": "RETAIN", "validation_score": 1.2, "structural_score": 3.0},
            {"truth_class": "REFUSE", "validation_score": 4.0, "structural_score": 8.0},
        ]
        thresholds = calibrate_thresholds(pairs)
        self.assertGreater(thresholds["validation_threshold"], 1.2)
        self.assertGreater(thresholds["structural_threshold"], 3.0)

    def test_strong_structural_score_can_refuse(self):
        pair = {
            "numerical_eligible": True,
            "structural_score": 4.0,
            "validation_score": 0.5,
            "structural_refuse_votes": 0,
            "validation_axis": "INDETERMINATE",
            "structural_axis": "INDETERMINATE",
        }
        self.assertEqual(morphology_decision(pair, 1.0, 3.0)[0], "REFUSE")

    def test_validation_refusal_requires_structural_support(self):
        pair = {
            "numerical_eligible": True,
            "structural_score": 2.0,
            "validation_score": 4.0,
            "structural_refuse_votes": 0,
            "validation_axis": "REFUSE",
            "structural_axis": "RETAIN",
        }
        self.assertEqual(morphology_decision(pair, 1.0, 3.0)[0], "INDETERMINATE")
        pair["structural_refuse_votes"] = 1
        self.assertEqual(morphology_decision(pair, 1.0, 3.0)[0], "REFUSE")


if __name__ == "__main__":
    unittest.main()
