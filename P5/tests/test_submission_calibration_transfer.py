import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "submission_calibration_transfer.json"


class SubmissionCalibrationTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OUTPUT.exists():
            raise AssertionError(f"missing submission calibration artifact: {OUTPUT}")
        cls.payload = json.loads(OUTPUT.read_text(encoding="utf-8"))

    def test_seed_partitions_are_disjoint_and_large_enough(self):
        design = self.payload["predeclared_design"]
        calibration = set(design["calibration_seeds"])
        evaluation = set(design["evaluation_seeds"])
        self.assertTrue(calibration.isdisjoint(evaluation))
        self.assertGreaterEqual(len(evaluation), 40)

    def test_four_noise_generators_are_evaluated_without_retuning(self):
        expected = {"iid_gaussian", "ar1", "ar2", "heteroscedastic"}
        transfer = self.payload["transfer_evaluation"]
        self.assertEqual(set(transfer), expected)
        thresholds = {row["frozen_threshold"] for row in transfer.values()}
        self.assertEqual(len(thresholds), 1)

    def test_principal_metrics_have_wilson_intervals(self):
        principal = self.payload["principal_evaluation"]
        for key in ("separated_support", "coalesced_refusal"):
            metric = principal[key]
            self.assertGreaterEqual(metric["n"], 40)
            self.assertLessEqual(metric["successes"], metric["n"])
            low, high = metric["wilson_95"]
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)
            self.assertLessEqual(low, metric["rate"])
            self.assertGreaterEqual(high, metric["rate"])

    def test_claim_boundary_remains_design_conditional(self):
        boundary = self.payload["claim_boundary"].lower()
        self.assertIn("design conditional", boundary)
        self.assertIn("not a universal", boundary)


if __name__ == "__main__":
    unittest.main()
