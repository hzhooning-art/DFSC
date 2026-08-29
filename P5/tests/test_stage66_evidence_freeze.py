import json
import unittest
from pathlib import Path


RESULT = Path(__file__).resolve().parents[1] / "results" / "stage66_extension_combination.json"


class Stage66EvidenceFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_predeclared_route_checks_remain_satisfied(self):
        self.assertEqual(self.payload["stage"], "66-recommended-extension-combination")
        self.assertEqual(len(self.payload["seeds"]), 12)
        self.assertTrue(all(self.payload["checks"].values()))
        self.assertTrue(self.payload["route_pass"])

    def test_reported_summaries_match_frozen_evidence(self):
        osc = self.payload["oscillatory"]["summary"]
        partial = self.payload["partial_sharing"]["summary"]
        spectrum = self.payload["continuous_spectrum"]["summary"]
        conformal = self.payload["group_conformal"]["summary"]

        self.assertLess(osc["median_oscillatory_nrmse"], 0.003)
        self.assertGreater(osc["median_relative_gain"], 0.98)
        self.assertGreater(partial["median_relative_gain"], 0.90)
        self.assertGreaterEqual(spectrum["refusal_rate"], 11.0 / 12.0)
        self.assertLessEqual(conformal["mean_null_false_alarm_rate"], 0.050000000000001)
        self.assertEqual(conformal["mean_shift_detection_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
