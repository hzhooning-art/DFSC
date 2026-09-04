import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CableWindowSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((ROOT / "results" / "cable_window_sensitivity.json").read_text(encoding="utf-8"))

    def test_all_frozen_windows_are_present(self):
        self.assertEqual(self.payload["window_count"], 12)
        self.assertEqual(len(self.payload["records"]), 12)

    def test_success_requires_scope_and_decision_stability(self):
        self.assertEqual(self.payload["scope_eligible_count"], 12)
        self.assertEqual(self.payload["matching_parent_decision_count"], 12)
        self.assertTrue(self.payload["success_rule_passes"])

    def test_claim_does_not_promote_overlapping_windows(self):
        self.assertIn("not independent", self.payload["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
