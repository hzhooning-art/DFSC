import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "statistical_robustness_audit.json"


class StatisticalRobustnessAuditTests(unittest.TestCase):
    def test_audit_when_present(self):
        if not OUTPUT.exists():
            self.skipTest("joint statistical audit has not been generated")
        payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["pva_threshold_sensitivity"]), 27)
        self.assertEqual(len(payload["gas_threshold_sensitivity"]), 27)
        self.assertEqual(len(payload["pva_leave_one_specimen"]["3"]), 3)
        self.assertEqual(len(payload["gas_leave_one_batch_groups"]), 5)


if __name__ == "__main__":
    unittest.main()
