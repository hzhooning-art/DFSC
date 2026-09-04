import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "p4_complete_historical_pair.json"


class CompleteHistoricalPairCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_pair_uses_identical_frozen_runner(self):
        self.assertTrue(self.payload["runner_unchanged_between_sides"])
        self.assertEqual(
            self.payload["runner_sha256"],
            "fdb828ccad16c301981a9334352ae47a337b3a00ca8913e40f4b028334ee4014",
        )

    def test_buggy_and_fixed_roles_are_both_confirmed(self):
        self.assertTrue(self.payload["buggy_side"]["result"]["role_confirmed"])
        self.assertTrue(self.payload["fixed_side"]["result"]["role_confirmed"])
        self.assertEqual(self.payload["buggy_side"]["result"]["torch"], "1.11.0+cpu")

    def test_exactly_one_complete_pair_is_claimed(self):
        self.assertTrue(self.payload["complete_pair"])
        self.assertEqual(self.payload["complete_pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
