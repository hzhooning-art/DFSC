import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SciPyResamplePolyPairCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((ROOT / "results" / "p4_scipy_resample_poly_pair.json").read_text(encoding="utf-8"))

    def test_runner_is_frozen_and_both_roles_pass(self):
        runner = ROOT / "experiments" / "scipy_resample_poly_historical_pair_runner.py"
        self.assertEqual(hashlib.sha256(runner.read_bytes()).hexdigest(), self.payload["runner_sha256"])
        self.assertTrue(self.payload["runner_unchanged_between_sides"])
        self.assertTrue(self.payload["buggy_side"]["result"]["role_confirmed"])
        self.assertTrue(self.payload["fixed_side"]["result"]["role_confirmed"])

    def test_buggy_release_reproduces_silent_integer_failure(self):
        observation = self.payload["buggy_side"]["result"]["observation"]
        self.assertTrue(observation["integer_outputs_all_zero"])
        self.assertTrue(observation["float_reference_nonzero"])
        self.assertAlmostEqual(observation["integer_max_abs_error_vs_float64"]["int16"], 3.0015524117881807)

    def test_fixed_release_matches_float_reference(self):
        observation = self.payload["fixed_side"]["result"]["observation"]
        self.assertTrue(observation["integer_outputs_match_reference"])
        self.assertEqual(observation["integer_max_abs_error_vs_float64"]["int16"], 0.0)
        self.assertEqual(self.payload["historical_family_ordinal"], 3)


if __name__ == "__main__":
    unittest.main()
