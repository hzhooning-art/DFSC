import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SciPyCompletePairCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((ROOT / "results" / "p4_scipy_complete_pair.json").read_text(encoding="utf-8"))

    def test_runner_is_frozen_and_both_roles_pass(self):
        self.assertTrue(self.payload["runner_unchanged_between_sides"])
        self.assertEqual(self.payload["runner_sha256"], "ce901b112325640d58db45dd31b1244f32e5752de85e1bbf5e3d69454c9e1bc8")
        self.assertTrue(self.payload["buggy_side"]["result"]["role_confirmed"])
        self.assertTrue(self.payload["fixed_side"]["result"]["role_confirmed"])

    def test_buggy_release_exposes_representation_asymmetry(self):
        observation = self.payload["buggy_side"]["result"]["observation"]
        self.assertFalse(observation["conventional_u0"]["executed"])
        self.assertEqual(observation["conventional_u0"]["exception"], "IndexError")
        self.assertTrue(observation["padded_u1"]["executed"])

    def test_second_project_pair_is_complete(self):
        self.assertTrue(self.payload["complete_pair"])
        self.assertEqual(self.payload["buggy_side"]["result"]["scipy"], "1.14.1")


if __name__ == "__main__":
    unittest.main()
