import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "p4_historical_defect_intake.py"
SPEC = importlib.util.spec_from_file_location("p4_historical_defect_intake", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HistoricalDefectIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = MODULE.run()

    def test_current_xlogy_fixed_behavior_passes(self):
        case = next(row for row in self.result["cases"] if row["case_id"].startswith("pytorch_80770"))
        self.assertTrue(case["current_fixed_replay"]["passes_fixed_expectation"])

    def test_fixed_only_replay_is_not_complete_pair(self):
        self.assertEqual(self.result["summary"]["complete_buggy_fixed_pairs"], 0)
        self.assertFalse(self.result["jss_readiness_gate"]["passes"])

    def test_candidate_projects_are_not_counted_as_verified(self):
        projects = {row["project"] for row in self.result["cases"]}
        self.assertGreaterEqual(len(projects), 3)
        self.assertEqual(self.result["jss_readiness_gate"]["observed_projects_with_complete_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
