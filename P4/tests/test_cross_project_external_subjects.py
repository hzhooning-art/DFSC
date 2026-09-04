import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "p4_cross_project_external_subjects.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p4_cross_project_external_subjects", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CrossProjectExternalSubjectTests(unittest.TestCase):
    def test_clean_controls_pass_all_strategies(self):
        result = MODULE.run()
        self.assertTrue(all(row["false_rejections"] == 0 for row in result["summary"].values()))

    def test_three_independent_projects_are_exercised(self):
        result = MODULE.run()
        self.assertEqual(len(result["design"]["independent_sut_projects"]), 3)

    def test_strategy_coverage_is_monotone_and_complete(self):
        result = MODULE.run()
        detections = [row["detected_injected_trials"] for row in result["summary"].values()]
        self.assertEqual(detections, sorted(detections))
        self.assertEqual(detections[-1], 3 * 7 * 12)


if __name__ == "__main__":
    unittest.main()
