import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "p4_cross_project_fixed_regressions.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p4_cross_project_fixed_regressions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CrossProjectFixedRegressionTests(unittest.TestCase):
    def test_scipy_8906_current_fixed_behavior(self):
        record = MODULE.replay_scipy_8906()
        self.assertTrue(record["role_confirmed"])
        self.assertTrue(record["observation"]["right_hand_side_unchanged"])

    def test_project_expansion_does_not_create_complete_pairs(self):
        result = MODULE.run()
        self.assertEqual(result["summary"]["project_count"], 2)
        self.assertEqual(result["summary"]["complete_pairs"], 0)
        self.assertEqual(result["summary"]["fixed_roles_confirmed"], 3)


if __name__ == "__main__":
    unittest.main()
