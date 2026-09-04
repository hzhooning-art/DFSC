import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "p4_external_subject_heterogeneity.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p4_external_subject_heterogeneity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExternalSubjectHeterogeneityTests(unittest.TestCase):
    def test_wilson_interval_is_bounded(self):
        for successes in range(11):
            low, high = MODULE.wilson_interval(successes, 10)
            self.assertLessEqual(0.0, low)
            self.assertLessEqual(low, high)
            self.assertLessEqual(high, 1.0)

    def test_jss_gate_cannot_pass_single_project_pilot(self):
        result = MODULE.run()
        self.assertFalse(result["jss_readiness_gate"]["passes"])
        self.assertEqual(result["jss_readiness_gate"]["observed_independent_sut_projects"], 1)

    def test_complete_strategy_has_no_cluster_regression(self):
        result = MODULE.run()
        final_increment = result["paired_strategy_increments"][-1]
        self.assertEqual(final_increment["regressed_clusters"], 0)


if __name__ == "__main__":
    unittest.main()
