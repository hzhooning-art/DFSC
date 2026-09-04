import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "p4_historical_fault_derived_strategy.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p4_historical_fault_derived_strategy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HistoricalFaultDerivedStrategyTests(unittest.TestCase):
    def test_weak_example_masks_reported_fault(self):
        result = MODULE.run()
        self.assertEqual(result["summary"]["weak_example_detections"], 0)

    def test_equivalence_property_detects_every_paired_variant(self):
        result = MODULE.run()
        self.assertEqual(result["summary"]["equivalence_property_detections"], MODULE.TRIALS)
        self.assertEqual(result["summary"]["fixed_control_passes"], MODULE.TRIALS)

    def test_surrogate_does_not_count_as_environment_pair(self):
        result = MODULE.run()
        self.assertFalse(result["case"]["old_buggy_package_executed"])
        self.assertEqual(result["summary"]["complete_buggy_fixed_environment_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
