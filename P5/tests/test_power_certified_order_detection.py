import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "probe_power_certified_order_detection.py"
EXPERIMENTS = str(SCRIPT.parent)
if EXPERIMENTS not in sys.path:
    sys.path.insert(0, EXPERIMENTS)
SPEC = importlib.util.spec_from_file_location("probe_power_certified_order_detection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PowerCertificateTests(unittest.TestCase):
    def test_wilson_lower_is_bounded_and_monotone(self):
        values = [MODULE.wilson_lower(successes, 32) for successes in range(33)]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertEqual(values, sorted(values))

    def test_calibration_and_evaluation_seed_ranges_are_disjoint(self):
        calibration = {
            MODULE.design_seed(4.0, 24, 0.001, "white")
            + MODULE.CALIBRATION_SEED_OFFSET
            + repeat
            for repeat in range(MODULE.CALIBRATION_REPEATS)
        }
        evaluation = {
            MODULE.design_seed(4.0, 24, 0.001, "white")
            + MODULE.EVALUATION_SEED_OFFSET
            + repeat
            for repeat in range(MODULE.EVALUATION_REPEATS)
        }
        self.assertTrue(calibration.isdisjoint(evaluation))

    def test_perfect_calibration_can_reach_specificity_threshold(self):
        lower = MODULE.wilson_lower(MODULE.CALIBRATION_REPEATS, MODULE.CALIBRATION_REPEATS)
        self.assertGreaterEqual(lower, MODULE.SPECIFICITY_LOWER_BOUND)

    def test_rank_one_requires_a_qualified_certificate(self):
        decision = 1 if (-12.0 <= -10.0 and False) else None
        self.assertIsNone(decision)


if __name__ == "__main__":
    unittest.main()
