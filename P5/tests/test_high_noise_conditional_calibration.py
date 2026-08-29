import math
import sys
import unittest
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_high_noise_conditional_calibration import (  # noqa: E402
    CALIBRATION_DRIFTS,
    apply_frozen_calibration,
    fit_conditional_multiplier,
)


def record(ratio: float, drift: float = 0.0) -> dict:
    return {
        "log_spectral_drift": drift,
        "shared_val_rmse": ratio,
        "augmented_total_tolerance": 1.0,
        "group_bic_support": -2.0,
        "grouped_val_rmse": 0.5,
    }


class HighNoiseConditionalCalibrationTests(unittest.TestCase):
    def test_calibration_rejects_severe_drift(self):
        with self.assertRaises(ValueError):
            fit_conditional_multiplier([record(1.0, 0.15)])

    def test_multiplier_is_one_sided_and_capped(self):
        calibration = fit_conditional_multiplier([record(10.0, CALIBRATION_DRIFTS[0])])
        self.assertEqual(calibration["validation_tolerance_multiplier"], 2.5)

    def test_frozen_calibration_changes_only_declared_stage52_fields(self):
        calibration = {"validation_tolerance_multiplier": 2.0}
        updated = apply_frozen_calibration(record(1.5), calibration)
        self.assertTrue(math.isclose(updated["stage52_adjusted_tolerance"], 2.0))
        self.assertEqual(updated["stage52_decision_class"], "RETAIN")


if __name__ == "__main__":
    unittest.main()
