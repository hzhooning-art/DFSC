from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_decomposed_tolerance_transfer import (  # noqa: E402
    build_transfer_observation,
    frozen_total_tolerance,
)
from probe_noise_scale_optimizer_transfer import (  # noqa: E402
    fit_calibration,
    mixed_difference_noise_scale,
    scale_correction,
)


class NoiseScaleOptimizerTransferTests(unittest.TestCase):
    def test_noise_proxy_tracks_scale(self) -> None:
        low = build_transfer_observation(64, 3.0e-4, "antisymmetric", 0.0, 101)[1]
        high = build_transfer_observation(64, 1.8e-3, "curved", 0.0, 102)[1]
        self.assertGreater(
            mixed_difference_noise_scale(high),
            2.0 * mixed_difference_noise_scale(low),
        )

    def test_calibration_is_one_sided_and_has_scope(self) -> None:
        records = []
        for index, proxy in enumerate((0.0002, 0.0004, 0.0008, 0.0012)):
            records.append(
                {
                    "noise_scale_proxy": proxy,
                    "consensus_val_rmse": 0.002 + 1.5 * proxy,
                    "stage45_total_tolerance": 0.0018,
                    "second_best_train_to_noise_ratio": 1.0 + 0.1 * index,
                    "noise_std_diagnostic": proxy,
                }
            )
        calibration = fit_calibration(records)
        self.assertGreaterEqual(calibration["scale_correction_slope"], 0.0)
        self.assertLessEqual(calibration["noise_proxy_min"], 0.0002)
        self.assertGreaterEqual(calibration["noise_proxy_max"], 0.0012)

    def test_stage45_budget_remains_a_floor(self) -> None:
        calibration = {
            "scale_correction_intercept": -1.0,
            "scale_correction_slope": 0.0,
            "scale_correction_one_sided_residual": 0.0,
        }
        base = frozen_total_tolerance(0.5)
        augmented = base + scale_correction(0.001, calibration)
        self.assertEqual(augmented, base)


if __name__ == "__main__":
    unittest.main()
