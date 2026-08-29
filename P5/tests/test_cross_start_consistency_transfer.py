from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_cross_start_consistency_transfer import (  # noqa: E402
    cross_start_diagnostics,
    fit_consistency_calibration,
    select_consistent_candidate,
)
from probe_high_dimensional_shared_spectrum import DEVICE, DTYPE  # noqa: E402


def candidate(train_rmse: float, rates=(0.25, 1.0), weights=((0.4, 0.3),)):
    return SimpleNamespace(
        train_rmse=train_rmse,
        val_rmse=train_rmse,
        rates=list(rates),
        weights=[list(row) for row in weights],
    )


class CrossStartConsistencyTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.times = torch.linspace(0.0, 2.0, 10, dtype=DTYPE, device=DEVICE)
        self.observations = torch.ones((10, 1), dtype=DTYPE, device=DEVICE)

    def test_identical_candidates_have_zero_gaps(self) -> None:
        _, diagnostics, _ = cross_start_diagnostics(
            [candidate(0.2), candidate(0.2)], self.times, self.observations
        )
        self.assertEqual(diagnostics[1]["train_objective_gap"], 0.0)
        self.assertEqual(diagnostics[1]["prediction_gap"], 0.0)

    def test_calibration_uses_exact_control_maxima(self) -> None:
        records = [
            {
                "second_start_objective_gap": objective,
                "second_start_prediction_gap": prediction,
            }
            for objective, prediction in ((0.01, 0.03), (0.02, 0.02), (0.015, 0.04))
        ]
        calibration = fit_consistency_calibration(records)
        self.assertEqual(calibration["objective_gap_threshold"], 0.02)
        self.assertEqual(calibration["prediction_gap_threshold"], 0.04)

    def test_selection_does_not_use_absolute_residual(self) -> None:
        candidates = [candidate(10.0), candidate(10.0)]
        calibration = {
            "objective_gap_threshold": 0.0,
            "prediction_gap_threshold": 0.0,
            "minimum_consistent_starts": 2,
        }
        selected, _, count, _ = select_consistent_candidate(
            candidates, self.times, self.observations, calibration
        )
        self.assertIsNotNone(selected)
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
