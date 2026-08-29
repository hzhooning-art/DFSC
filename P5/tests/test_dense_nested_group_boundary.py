from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_dense_nested_group_boundary import DRIFTS, crossing_bracket, summarize  # noqa: E402


class DenseNestedGroupBoundaryTests(unittest.TestCase):
    def test_crossing_bracket_and_interpolation(self) -> None:
        rows = [
            {"log_spectral_drift": 0.05, "refuse_fraction": 0.25},
            {"log_spectral_drift": 0.06, "refuse_fraction": 0.75},
        ]
        bracket = crossing_bracket(rows)
        self.assertIsNotNone(bracket)
        self.assertAlmostEqual(bracket["linear_estimate"], 0.055)

    def test_no_crossing_returns_none(self) -> None:
        rows = [
            {"log_spectral_drift": 0.05, "refuse_fraction": 0.0},
            {"log_spectral_drift": 0.06, "refuse_fraction": 0.25},
        ]
        self.assertIsNone(crossing_bracket(rows))

    def test_frozen_dense_summary(self) -> None:
        records = []
        refused_by_drift = dict(zip(DRIFTS, (1, 2, 3, 4, 5)))
        for drift in DRIFTS:
            for rho in (0.0, 0.60):
                for repeat in range(6):
                    refused = repeat < refused_by_drift[drift]
                    records.append(
                        {
                            "log_spectral_drift": drift,
                            "noise_correlation": rho,
                            "decision": "REFUSE_SHARED_MECHANISM" if refused else "ACCEPT_SHARED_MECHANISM",
                            "shared_val_rmse": 0.002 + drift * 0.02,
                            "group_bic_support": drift * 100.0,
                        }
                    )
        result = summarize(records)
        self.assertTrue(result["dense_boundary_pass"])
        self.assertIsNotNone(result["crossing_bracket"])


if __name__ == "__main__":
    unittest.main()
