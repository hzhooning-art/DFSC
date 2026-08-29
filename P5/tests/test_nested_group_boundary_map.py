from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_nested_group_boundary_map import interpolate_boundary, summarize, wilson_interval  # noqa: E402


class NestedGroupBoundaryMapTests(unittest.TestCase):
    def test_wilson_interval_is_bounded(self) -> None:
        low, high = wilson_interval(3, 5)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)

    def test_boundary_interpolation(self) -> None:
        rows = [
            {"log_spectral_drift": 0.075, "noise_correlation": 0.0, "refuse_fraction": 0.2},
            {"log_spectral_drift": 0.100, "noise_correlation": 0.0, "refuse_fraction": 0.6},
        ]
        self.assertAlmostEqual(interpolate_boundary(rows, 0.0), 0.09375)

    def test_frozen_summary_rule(self) -> None:
        records = []
        for drift, refused in ((0.075, 1), (0.100, 3), (0.125, 4)):
            for rho in (0.0, 0.60):
                for repeat in range(5):
                    records.append(
                        {
                            "log_spectral_drift": drift,
                            "noise_correlation": rho,
                            "decision": "REFUSE_SHARED_MECHANISM" if repeat < refused else "ACCEPT_SHARED_MECHANISM",
                            "group_bic_support": float(refused),
                            "shared_val_rmse": 1.0e-3 * (1 + refused),
                            "shared_to_grouped_val_ratio": 1.0,
                        }
                    )
        self.assertTrue(summarize(records)["boundary_map_pass"])


if __name__ == "__main__":
    unittest.main()
