from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_extended_refinement_transfer import (  # noqa: E402
    EXTENDED_LBFGS_STEPS,
    load_frozen_stage48,
    severe_rates,
)


class ExtendedRefinementTransferTests(unittest.TestCase):
    def test_extended_budget_exceeds_baseline(self) -> None:
        self.assertGreater(EXTENDED_LBFGS_STEPS, 80)

    def test_frozen_calibration_loader(self) -> None:
        payload = {
            "frozen_noise_calibration": {"noise_proxy_min": 0.1},
            "frozen_consistency_calibration": {"objective_gap_threshold": 0.2},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            noise, consistency = load_frozen_stage48(path)
        self.assertEqual(noise["noise_proxy_min"], 0.1)
        self.assertEqual(consistency["objective_gap_threshold"], 0.2)

    def test_severe_rate_summary(self) -> None:
        summary = {
            "rows": [
                {
                    "log_spectral_drift": 0.15,
                    "refuse_fraction": 2.0 / 3.0,
                    "indeterminate_fraction": 1.0 / 3.0,
                },
                {
                    "log_spectral_drift": 0.15,
                    "refuse_fraction": 1.0,
                    "indeterminate_fraction": 0.0,
                },
                {
                    "log_spectral_drift": 0.05,
                    "refuse_fraction": 0.0,
                    "indeterminate_fraction": 0.0,
                },
            ]
        }
        rates = severe_rates(summary)
        self.assertEqual(rates["cells_meeting_explicit_refusal_rule"], 2)
        self.assertAlmostEqual(rates["mean_refuse_fraction"], 5.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
