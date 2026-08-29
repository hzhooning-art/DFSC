from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_approximate_sharing_refusal_boundary import (  # noqa: E402
    build_block_heterogeneous_observation,
    log_spectrum_dispersion,
    summarize,
)


class ApproximateSharingRefusalTests(unittest.TestCase):
    def test_generator_has_declared_correlation_and_spectral_centre(self) -> None:
        _, _, _, _, rates, labels = build_block_heterogeneous_observation(0.10, 0.60, 7)
        self.assertEqual(rates.shape, (64, 2))
        self.assertEqual(set(labels.tolist()), {0, 1, 2, 3})
        geometric_centre = np.exp(np.log(rates).mean(axis=0))
        self.assertTrue(np.allclose(geometric_centre, [0.25, 1.0], atol=1.0e-12))

    def test_dispersion_is_zero_for_identical_spectra(self) -> None:
        rates = np.tile(np.asarray([0.25, 1.0]), (4, 1))
        self.assertAlmostEqual(log_spectrum_dispersion(rates), 0.0, places=14)

    def test_summary_applies_frozen_accept_and_refuse_rule(self) -> None:
        records = []
        for drift in (0.0, 0.05, 0.15):
            for rho in (0.0, 0.60):
                for repeat in range(3):
                    accepted = drift <= 0.05
                    records.append(
                        {
                            "log_spectral_drift": drift,
                            "noise_correlation": rho,
                            "decision": (
                                "ACCEPT_SHARED_MECHANISM"
                                if accepted
                                else "REFUSE_SHARED_MECHANISM"
                            ),
                            "old_gate_pass": True,
                            "observed_subgroup_dispersion": 0.04 if accepted else 0.16,
                            "true_subgroup_dispersion": drift,
                            "channel_mechanism_distortion": drift,
                            "shared_to_independent_val_ratio": 1.0,
                        }
                    )
        result = summarize(records)
        self.assertTrue(result["route_pass"])
        self.assertTrue(result["checks"]["old_gate_blind_spot_observed"])


if __name__ == "__main__":
    unittest.main()
