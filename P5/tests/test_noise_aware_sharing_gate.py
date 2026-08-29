from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_high_dimensional_shared_spectrum import DTYPE, DEVICE  # noqa: E402
from probe_noise_aware_sharing_gate import (  # noqa: E402
    calibrated_limit,
    classify_with_limit,
    fit_calibration_envelope,
    second_difference_correlation_proxy,
)


class NoiseAwareSharingGateTests(unittest.TestCase):
    def test_proxy_tracks_common_noise(self) -> None:
        generator = torch.Generator(device=DEVICE).manual_seed(7)
        independent = torch.randn((65, 32), dtype=DTYPE, device=DEVICE, generator=generator)
        common = torch.randn((65, 1), dtype=DTYPE, device=DEVICE, generator=generator)
        correlated = 0.8**0.5 * common + 0.2**0.5 * independent
        self.assertGreater(
            second_difference_correlation_proxy(correlated),
            second_difference_correlation_proxy(independent) + 0.4,
        )

    def test_calibration_limit_is_monotone(self) -> None:
        records = [
            {"correlation_proxy": 0.0, "shared_val_rmse": 0.0020},
            {"correlation_proxy": 0.2, "shared_val_rmse": 0.0022},
            {"correlation_proxy": 0.4, "shared_val_rmse": 0.0025},
            {"correlation_proxy": 0.6, "shared_val_rmse": 0.0028},
        ]
        envelope = fit_calibration_envelope(records)
        self.assertGreaterEqual(calibrated_limit(0.6, envelope), calibrated_limit(0.0, envelope))

    def test_classification_uses_supplied_limit(self) -> None:
        self.assertEqual(
            classify_with_limit(-10.0, 0.0032, 0.0040, 0.0035),
            "ACCEPT_SHARED_MECHANISM",
        )
        self.assertEqual(
            classify_with_limit(-10.0, 0.0036, 0.0040, 0.0035),
            "REFUSE_SHARED_MECHANISM",
        )


if __name__ == "__main__":
    unittest.main()
