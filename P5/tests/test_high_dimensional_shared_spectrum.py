from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_high_dimensional_shared_spectrum import (  # noqa: E402
    DTYPE,
    DEVICE,
    independent_lifted_response,
    summarize,
)


class HighDimensionalSharedSpectrumTests(unittest.TestCase):
    def test_independent_response_shape_and_shared_equivalence(self) -> None:
        times = torch.linspace(0.0, 2.0, 9, dtype=DTYPE, device=DEVICE)
        weights = torch.tensor(
            [[0.20, 0.10], [0.35, 0.15], [0.50, 0.20]],
            dtype=DTYPE,
            device=DEVICE,
        )
        shared_rates = torch.tensor([0.25, 1.0], dtype=DTYPE, device=DEVICE)
        independent_rates = shared_rates.expand(weights.shape[0], -1).clone()

        response = independent_lifted_response(times, weights, independent_rates)
        self.assertEqual(tuple(response.shape), (9, 3))

        from probe_memory_rank import lifted_response

        expected = lifted_response(times, weights, shared_rates)
        self.assertTrue(torch.allclose(response, expected, atol=1.0e-12, rtol=1.0e-12))

    def test_summary_applies_frozen_route_rule(self) -> None:
        records = []
        for channels in (1, 16, 64, 256):
            for repeat in range(3):
                records.append(
                    {
                        "channels": channels,
                        "repeat": repeat,
                        "shared_resolved": channels != 1,
                        "shared_rate_error": 0.12 if channels == 1 else 0.06,
                        "shared_val_rmse": 8.0e-4,
                        "shared_bic_support": 20.0,
                        "independent_resolved_fraction": 0.10,
                        "independent_median_rate_error": 0.50,
                        "shared_elapsed_seconds": float(channels),
                        "independent_elapsed_seconds": float(channels) * 1.5,
                        "peak_memory_bytes": channels * 1024,
                    }
                )

        result = summarize(records)
        self.assertTrue(result["route_pass"])
        self.assertEqual(result["minimum_passing_channels"], 16)
        self.assertEqual(result["frozen_rule"]["required_channel_counts"], [16, 64, 256])


if __name__ == "__main__":
    unittest.main()
