from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_high_dimensional_shared_spectrum import DTYPE, DEVICE, independent_lifted_response  # noqa: E402
from probe_nested_group_sharing_gate import classify, grouped_response, summarize  # noqa: E402


class NestedGroupSharingGateTests(unittest.TestCase):
    def test_grouped_response_matches_channel_specific_response(self) -> None:
        times = torch.linspace(0.0, 1.0, 5, dtype=DTYPE, device=DEVICE)
        weights = torch.tensor([[0.2, 0.1], [0.3, 0.15], [0.4, 0.2], [0.5, 0.25]], dtype=DTYPE, device=DEVICE)
        group_rates = torch.tensor([[0.25, 1.0], [0.30, 0.90]], dtype=DTYPE, device=DEVICE)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=DEVICE)
        actual = grouped_response(times, weights, group_rates, labels)
        expected = independent_lifted_response(times, weights, group_rates[labels])
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12))

    def test_two_tier_classification(self) -> None:
        self.assertEqual(classify(2.0, 8e-4, 7e-4), "ACCEPT_SHARED_MECHANISM")
        self.assertEqual(classify(20.0, 9e-4, 8e-4), "ACCEPT_WITH_SCOPE_LIMITS")
        self.assertEqual(classify(20.0, 2e-3, 1e-3), "REFUSE_SHARED_MECHANISM")

    def test_summary_frozen_rule(self) -> None:
        records = []
        for drift in (0.0, 0.05, 0.15):
            for rho in (0.0, 0.60):
                for _ in range(3):
                    records.append({
                        "log_spectral_drift": drift,
                        "noise_correlation": rho,
                        "decision": "REFUSE_SHARED_MECHANISM" if drift == 0.15 else "ACCEPT_WITH_SCOPE_LIMITS",
                        "group_bic_support": 20.0,
                        "shared_val_rmse": 8e-4 if drift < 0.15 else 4e-3,
                        "shared_to_grouped_val_ratio": 1.1 if drift < 0.15 else 2.0,
                    })
        self.assertTrue(summarize(records)["route_pass"])


if __name__ == "__main__":
    unittest.main()
