from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_decomposed_tolerance_sharing_gate import (  # noqa: E402
    decomposed_limit,
    fit_model_allowance,
    fit_proxy_scope,
)


class DecomposedToleranceSharingGateTests(unittest.TestCase):
    def test_model_allowance_is_nonnegative(self) -> None:
        envelope = {"intercept": 0.001, "slope": 0.001}
        records = [
            {"correlation_proxy": 0.1, "shared_val_rmse": 0.0018},
            {"correlation_proxy": 0.5, "shared_val_rmse": 0.0024},
        ]
        budget = fit_model_allowance(records, envelope)
        self.assertGreaterEqual(budget["allowance"], 0.0)

    def test_decomposed_limit_adds_model_budget(self) -> None:
        envelope = {
            "intercept": 0.001,
            "slope": 0.001,
            "one_sided_residual": 0.0002,
        }
        budget = {"allowance": 0.0007}
        self.assertAlmostEqual(decomposed_limit(0.5, envelope, budget), 0.0024)

    def test_proxy_scope_uses_calibration_only_padding(self) -> None:
        records = [
            {"noise_correlation_diagnostic": 0.0, "correlation_proxy": 0.10},
            {"noise_correlation_diagnostic": 0.0, "correlation_proxy": 0.12},
            {"noise_correlation_diagnostic": 0.6, "correlation_proxy": 0.60},
            {"noise_correlation_diagnostic": 0.6, "correlation_proxy": 0.64},
        ]
        scope = fit_proxy_scope(records)
        self.assertLess(scope["proxy_min"], 0.10)
        self.assertGreater(scope["proxy_max"], 0.64)


if __name__ == "__main__":
    unittest.main()
