from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_optimizer_budget_stability import (  # noqa: E402
    LBFGS_BUDGETS,
    decision_class,
    paired_summary,
)


class OptimizerBudgetStabilityTests(unittest.TestCase):
    def test_budgets_are_strictly_increasing(self) -> None:
        self.assertEqual(tuple(sorted(set(LBFGS_BUDGETS))), LBFGS_BUDGETS)

    def test_decision_class_preserves_tri_state(self) -> None:
        self.assertEqual(decision_class("ACCEPT_WITH_SCOPE_LIMITS"), "RETAIN")
        self.assertEqual(decision_class("REFUSE_SHARED_MECHANISM"), "REFUSE")
        self.assertEqual(
            decision_class("INDETERMINATE_OPTIMIZATION"), "INDETERMINATE"
        )

    def test_direct_reversal_is_not_created_by_indeterminate_only(self) -> None:
        records = []
        for repeat in range(24):
            for budget, decision in zip(
                LBFGS_BUDGETS,
                (
                    "ACCEPT_SHARED_MECHANISM",
                    "INDETERMINATE_OPTIMIZATION",
                    "ACCEPT_WITH_SCOPE_LIMITS",
                ),
            ):
                records.append(
                    {
                        "channels": 32,
                        "noise_std_diagnostic": 0.0004,
                        "heterogeneity_construction": "antisymmetric",
                        "repeat": repeat,
                        "seed": 1000 + repeat,
                        "lbfgs_steps": budget,
                        "decision": decision,
                        "decision_class": decision_class(decision),
                        "diagnostics_in_calibration_scope": True,
                    }
                )
        summary = paired_summary(records)
        self.assertEqual(summary["direct_reversal_fraction"], 0.0)
        self.assertEqual(summary["indeterminate_pair_fraction"], 1.0)
        self.assertFalse(summary["route_pass"])


if __name__ == "__main__":
    unittest.main()
