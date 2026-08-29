from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_budget_consensus_abstention import (  # noqa: E402
    consensus_decision,
    summarize,
)
from probe_optimizer_budget_stability import LBFGS_BUDGETS  # noqa: E402


class BudgetConsensusAbstentionTests(unittest.TestCase):
    def test_binary_conflict_always_abstains(self) -> None:
        decision, reason = consensus_decision(["RETAIN", "REFUSE", "REFUSE"])
        self.assertEqual(decision, "INDETERMINATE")
        self.assertEqual(reason, "BUDGET_SENSITIVE_BINARY_CONFLICT")

    def test_two_votes_without_opposition_are_determinate(self) -> None:
        self.assertEqual(
            consensus_decision(["RETAIN", "RETAIN", "INDETERMINATE"])[0],
            "RETAIN",
        )
        self.assertEqual(
            consensus_decision(["REFUSE", "INDETERMINATE", "REFUSE"])[0],
            "REFUSE",
        )

    def test_single_determinate_vote_abstains(self) -> None:
        self.assertEqual(
            consensus_decision(["REFUSE", "INDETERMINATE", "INDETERMINATE"])[0],
            "INDETERMINATE",
        )

    def test_summary_requires_complete_matrix(self) -> None:
        records = []
        for drift in (0.0, 0.05, 0.15):
            for repeat in range(24):
                for budget in LBFGS_BUDGETS:
                    records.append(
                        {
                            "channels": 32,
                            "noise_std_diagnostic": 0.0004,
                            "heterogeneity_construction": "antisymmetric",
                            "log_spectral_drift": drift,
                            "repeat": repeat,
                            "seed": 100000 + int(drift * 1000) + repeat,
                            "lbfgs_steps": budget,
                            "decision": "ACCEPT_SHARED_MECHANISM"
                            if drift < 0.15
                            else "REFUSE_SHARED_MECHANISM",
                            "decision_class": "RETAIN" if drift < 0.15 else "REFUSE",
                            "diagnostics_in_calibration_scope": True,
                        }
                    )
        summary = summarize(records)
        self.assertTrue(summary["checks"]["complete_independent_matrix"])
        self.assertTrue(summary["route_pass"])


if __name__ == "__main__":
    unittest.main()
