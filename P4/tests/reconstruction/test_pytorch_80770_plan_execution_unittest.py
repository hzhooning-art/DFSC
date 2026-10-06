"""Full PyTorch #80770 plan, saved evidence and fail-closed controls."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from p4_testing.compiler import Budget, compile_plan
from p4_testing.contracts import RiskProfile, load_contract
from p4_testing.execution import execute_plan, load_registry


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
import pytorch_80770_relation_worker as worker  # noqa: E402


def saved_records():
    report = json.loads((ROOT / "results/reconstruction/pytorch_80770_full_execution_111_vs_2110.json")
                        .read_text(encoding="utf-8"))
    return {row["test_id"]: row for row in report["observations"]}


def wrong_gradient(record):
    changed = copy.deepcopy(record)
    result = changed["payload"]["observation"]
    result["observed_gradient"] = 0.0
    result["maximum_absolute_error"] = result["expected_gradient"]
    result["matches_fixed_expectation"] = False
    result["matches_reported_bug"] = True
    changed["payload"]["role_confirmed"] = False
    return changed


class Torch80770ExecutionTests(unittest.TestCase):
    def test_saved_complete_three_relation_report(self):
        plan = json.loads((ROOT / "results/reconstruction/pytorch_80770_full_execution_plan.json")
                          .read_text(encoding="utf-8"))
        report = json.loads((ROOT / "results/reconstruction/pytorch_80770_full_execution_111_vs_2110.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(report["plan_digest"], plan["digest"])
        self.assertEqual(report["status"], "CONFORMANT")
        self.assertEqual(report["selected_relation_count"], 3)
        self.assertEqual(report["reported_subject_invocations"], 5)
        self.assertEqual({row["test_id"]: row["outcome"] for row in report["observations"]},
                         {key: "PASS" for key in worker.TESTS})
        self.assertTrue(all(not row["details"]["captured_pair_file_used_as_observation"]
                            for row in report["observations"]))

    def test_missing_old_interpreter_abstains(self):
        with patch.object(worker, "OLD", ROOT / ".runtime/missing/python.exe"):
            result = worker.run("replay.version_pair")
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["subject_invocations"], 0)

    def test_frozen_runner_hash_change_abstains(self):
        with patch.object(worker, "digest", return_value="wrong"):
            result = worker.run("reference.analytic_derivative")
        self.assertEqual(result["outcome"], "NON_EVALUABLE")

    def test_current_gradient_regression_is_fail_not_abstain(self):
        fixed = wrong_gradient(saved_records()["reference.analytic_derivative"]["details"]["fixed"])
        # Mocked role outputs test validation without requiring old wheels.
        with patch.object(worker, "NEW", Path(sys.executable)), \
                patch.object(worker, "role_run", return_value=fixed) as role_run:
            result = worker.run("reference.analytic_derivative")
        role_run.assert_called_once()
        self.assertEqual(result["outcome"], "FAIL")

    def test_repeatability_mismatch_is_fail(self):
        rows = saved_records()["execution.repeatability"]["details"]
        with patch.object(worker, "NEW", Path(sys.executable)), \
                patch.object(worker, "role_run", side_effect=[rows["fixed_first"],
                                                          wrong_gradient(rows["fixed_second"])]) as role_run:
            result = worker.run("execution.repeatability")
        self.assertEqual(role_run.call_count, 2)
        self.assertEqual(result["outcome"], "FAIL")

    def test_unreproduced_old_role_abstains(self):
        rows = saved_records()["replay.version_pair"]["details"]
        old = copy.deepcopy(rows["buggy"])
        old["payload"]["torch"] = "1.13.1+cpu"
        with patch.object(worker, "NEW", Path(sys.executable)), \
                patch.object(worker, "OLD", Path(sys.executable)), \
                patch.object(worker, "role_run", side_effect=[old, rows["fixed"]]) as role_run:
            result = worker.run("replay.version_pair")
        self.assertEqual(role_run.call_count, 2)
        self.assertEqual(result["outcome"], "NON_EVALUABLE")

    @unittest.skipUnless(worker.OLD.is_file() and worker.NEW.is_file(),
                         "local old/new PyTorch interpreters unavailable")
    def test_live_three_relation_plan(self):
        contract = load_contract(ROOT / "contracts/cases/pytorch_80770.json")
        risk = RiskProfile.from_dict({"mechanisms": ["derivative_error", "repeatability",
                                                    "historical_regression", "replay"]})
        plan = compile_plan(contract, risk, Budget(3, 90.0), seed=80770)
        registry = load_registry(ROOT / "contracts/cases/pytorch_80770_full_registry.json")
        report = execute_plan(plan, contract, registry, root=ROOT)
        self.assertEqual(report["status"], "CONFORMANT")
        self.assertEqual(report["reported_subject_invocations"], 5)


if __name__ == "__main__":
    unittest.main()
