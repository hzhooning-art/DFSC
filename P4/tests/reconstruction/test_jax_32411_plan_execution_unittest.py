"""Cross-project JAX #32411 live plan and negative controls."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p4_testing.compiler import Budget, compile_plan
from p4_testing.contracts import RiskProfile, load_contract
from p4_testing.execution import execute_plan, load_registry


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / ".runtime/p4_jax072/Scripts/python.exe"
NEW = ROOT / ".runtime/p4_jax080/Scripts/python.exe"
sys.path.insert(0, str(ROOT / "examples"))
import jax_32411_relation_worker as worker  # noqa: E402


def _args(old: Path = OLD, runner: Path | None = None, current: Path = NEW):
    return (worker.REPLAY_ID, ROOT / "contracts/cases/jax_32411.json",
            ROOT / "benchmark/cases/jax_32411.json",
            runner or ROOT / "benchmark/runners/jax/jax_32411.py",
            current, old)


class Jax32411ExecutionTests(unittest.TestCase):
    def test_missing_old_python_abstains(self):
        result = worker.run(*_args(old=ROOT / ".runtime/missing/python.exe"))
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["subject_invocations"], 0)

    def test_changed_frozen_runner_abstains(self):
        source = ROOT / "benchmark/runners/jax/jax_32411.py"
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "runner.py"
            altered.write_bytes(source.read_bytes() + b"\n# modified\n")
            result = worker.run(*_args(runner=altered))
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["subject_invocations"], 0)

    def test_new_gradient_regression_fails_even_if_runner_says_pass(self):
        report = json.loads((ROOT / "results/reconstruction/jax_32411_full_execution_072_vs_080.json")
                            .read_text(encoding="utf-8"))
        replay = next(row for row in report["observations"] if row["test_id"] == worker.REPLAY_ID)
        old = replay["details"]["buggy"]
        fixed = json.loads(json.dumps(replay["details"]["fixed"]))
        fixed["payload"]["details"]["max_absolute_gradient_error"] = 0.01
        # The role process is mocked; only input-file validation uses this Python.
        with patch.object(worker, "_role_run", side_effect=[old, fixed]) as role_run:
            result = worker.run(*_args(old=Path(sys.executable), current=Path(sys.executable)))
        self.assertEqual(role_run.call_count, 2)
        self.assertEqual(result["outcome"], "FAIL")
        self.assertEqual(result["subject_invocations"], 4)

    def test_unreproduced_old_gradient_abstains(self):
        report = json.loads((ROOT / "results/reconstruction/jax_32411_full_execution_072_vs_080.json")
                            .read_text(encoding="utf-8"))
        replay = next(row for row in report["observations"] if row["test_id"] == worker.REPLAY_ID)
        old = json.loads(json.dumps(replay["details"]["buggy"]))
        old["payload"]["details"]["max_absolute_gradient_error"] = 0.0
        fixed = replay["details"]["fixed"]
        with patch.object(worker, "_role_run", side_effect=[old, fixed]) as role_run:
            result = worker.run(*_args(old=Path(sys.executable), current=Path(sys.executable)))
        self.assertEqual(role_run.call_count, 2)
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["details"]["reason"], "HISTORICAL_ROLE_MISMATCH")

    def test_saved_report_binds_both_selected_relations(self):
        base = ROOT / "results/reconstruction"
        plan = json.loads((base / "jax_32411_full_execution_plan.json").read_text(encoding="utf-8"))
        report = json.loads((base / "jax_32411_full_execution_072_vs_080.json").read_text(encoding="utf-8"))
        self.assertEqual(report["plan_digest"], plan["digest"])
        self.assertEqual(report["status"], "CONFORMANT")
        self.assertEqual(report["selected_relation_count"], 2)
        self.assertEqual(report["reported_subject_invocations"], 6)
        self.assertEqual({row["test_id"]: row["outcome"] for row in report["observations"]},
                         {worker.PROPERTY_ID: "PASS", worker.REPLAY_ID: "PASS"})
        replay = next(row for row in report["observations"] if row["test_id"] == worker.REPLAY_ID)
        self.assertEqual((replay["details"]["buggy_version"], replay["details"]["fixed_version"]),
                         ("0.7.2", "0.8.0"))
        self.assertFalse(replay["details"]["captured_pair_file_used_as_observation"])

    @unittest.skipUnless(OLD.is_file() and NEW.is_file(), "local JAX wheel environments not configured")
    def test_live_two_relation_plan(self):
        contract = load_contract(ROOT / "contracts/cases/jax_32411.json")
        risk = RiskProfile.from_dict({"mechanisms": ["derivative_error", "historical_regression", "replay"]})
        plan = compile_plan(contract, risk, Budget(2, 30.0), seed=32411)
        registry = load_registry(ROOT / "contracts/cases/jax_32411_full_registry.json")
        result = execute_plan(plan, contract, registry, root=ROOT)
        self.assertEqual(result["status"], "CONFORMANT")
        self.assertTrue(result["selected_relations_executed"])
        self.assertEqual(result["reported_subject_invocations"], 6)

    @unittest.skipUnless(NEW.is_file(), "local fixed JAX wheel environment not configured")
    def test_missing_old_python_abstains_whole_plan(self):
        contract = load_contract(ROOT / "contracts/cases/jax_32411.json")
        risk = RiskProfile.from_dict({"mechanisms": ["derivative_error", "historical_regression", "replay"]})
        plan = compile_plan(contract, risk, Budget(2, 30.0), seed=32411)
        registry = load_registry(ROOT / "contracts/cases/jax_32411_full_registry.json")
        command = list(registry[worker.REPLAY_ID])
        command[command.index("--legacy-python") + 1] = str(ROOT / ".runtime/missing/python.exe")
        registry[worker.REPLAY_ID] = tuple(command)
        result = execute_plan(plan, contract, registry, root=ROOT)
        self.assertEqual(result["status"], "ABSTAIN")
        self.assertFalse(result["selected_relations_executed"])


if __name__ == "__main__":
    unittest.main()
