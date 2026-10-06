"""Live execution and fail-closed checks for the SciPy #8928 plan."""

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
sys.path.insert(0, str(ROOT / "examples"))
import scipy_8928_pair_worker as pair_worker  # noqa: E402
from scipy_8928_pair_worker import run as run_live_pair  # noqa: E402
from scipy_8928_reference_worker import _contract_inputs  # noqa: E402


class SciPy8928ExecutionTests(unittest.TestCase):
    def test_analytic_oracle_inputs_are_bound_to_contract(self):
        cases, tolerance, contract_hash = _contract_inputs(ROOT / "contracts/cases/scipy_8928.json")
        self.assertEqual(cases, [(36.0, 1.0), (10.0, 0.1)])
        self.assertEqual(tolerance, 1e-10)
        self.assertEqual(len(contract_hash), 64)

    def test_missing_historical_interpreter_abstains(self):
        result = run_live_pair(
            ROOT / "benchmark/cases/scipy_8928.json",
            ROOT / "contracts/cases/scipy_8928.json",
            ROOT / "benchmark/runners/scipy/scipy_8928.py",
            ROOT / ".runtime/definitely_missing/python.exe",
            Path(sys.executable),
        )
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["subject_invocations"], 0)

    def test_modified_historical_runner_abstains_before_execution(self):
        source = ROOT / "benchmark/runners/scipy/scipy_8928.py"
        with tempfile.TemporaryDirectory() as directory:
            edited = Path(directory) / "edited_runner.py"
            edited.write_bytes(source.read_bytes() + b"\n# altered\n")
            result = run_live_pair(
                ROOT / "benchmark/cases/scipy_8928.json",
                ROOT / "contracts/cases/scipy_8928.json",
                edited,
                ROOT / ".runtime/p4_scipy181_replay/python310/python.exe",
                Path(sys.executable),
            )
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["subject_invocations"], 0)

    def test_postfix_regression_is_a_failure_not_a_pass(self):
        saved = json.loads((ROOT / "results/reconstruction/scipy_8928_full_execution_181_vs_1163.json")
                           .read_text(encoding="utf-8"))
        replay = next(row for row in saved["observations"] if row["test_id"] == "replay.version_pair")
        buggy = replay["details"]["buggy"]
        fixed = json.loads(json.dumps(replay["details"]["fixed"]))
        fixed["payload"]["outcome"] = "fail"
        fixed["payload"]["details"]["checks"][0]["observed"] = 0.0
        fixed["payload"]["details"]["checks"][0]["absolute_error"] = 36.0
        # No historical process is launched by this simulated-output control.
        with patch.object(pair_worker, "_role_run", side_effect=[buggy, fixed]) as role_run:
            result = run_live_pair(
                ROOT / "benchmark/cases/scipy_8928.json",
                ROOT / "contracts/cases/scipy_8928.json",
                ROOT / "benchmark/runners/scipy/scipy_8928.py",
                Path(sys.executable),
                Path(sys.executable),
            )
        self.assertEqual(role_run.call_count, 2)
        self.assertEqual(result["outcome"], "FAIL")
        self.assertEqual(result["subject_invocations"], 4)

    @unittest.skipUnless(os.environ.get("P4_SCIPY_SITE_PACKAGES"), "live SciPy path not configured")
    def test_live_two_relation_plan(self):
        contract = load_contract(ROOT / "contracts/cases/scipy_8928.json")
        profile = RiskProfile.from_dict({"mechanisms": ["value_error", "historical_regression", "replay"]})
        plan = compile_plan(contract, profile, Budget(2, 15.0), seed=8928)
        registry = load_registry(ROOT / "contracts/cases/scipy_8928_full_registry.json")
        report = execute_plan(plan, contract, registry, root=ROOT)
        self.assertEqual(report["status"], "CONFORMANT")
        self.assertEqual(report["reported_subject_invocations"], 6)
        self.assertEqual({row["test_id"]: row["outcome"] for row in report["observations"]},
                         {"reference.analytic_mean": "PASS", "replay.version_pair": "PASS"})
        replay = next(row for row in report["observations"] if row["test_id"] == "replay.version_pair")
        self.assertEqual(replay["details"]["buggy_version"], "1.8.1")
        self.assertEqual(replay["details"]["fixed_version"], "1.16.3")
        self.assertFalse(replay["details"]["captured_pair_file_used_as_observation"])

    @unittest.skipUnless(os.environ.get("P4_SCIPY_SITE_PACKAGES"), "live SciPy path not configured")
    def test_missing_historical_interpreter_abstains_whole_plan(self):
        contract = load_contract(ROOT / "contracts/cases/scipy_8928.json")
        profile = RiskProfile.from_dict({"mechanisms": ["value_error", "historical_regression", "replay"]})
        plan = compile_plan(contract, profile, Budget(2, 15.0), seed=8928)
        registry = load_registry(ROOT / "contracts/cases/scipy_8928_full_registry.json")
        command = list(registry["replay.version_pair"])
        command[command.index("--legacy-python") + 1] = str(ROOT / ".runtime/definitely_missing/python.exe")
        registry["replay.version_pair"] = tuple(command)
        report = execute_plan(plan, contract, registry, root=ROOT)
        self.assertEqual(report["status"], "ABSTAIN")
        self.assertFalse(report["selected_relations_executed"])

    def test_saved_full_execution_report_matches_plan(self):
        results = ROOT / "results/reconstruction"
        plan = json.loads((results / "scipy_8928_full_execution_plan.json").read_text(encoding="utf-8"))
        report = json.loads((results / "scipy_8928_full_execution_181_vs_1163.json").read_text(encoding="utf-8"))
        self.assertEqual(report["plan_digest"], plan["digest"])
        self.assertEqual(report["status"], "CONFORMANT")
        self.assertEqual(report["selected_relation_count"], 2)
        self.assertEqual(report["reported_subject_invocations"], 6)
        replay = next(row for row in report["observations"] if row["test_id"] == "replay.version_pair")
        self.assertEqual(replay["details"]["buggy_max_absolute_error"], 35.99999999966363)
        self.assertLess(replay["details"]["fixed_max_absolute_error"], 1e-10)


if __name__ == "__main__":
    unittest.main()
