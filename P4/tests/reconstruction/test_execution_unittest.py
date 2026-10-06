"""Stdlib-only executable checks for the first full-plan execution slice."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from p4_testing.cli import main
from p4_testing.compiler import Budget, compile_plan
from p4_testing.contracts import RiskProfile, load_contract, parse_contract
from p4_testing.execution import execute_plan, load_registry


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/examples/expm1_execution_demo.json"
REGISTRY = ROOT / "contracts/examples/expm1_execution_registry.json"
PROFILE = RiskProfile.from_dict({"mechanisms": ["value_error", "derivative_error"]})


def make_plan(wall_seconds: float = 5.0):
    return compile_plan(load_contract(CONTRACT), PROFILE, Budget(2, wall_seconds), seed=17)


class ExecutionTests(unittest.TestCase):
    def test_complete_real_execution_passes(self):
        plan = make_plan()
        report = execute_plan(plan, load_contract(CONTRACT), load_registry(REGISTRY), root=ROOT)
        self.assertEqual(report["status"], "CONFORMANT")
        self.assertTrue(report["selected_relations_executed"])
        self.assertEqual([row["outcome"] for row in report["observations"]], ["PASS", "PASS"])
        self.assertEqual(report["reported_subject_invocations"], 6)
        self.assertTrue(all(row["stdout_sha256"] and row["command_file_sha256"]
                            for row in report["observations"]))

    def test_observed_numeric_failure_is_nonconformant(self):
        registry = load_registry(REGISTRY)
        registry["reference.decimal_value"] += ("--bias", "0.01")
        report = execute_plan(make_plan(), load_contract(CONTRACT), registry, root=ROOT)
        self.assertEqual(report["status"], "NONCONFORMANT")
        self.assertTrue(report["selected_relations_executed"])
        self.assertEqual(report["observations"][0]["outcome"], "FAIL")

    def test_missing_executor_cannot_pass(self):
        registry = load_registry(REGISTRY)
        del registry["property.central_derivative"]
        report = execute_plan(make_plan(), load_contract(CONTRACT), registry, root=ROOT)
        self.assertEqual(report["status"], "ABSTAIN")
        self.assertFalse(report["selected_relations_executed"])
        self.assertEqual(report["observations"][1]["reason"], "MISSING_EXECUTOR")

    def test_invalid_worker_json_cannot_pass(self):
        registry = load_registry(REGISTRY)
        registry["reference.decimal_value"] = (sys.executable, "-c", "print('not-json')")
        report = execute_plan(make_plan(), load_contract(CONTRACT), registry, root=ROOT)
        self.assertEqual(report["status"], "ABSTAIN")
        self.assertEqual(report["observations"][0]["reason"], "INVALID_JSON")

    def test_zero_call_pass_and_wrong_relation_are_refused(self):
        registry = load_registry(REGISTRY)
        payload = {"schema": "P4-Relation-Observation-v1", "test_id": "reference.decimal_value",
                   "outcome": "PASS", "subject_invocations": 0, "details": {}}
        registry["reference.decimal_value"] = (sys.executable, "-c", f"print({json.dumps(json.dumps(payload))})")
        report = execute_plan(make_plan(), load_contract(CONTRACT), registry, root=ROOT)
        self.assertEqual(report["status"], "ABSTAIN")
        self.assertEqual(report["observations"][0]["reason"], "NO_SUBJECT_EXECUTION")
        payload["subject_invocations"] = 1
        payload["test_id"] = "property.central_derivative"
        registry["reference.decimal_value"] = (sys.executable, "-c", f"print({json.dumps(json.dumps(payload))})")
        report = execute_plan(make_plan(), load_contract(CONTRACT), registry, root=ROOT)
        self.assertEqual(report["observations"][0]["reason"], "OBSERVATION_BINDING_MISMATCH")

    def test_abstaining_plan_never_runs_commands(self):
        plan = compile_plan(load_contract(CONTRACT), PROFILE, Budget(1, 5.0), seed=17)
        report = execute_plan(plan, load_contract(CONTRACT), load_registry(REGISTRY), root=ROOT)
        self.assertEqual(report["status"], "ABSTAIN")
        self.assertEqual(report["observations"], [])
        self.assertFalse(report["selected_relations_executed"])

    def test_timeout_cannot_pass(self):
        registry = load_registry(REGISTRY)
        registry["reference.decimal_value"] = (sys.executable, "-c", "import time; time.sleep(2)")
        report = execute_plan(make_plan(0.05), load_contract(CONTRACT), registry, root=ROOT)
        self.assertEqual(report["status"], "ABSTAIN")
        self.assertEqual(report["observations"][0]["reason"], "TIMEOUT")

    def test_changed_contract_invalidates_plan(self):
        payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
        payload["relations"]["decimal_value"]["mechanisms"] = ["repeatability"]
        with self.assertRaisesRegex(ValueError, "plan does not match"):
            execute_plan(make_plan(), parse_contract(payload), load_registry(REGISTRY), root=ROOT)

    def test_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            report_path = Path(directory) / "report.json"
            self.assertEqual(main(["compile", "--contract", str(CONTRACT),
                                   "--risk-profile", str(ROOT / "contracts/examples/expm1_execution_risk.json"),
                                   "--seed", "17", "--max-invocations", "2",
                                   "--wall-seconds", "5", "--output", str(plan_path)]), 0)
            self.assertEqual(main(["execute", "--contract", str(CONTRACT),
                                   "--plan", str(plan_path), "--registry", str(REGISTRY),
                                   "--artifact-root", str(ROOT),
                                   "--output", str(report_path)]), 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "CONFORMANT")
            self.assertEqual(len(report["contract_file_sha256"]), 64)
            self.assertEqual(len(report["registry_file_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
