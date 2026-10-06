"""Independent-oracle and optional live-wheel tests for SciPy #4130."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from p4_testing.compiler import Budget, compile_plan
from p4_testing.contracts import RiskProfile, load_contract
from p4_testing.execution import execute_plan, load_registry


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from scipy_4130_reference_worker import exact_two_sided_pvalue, _table_and_tolerance  # noqa: E402
from scipy_4130_pair_worker import run as run_live_pair  # noqa: E402


class SciPy4130ExecutionTests(unittest.TestCase):
    def test_integer_reference_matches_frozen_value(self):
        table, tolerance, contract_hash = _table_and_tolerance(ROOT / "contracts/cases/scipy_4130.json")
        self.assertEqual(table, ((345, 455), (260, 345)))
        self.assertAlmostEqual(exact_two_sided_pvalue(table), 0.9566778639926435, places=14)
        self.assertEqual(tolerance, 1e-12)
        self.assertEqual(len(contract_hash), 64)

    def test_saved_report_is_current_side_only_and_bound_to_plan(self):
        results = ROOT / "results/reconstruction"
        plan = json.loads((results / "scipy_4130_reference_execution_plan.json").read_text(encoding="utf-8"))
        report = json.loads((results / "scipy_4130_reference_execution_scipy_1163.json").read_text(encoding="utf-8"))
        self.assertEqual(report["plan_digest"], plan["digest"])
        self.assertEqual(report["contract_file_sha256"], report["observations"][0]["details"]["contract_sha256"])
        self.assertEqual(report["status"], "CONFORMANT")
        self.assertTrue(report["selected_relations_executed"])
        self.assertEqual(report["selected_relation_count"], 1)
        self.assertEqual(report["observations"][0]["details"]["scipy_version"], "1.16.3")
        self.assertFalse(report["observations"][0]["details"]["historical_pair_executed"])

    def test_saved_full_plan_has_two_newly_executed_relations(self):
        results = ROOT / "results/reconstruction"
        plan = json.loads((results / "scipy_4130_full_execution_plan.json").read_text(encoding="utf-8"))
        report = json.loads((results / "scipy_4130_full_execution_181_vs_1163.json").read_text(encoding="utf-8"))
        self.assertEqual(report["plan_digest"], plan["digest"])
        self.assertEqual(report["status"], "CONFORMANT")
        self.assertTrue(report["selected_relations_executed"])
        self.assertEqual(report["selected_relation_count"], 2)
        self.assertEqual({row["test_id"]: row["outcome"] for row in report["observations"]},
                         {"reference.exact_enumeration": "PASS", "replay.version_pair": "PASS"})
        replay = next(row for row in report["observations"] if row["test_id"] == "replay.version_pair")
        self.assertEqual(replay["details"]["buggy_version"], "1.8.1")
        self.assertEqual(replay["details"]["fixed_version"], "1.16.3")
        self.assertFalse(replay["details"]["captured_pair_file_used_as_observation"])
        self.assertEqual(replay["details"]["buggy"]["status"], "OBSERVED")
        self.assertEqual(replay["details"]["fixed"]["status"], "OBSERVED")

    def test_missing_legacy_interpreter_abstains(self):
        result = run_live_pair(
            ROOT / "benchmark/cases/scipy_4130.json",
            ROOT / "contracts/cases/scipy_4130.json",
            ROOT / "benchmark/runners/scipy/scipy_4130.py",
            ROOT / ".runtime/definitely_missing/python.exe",
            Path(sys.executable),
        )
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["subject_invocations"], 0)
        self.assertEqual(result["details"]["reason"], "INVALID_OR_MISSING_REPLAY_INPUT")

    def test_edited_historical_runner_abstains_before_execution(self):
        source = ROOT / "benchmark/runners/scipy/scipy_4130.py"
        with tempfile.TemporaryDirectory() as directory:
            edited = Path(directory) / "edited_runner.py"
            edited.write_bytes(source.read_bytes() + b"\n# edited\n")
            result = run_live_pair(
                ROOT / "benchmark/cases/scipy_4130.json",
                ROOT / "contracts/cases/scipy_4130.json",
                edited,
                ROOT / ".runtime/p4_scipy181_replay/python310/python.exe",
                Path(sys.executable),
            )
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["subject_invocations"], 0)
        self.assertIn("SHA-256 mismatch", result["details"]["error"])

    @unittest.skipUnless(os.environ.get("P4_SCIPY_SITE_PACKAGES"), "live SciPy path not configured")
    def test_live_installed_scipy_reference_plan(self):
        contract = load_contract(ROOT / "contracts/cases/scipy_4130.json")
        risk = RiskProfile.from_dict({"mechanisms": ["value_error"]})
        plan = compile_plan(contract, risk, Budget(1, 10.0), seed=4130)
        registry = load_registry(ROOT / "contracts/cases/scipy_4130_reference_registry.json")
        report = execute_plan(plan, contract, registry, root=ROOT)
        self.assertEqual(report["status"], "CONFORMANT")
        self.assertEqual(report["reported_subject_invocations"], 1)
        self.assertTrue(report["observations"][0]["details"]["scipy_version"])
        self.assertLessEqual(report["observations"][0]["details"]["absolute_pvalue_error"], 1e-12)
        self.assertFalse(report["observations"][0]["details"]["historical_pair_executed"])

    @unittest.skipUnless(os.environ.get("P4_SCIPY_SITE_PACKAGES"), "live SciPy path not configured")
    def test_live_two_relation_plan(self):
        contract = load_contract(ROOT / "contracts/cases/scipy_4130.json")
        risk = RiskProfile.from_dict({"mechanisms": ["value_error", "historical_regression", "replay"]})
        plan = compile_plan(contract, risk, Budget(2, 15.0), seed=4130)
        registry = load_registry(ROOT / "contracts/cases/scipy_4130_full_registry.json")
        report = execute_plan(plan, contract, registry, root=ROOT)
        self.assertEqual(report["status"], "CONFORMANT")
        self.assertTrue(report["selected_relations_executed"])
        self.assertEqual(report["reported_subject_invocations"], 3)
        replay = next(row for row in report["observations"] if row["test_id"] == "replay.version_pair")
        self.assertEqual(replay["details"]["buggy_version"], "1.8.1")
        self.assertNotEqual(replay["details"]["fixed_version"], "1.8.1")

    @unittest.skipUnless(os.environ.get("P4_SCIPY_SITE_PACKAGES"), "live SciPy path not configured")
    def test_full_plan_abstains_if_historical_interpreter_is_missing(self):
        contract = load_contract(ROOT / "contracts/cases/scipy_4130.json")
        risk = RiskProfile.from_dict({"mechanisms": ["value_error", "historical_regression", "replay"]})
        plan = compile_plan(contract, risk, Budget(2, 15.0), seed=4130)
        registry = load_registry(ROOT / "contracts/cases/scipy_4130_full_registry.json")
        command = list(registry["replay.version_pair"])
        command[command.index("--legacy-python") + 1] = str(ROOT / ".runtime/definitely_missing/python.exe")
        registry["replay.version_pair"] = tuple(command)
        report = execute_plan(plan, contract, registry, root=ROOT)
        self.assertEqual(report["status"], "ABSTAIN")
        self.assertFalse(report["selected_relations_executed"])
        self.assertEqual({row["test_id"]: row["outcome"] for row in report["observations"]},
                         {"reference.exact_enumeration": "PASS", "replay.version_pair": "ABSTAIN"})


if __name__ == "__main__":
    unittest.main()
