"""Three-relation execution and negative controls for SciPy #8906."""

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
import scipy_8906_relation_worker as worker  # noqa: E402


def _pair_args(runner: Path | None = None, legacy: Path | None = None):
    return ("replay.version_pair", ROOT / "contracts/cases/scipy_8906.json",
            ROOT / "benchmark/cases/scipy_8906.json",
            runner or ROOT / "experiments/scipy_historical_pair_runner.py",
            legacy or ROOT / ".runtime/p4_scipy114_replay/python310/python.exe",
            Path(sys.executable))


class SciPy8906ExecutionTests(unittest.TestCase):
    def test_missing_legacy_python_abstains(self):
        result = worker.run(*_pair_args(legacy=ROOT / ".runtime/missing/python.exe"))
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["subject_invocations"], 0)

    def test_changed_frozen_runner_abstains(self):
        source = ROOT / "experiments/scipy_historical_pair_runner.py"
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "runner.py"
            altered.write_bytes(source.read_bytes() + b"\n# altered\n")
            result = worker.run(*_pair_args(runner=altered))
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["subject_invocations"], 0)

    def test_fixed_side_regression_fails_despite_role_flag(self):
        report = json.loads((ROOT / "results/reconstruction/scipy_8906_full_execution_1141_vs_1163.json")
                            .read_text(encoding="utf-8"))
        replay = next(row for row in report["observations"] if row["test_id"] == "replay.version_pair")
        old = replay["details"]["buggy"]
        new = json.loads(json.dumps(replay["details"]["fixed"]))
        new["payload"]["observation"]["conventional_u0"]["solution"] = [[1.0, 2.0, 3.0]]
        # No historical process is launched by this simulated-output control.
        with patch.object(worker, "_run_role", side_effect=[old, new]) as run_role:
            result = worker.run(*_pair_args(legacy=Path(sys.executable)))
        self.assertEqual(run_role.call_count, 2)
        self.assertEqual(result["outcome"], "FAIL")
        self.assertEqual(result["subject_invocations"], 4)

    def test_equal_but_wrong_results_are_not_an_equivalence_pass(self):
        wrong = [[1.0, 2.0, 3.0]]
        self.assertTrue(worker._close(wrong, wrong, 1e-12, 1e-12))
        self.assertFalse(worker._close(wrong, worker.EXPECTED, 1e-12, 1e-12))

    def test_saved_report_binds_all_three_relations(self):
        base = ROOT / "results/reconstruction"
        plan = json.loads((base / "scipy_8906_full_execution_plan.json").read_text(encoding="utf-8"))
        report = json.loads((base / "scipy_8906_full_execution_1141_vs_1163.json").read_text(encoding="utf-8"))
        self.assertEqual(report["plan_digest"], plan["digest"])
        self.assertEqual(report["status"], "CONFORMANT")
        self.assertEqual(report["selected_relation_count"], 3)
        self.assertEqual(report["reported_subject_invocations"], 7)
        self.assertEqual({row["test_id"]: row["outcome"] for row in report["observations"]},
                         {"replay.version_pair": "PASS", "reference.reference_solution": "PASS",
                          "property.representation_equivalence": "PASS"})
        replay = next(row for row in report["observations"] if row["test_id"] == "replay.version_pair")
        self.assertEqual(replay["details"]["buggy_version"], "1.14.1")
        self.assertEqual(replay["details"]["fixed_version"], "1.16.3")
        self.assertFalse(replay["details"]["captured_pair_file_used_as_observation"])

    @unittest.skipUnless(os.environ.get("P4_SCIPY_SITE_PACKAGES"), "live SciPy path not configured")
    def test_live_three_relation_plan(self):
        contract = load_contract(ROOT / "contracts/cases/scipy_8906.json")
        risk = RiskProfile.from_dict({"mechanisms": ["value_error", "representation_mismatch",
                                                     "historical_regression", "replay"]})
        plan = compile_plan(contract, risk, Budget(3, 20.0), seed=8906)
        registry = load_registry(ROOT / "contracts/cases/scipy_8906_full_registry.json")
        result = execute_plan(plan, contract, registry, root=ROOT)
        self.assertEqual(result["status"], "CONFORMANT")
        self.assertTrue(result["selected_relations_executed"])
        self.assertEqual(result["reported_subject_invocations"], 7)

    @unittest.skipUnless(os.environ.get("P4_SCIPY_SITE_PACKAGES"), "live SciPy path not configured")
    def test_missing_historical_python_abstains_whole_plan(self):
        contract = load_contract(ROOT / "contracts/cases/scipy_8906.json")
        risk = RiskProfile.from_dict({"mechanisms": ["value_error", "representation_mismatch",
                                                     "historical_regression", "replay"]})
        plan = compile_plan(contract, risk, Budget(3, 20.0), seed=8906)
        registry = load_registry(ROOT / "contracts/cases/scipy_8906_full_registry.json")
        command = list(registry["replay.version_pair"])
        command[command.index("--legacy-python") + 1] = str(ROOT / ".runtime/missing/python.exe")
        registry["replay.version_pair"] = tuple(command)
        result = execute_plan(plan, contract, registry, root=ROOT)
        self.assertEqual(result["status"], "ABSTAIN")
        self.assertFalse(result["selected_relations_executed"])


if __name__ == "__main__":
    unittest.main()
