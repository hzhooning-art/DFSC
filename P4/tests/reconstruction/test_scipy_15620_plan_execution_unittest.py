"""Live and fail-closed checks for SciPy #15620's three relations."""

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
import scipy_15620_relation_worker as worker  # noqa: E402


def _paths(runner: Path | None = None, legacy: Path | None = None,
           fixture: Path | None = None):
    return ("replay.version_pair", ROOT / "contracts/cases/scipy_15620.json",
            fixture or ROOT / "contracts/cases/scipy_15620_reference_fixture.json",
            ROOT / "benchmark/cases/scipy_15620.json",
            runner or ROOT / "experiments/scipy_resample_poly_historical_pair_runner.py",
            legacy or ROOT / ".runtime/p4_scipy114_replay/python310/python.exe",
            Path(sys.executable))


class SciPy15620ExecutionTests(unittest.TestCase):
    def test_recorded_reference_is_explicitly_not_independent_oracle(self):
        fixture = json.loads((ROOT / "contracts/cases/scipy_15620_reference_fixture.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(len(fixture["output"]), 14)
        self.assertIn("not a mathematically independent oracle", fixture["provenance"])

    def test_missing_old_python_abstains(self):
        result = worker.run(*_paths(legacy=ROOT / ".runtime/missing/python.exe"))
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["subject_invocations"], 0)

    def test_changed_runner_abstains(self):
        source = ROOT / "experiments/scipy_resample_poly_historical_pair_runner.py"
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "runner.py"
            altered.write_bytes(source.read_bytes() + b"\n# modified\n")
            result = worker.run(*_paths(runner=altered))
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["subject_invocations"], 0)

    def test_corrupted_reference_cannot_confirm_historical_pair(self):
        source = ROOT / "contracts/cases/scipy_15620_reference_fixture.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["output"][3] = 500.0
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "fixture.json"
            altered.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(worker, "_run_role") as run_role:
                report = json.loads((ROOT / "results/reconstruction/scipy_15620_full_execution_1141_vs_1163.json")
                                    .read_text(encoding="utf-8"))
                replay = next(row for row in report["observations"] if row["test_id"] == "replay.version_pair")
                run_role.side_effect = [replay["details"]["buggy"], replay["details"]["fixed"]]
                # The process is mocked; use an existing interpreter for input validation.
                result = worker.run(*_paths(fixture=altered, legacy=Path(sys.executable)))
                self.assertEqual(run_role.call_count, 2)
        self.assertEqual(result["outcome"], "NON_EVALUABLE")
        self.assertEqual(result["details"]["reason"], "BUGGY_ROLE_NOT_REPRODUCED")

    def test_fixed_side_integer_zeros_fail_even_with_true_flag(self):
        report = json.loads((ROOT / "results/reconstruction/scipy_15620_full_execution_1141_vs_1163.json")
                            .read_text(encoding="utf-8"))
        replay = next(row for row in report["observations"] if row["test_id"] == "replay.version_pair")
        buggy = replay["details"]["buggy"]
        fixed = json.loads(json.dumps(replay["details"]["fixed"]))
        for dtype in ("int16", "int32"):
            item = fixed["payload"]["observation"]["by_dtype"][dtype]
            item["output"] = [0.0] * 14
            item["nonzero_count"] = 0
            item["max_abs"] = 0.0
        with patch.object(worker, "_run_role", side_effect=[buggy, fixed]) as run_role:
            result = worker.run(*_paths(legacy=Path(sys.executable)))
        self.assertEqual(run_role.call_count, 2)
        self.assertEqual(result["outcome"], "FAIL")
        self.assertEqual(result["subject_invocations"], 8)

    def test_saved_plan_has_three_real_passes(self):
        base = ROOT / "results/reconstruction"
        plan = json.loads((base / "scipy_15620_full_execution_plan.json").read_text(encoding="utf-8"))
        report = json.loads((base / "scipy_15620_full_execution_1141_vs_1163.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(report["plan_digest"], plan["digest"])
        self.assertEqual(report["status"], "CONFORMANT")
        self.assertEqual(report["selected_relation_count"], 3)
        self.assertEqual(report["reported_subject_invocations"], 12)
        self.assertEqual({row["test_id"]: row["outcome"] for row in report["observations"]},
                         {"reference.float64_reference": "PASS",
                          "property.dtype_consistency": "PASS", "replay.version_pair": "PASS"})
        replay = next(row for row in report["observations"] if row["test_id"] == "replay.version_pair")
        self.assertEqual((replay["details"]["buggy_version"], replay["details"]["fixed_version"]),
                         ("1.14.1", "1.16.3"))
        self.assertFalse(replay["details"]["captured_pair_file_used_as_observation"])
        self.assertFalse(replay["details"]["recorded_reference_is_independent_oracle"])

    @unittest.skipUnless(os.environ.get("P4_SCIPY_SITE_PACKAGES"), "live SciPy path not configured")
    def test_live_three_relation_plan(self):
        contract = load_contract(ROOT / "contracts/cases/scipy_15620.json")
        risk = RiskProfile.from_dict({"mechanisms": ["value_error", "dtype_device_mismatch",
                                                     "historical_regression", "replay"]})
        plan = compile_plan(contract, risk, Budget(3, 25.0), seed=15620)
        registry = load_registry(ROOT / "contracts/cases/scipy_15620_full_registry.json")
        result = execute_plan(plan, contract, registry, root=ROOT)
        self.assertEqual(result["status"], "CONFORMANT")
        self.assertTrue(result["selected_relations_executed"])
        self.assertEqual(result["reported_subject_invocations"], 12)

    @unittest.skipUnless(os.environ.get("P4_SCIPY_SITE_PACKAGES"), "live SciPy path not configured")
    def test_missing_old_python_abstains_whole_plan(self):
        contract = load_contract(ROOT / "contracts/cases/scipy_15620.json")
        risk = RiskProfile.from_dict({"mechanisms": ["value_error", "dtype_device_mismatch",
                                                     "historical_regression", "replay"]})
        plan = compile_plan(contract, risk, Budget(3, 25.0), seed=15620)
        registry = load_registry(ROOT / "contracts/cases/scipy_15620_full_registry.json")
        command = list(registry["replay.version_pair"])
        command[command.index("--legacy-python") + 1] = str(ROOT / ".runtime/missing/python.exe")
        registry["replay.version_pair"] = tuple(command)
        result = execute_plan(plan, contract, registry, root=ROOT)
        self.assertEqual(result["status"], "ABSTAIN")
        self.assertFalse(result["selected_relations_executed"])


if __name__ == "__main__":
    unittest.main()
