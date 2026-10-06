"""Fail-closed validation of the predeclared PyTorch #91631 pair capture."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkError


def assess_91631_observation(observation: dict[str, Any], role: dict[str, Any]) -> list[str]:
    """Return reasons for a non-evaluable or non-reproduced role observation."""
    problems = []
    if observation.get("schema") != "P4-Role-Independent-Runner-v1" or (
        observation.get("case_id") != "pytorch_91631_extension"
    ) or observation.get("package") != "torch":
        problems.append("runner identity differs")
    if observation.get("version") != role["expected_torch_version"]:
        problems.append("torch version differs")
    if observation.get("device") != "cpu" or observation.get("dtype") != "complex128":
        problems.append("device/dtype differs")
    cases = observation.get("cases")
    if not isinstance(cases, list) or len(cases) != 3 or observation.get("subject_invocations") != 3:
        return problems + ["case inventory differs"]
    expected_identity = [("csr", False), ("csr", True), ("csc", True)]
    if [(item.get("layout"), item.get("complex_values")) for item in cases] != expected_identity:
        problems.append("case order or identity differs")
    if cases[0].get("status") != "pass":
        problems.append("real CSR control did not pass")
    for index, (case, expected_status) in enumerate(zip(cases[1:], role["expected_complex_statuses"]), 1):
        if case.get("status") != expected_status:
            problems.append(f"complex case {index} status differs")
        if expected_status == "exception":
            if case.get("error_type") != "RuntimeError" or (
                role["expected_exception_fragment"] not in case.get("error_message", "")
            ):
                problems.append(f"complex case {index} exception differs")
        else:
            if case.get("observed_real") != case.get("expected_real") or (
                case.get("observed_imag") != case.get("expected_imag")
            ):
                problems.append(f"complex case {index} dense oracle differs")
    if observation.get("outcome") != ("pass" if all(c["status"] == "pass" for c in cases) else "fail"):
        problems.append("overall outcome differs from cases")
    return problems


def verify_91631_capture(
    zip_path: Path, lock_path: Path, prereg_path: Path, runner_path: Path
) -> dict[str, Any]:
    """Verify artifact hashes and re-evaluate all four raw role observations."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    prereg_bytes = prereg_path.read_bytes()
    prereg = json.loads(prereg_bytes)
    if prereg.get("schema") != "P4-PyTorch-91631-Pre-Replay-v1" or (
        prereg.get("runner_sha256") != hashlib.sha256(runner_path.read_bytes()).hexdigest()
    ):
        raise BenchmarkError("PyTorch #91631 preregistration/runner binding differs")
    if lock.get("schema") != "P4-PyTorch-91631-Pair-Capture-Lock-v1" or (
        lock.get("preregistration_sha256") != hashlib.sha256(prereg_bytes).hexdigest()
    ) or lock.get("zip_sha256") != hashlib.sha256(zip_path.read_bytes()).hexdigest():
        raise BenchmarkError("PyTorch #91631 capture lock hash differs")
    roles = {item["name"]: item for item in prereg["roles"]}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise BenchmarkError("PyTorch #91631 capture ZIP damaged or duplicated")
        manifest = json.loads(archive.read("manifest.json"))
        rows = manifest.get("runs")
        if manifest.get("schema") != "P4-PyTorch-91631-Pair-Capture-v1" or (
            manifest.get("preregistration_sha256") != lock["preregistration_sha256"]
        ) or not isinstance(rows, list) or len(rows) != 4 or (
            [row.get("role") for row in rows] != prereg["role_order"]
        ):
            raise BenchmarkError("PyTorch #91631 capture inventory differs")
        expected_names = {"manifest.json"}
        for row in rows:
            for kind in ("stdout", "stderr"):
                path = row[f"{kind}_path"]
                expected_names.add(path)
                raw = archive.read(path)
                if hashlib.sha256(raw).hexdigest() != row[f"{kind}_sha256"]:
                    raise BenchmarkError("PyTorch #91631 raw process output hash differs")
            problems = []
            if row.get("timed_out") or row.get("returncode") != 0:
                problems.append("process timeout or nonzero exit")
            try:
                observation = json.loads(archive.read(row["stdout_path"]))
                problems.extend(assess_91631_observation(observation, roles[row["role"]]))
            except (UnicodeDecodeError, ValueError, TypeError, KeyError):
                problems.append("runner stdout is not valid observation JSON")
            if problems != row.get("problems"):
                raise BenchmarkError("PyTorch #91631 recorded assessment differs from raw output")
        if set(names) != expected_names or lock.get("run_count") != len(rows):
            raise BenchmarkError("PyTorch #91631 capture ZIP file inventory differs")
        verdict = "REPRODUCED_LOCAL_PAIR" if all(not row["problems"] for row in rows) else (
            "NON_EVALUABLE_OR_NONREPRODUCED"
        )
        if manifest.get("verdict") != verdict or lock.get("verdict") != verdict or (
            manifest.get("historical_defect_family_admitted") is not False
        ):
            raise BenchmarkError("PyTorch #91631 capture verdict or admission flag differs")
    return manifest
