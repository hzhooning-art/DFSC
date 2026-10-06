"""Exercise the installed P4 testing CLI using a dependency-free example.

The positive, negative, and tamper checks are engineering smoke tests. They
do not add historical defect families or estimate field effectiveness.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "contracts" / "examples"
CONTRACT = EXAMPLES / "expm1_execution_demo.json"
PROFILE = EXAMPLES / "expm1_execution_risk.json"
REGISTRY = EXAMPLES / "expm1_execution_registry.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_cli(*args: str, expected_code: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "p4_testing", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != expected_code:
        raise AssertionError(
            f"CLI exited {completed.returncode}, expected {expected_code}: "
            f"{completed.stdout}\n{completed.stderr}"
        )
    return completed


def assert_report(path: Path, expected: str, first_outcome: str) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report["schema"] == "P4-Plan-Execution-v1", "wrong report schema")
    require(report["status"] == expected, f"wrong report status: {report['status']}")
    require(report["selected_relations_executed"] is True, "incomplete execution")
    require(report["selected_relation_count"] == 2, "unexpected relation count")
    require(report["reported_subject_invocations"] > 0, "no reported subject calls")
    require([row["outcome"] for row in report["observations"]] == [first_outcome, "PASS"],
            "unexpected relation outcomes")
    require(all(row["stdout_sha256"] and row["command_file_sha256"] for row in report["observations"]),
            "missing raw evidence hashes")
    require(bool(report["contract_file_sha256"] and report["registry_file_sha256"]),
            "missing input hashes")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="p4-software-smoke-") as directory:
        work = Path(directory)
        plan = work / "plan.json"
        positive = work / "positive.json"
        negative = work / "negative.json"
        bad_registry = work / "biased-registry.json"
        tampered_plan = work / "tampered-plan.json"

        run_cli(
            "compile", "--contract", str(CONTRACT), "--risk-profile", str(PROFILE),
            "--seed", "17", "--max-invocations", "2", "--wall-seconds", "10",
            "--output", str(plan),
        )
        run_cli(
            "execute", "--contract", str(CONTRACT), "--plan", str(plan),
            "--registry", str(REGISTRY), "--artifact-root", str(ROOT),
            "--output", str(positive),
        )
        assert_report(positive, "CONFORMANT", "PASS")

        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        registry["commands"]["reference.decimal_value"].extend(["--bias", "0.01"])
        bad_registry.write_text(json.dumps(registry), encoding="utf-8")
        run_cli(
            "execute", "--contract", str(CONTRACT), "--plan", str(plan),
            "--registry", str(bad_registry), "--artifact-root", str(ROOT),
            "--output", str(negative),
        )
        assert_report(negative, "NONCONFORMANT", "FAIL")

        changed = json.loads(plan.read_text(encoding="utf-8"))
        changed["seed"] += 1
        tampered_plan.write_text(json.dumps(changed), encoding="utf-8")
        refused = run_cli(
            "execute", "--contract", str(CONTRACT), "--plan", str(tampered_plan),
            "--registry", str(REGISTRY), "--artifact-root", str(ROOT),
            "--output", str(work / "refused.json"),
            expected_code=2,
        )
        require("digest mismatch" in refused.stderr, "tampered plan not explicitly rejected")
        require(not (work / "refused.json").exists(), "tampered plan wrote a report")

    print("P4 installed-CLI smoke: positive, negative control, and tamper refusal passed")


if __name__ == "__main__":
    main()
