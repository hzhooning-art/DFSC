"""Compare fixed workers under explicit workflow policies, not testing frameworks.

The strict scripted comparator deliberately implements comparable safeguards.
Agreement with P4 is a valid result. All scenarios are disclosed controls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("normal", "numerical_bias", "missing_worker", "invalid_json",
             "wrong_id", "zero_calls", "non_evaluable", "modified_plan", "changed_contract")
EXPECTED = ("CONFORMANT", "NONCONFORMANT", "ABSTAIN", "ABSTAIN", "ABSTAIN",
            "ABSTAIN", "ABSTAIN", "REFUSED", "REFUSED")


def save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_plan_check(plan: dict, contract: dict) -> None:
    """Independent reference checks for the fixed, complete two-relation plan."""
    payload = {key: value for key, value in plan.items() if key != "digest"}
    actual = digest(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False).encode())
    if actual != plan["digest"] or plan["contract_id"] != contract["contract_id"]:
        raise ValueError("plan integrity mismatch")
    required = {"value_error", "derivative_error"}
    if set(plan["risk_mechanisms"]) != required or plan["status"] != "READY":
        raise ValueError("unexpected risk profile or plan state")
    expected_ids = {value["layer"] + "." + key for key, value in contract["relations"].items()}
    actual_ids = [test["test_id"] for test in plan["tests"]]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise ValueError("incomplete selected relations")
    covered = set()
    for test in plan["tests"]:
        relation_id = test["test_id"].split(".", 1)[1]
        relation = contract["relations"][relation_id]
        seed_bytes = f"{plan['seed']}\0{contract['contract_id']}\0{relation_id}".encode()
        expected_parameters = {**relation["parameters"], "generator": relation["generator"],
                               "comparator": relation["comparator"],
                               "seed": int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")}
        if test["parameters"] != expected_parameters or test["layer"] != relation["layer"]:
            raise ValueError("contract declaration mismatch")
        covered.update(relation["mechanisms"])
    certificate = plan["coverage_certificate"]
    if not required <= covered or certificate["missing"] or set(certificate["covered"]) != required:
        raise ValueError("missing declared coverage")
    if len(actual_ids) > plan["max_invocations"]:
        raise ValueError("relation-slot budget exceeded")


def direct_workflow(plan: dict, contract: dict, registry: dict, *, strict: bool) -> dict:
    rows = []
    if strict:
        try:
            strict_plan_check(plan, contract)
        except (ValueError, KeyError, TypeError) as error:
            return {"status": "REFUSED", "reason": str(error), "observations": rows}
    deadline = time.perf_counter() + plan["wall_seconds"]
    for test in plan["tests"]:
        test_id = test["test_id"]
        command = registry["commands"].get(test_id)
        row = {"test_id": test_id, "outcome": "ABSTAIN"}
        if command is None:
            # The exit-only control has no declared completeness check.
            row["reason"] = "missing command"
            rows.append(row)
            continue
        argv = [arg.replace("{python}", sys.executable).replace("{p4_root}", str(ROOT))
                .replace("{test_id}", test_id) for arg in command]
        row["command_argv"] = argv
        try:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, plan["wall_seconds"])
            completed = subprocess.run(argv, cwd=ROOT, capture_output=True,
                                       timeout=remaining, check=False)
            row.update(exit_code=completed.returncode, stdout=completed.stdout.decode("utf-8", "replace"),
                       stderr=completed.stderr.decode("utf-8", "replace"))
            if strict:
                row.update(stdout_sha256=digest(completed.stdout), stderr_sha256=digest(completed.stderr),
                           command_file_sha256={str(Path(arg).resolve()): digest(Path(arg).read_bytes())
                                                for arg in argv if Path(arg).is_file()})
            if completed.returncode != 0:
                row["reason"] = "process error"
            elif not strict:
                row["outcome"] = "PASS"
            else:
                observation = json.loads(row["stdout"], parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                calls = observation.get("subject_invocations")
                outcome = observation.get("outcome")
                if (observation.get("schema") != "P4-Relation-Observation-v1"
                        or observation.get("test_id") != test_id
                        or outcome not in {"PASS", "FAIL", "NON_EVALUABLE"}
                        or not isinstance(calls, int) or isinstance(calls, bool) or calls < 0
                        or not isinstance(observation.get("details"), dict)
                        or (outcome in {"PASS", "FAIL"} and calls == 0)):
                    raise ValueError("invalid observation")
                json.dumps(observation["details"], allow_nan=False)
                row["outcome"] = outcome if outcome in {"PASS", "FAIL"} else "ABSTAIN"
                row["subject_invocations"] = calls
                row["details"] = observation["details"]
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, AttributeError) as error:
            row["reason"] = str(error)
        rows.append(row)
    if not strict:
        launched = [row for row in rows if "command_argv" in row]
        status = "CONFORMANT" if launched and all(row["outcome"] == "PASS" for row in launched) else "ABSTAIN"
    elif any(row["outcome"] == "FAIL" for row in rows):
        status = "NONCONFORMANT"
    elif len(rows) == len(plan["tests"]) and rows and all(row["outcome"] == "PASS" for row in rows):
        status = "CONFORMANT"
    else:
        status = "ABSTAIN"
    result = {"status": status, "observations": rows}
    if strict:
        result.update(plan_digest=plan["digest"], coverage_certificate=plan["coverage_certificate"],
                      contract_sha256=digest(json.dumps(contract, sort_keys=True).encode()),
                      registry_sha256=digest(json.dumps(registry, sort_keys=True).encode()))
    return result


def cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "p4_testing", *args],
                          cwd=ROOT, capture_output=True, text=True, check=False)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "contracts/examples"
    contract = json.loads((base / "expm1_execution_demo.json").read_text())
    registry = json.loads((base / "expm1_execution_registry.json").read_text())
    plan_path = output / "base_plan.json"
    compiled = cli("compile", "--contract", str(base / "expm1_execution_demo.json"),
                   "--risk-profile", str(base / "expm1_execution_risk.json"),
                   "--seed", "17", "--max-invocations", "2", "--wall-seconds", "10",
                   "--output", str(plan_path))
    if compiled.returncode:
        raise RuntimeError(compiled.stderr)
    plan = json.loads(plan_path.read_text())
    results = []
    for scenario, expected in zip(SCENARIOS, EXPECTED):
        directory = output / scenario
        selected_plan, selected_contract, selected_registry = copy.deepcopy(plan), copy.deepcopy(contract), copy.deepcopy(registry)
        if scenario == "numerical_bias":
            selected_registry["commands"]["reference.decimal_value"] += ["--bias", "0.01"]
        elif scenario == "missing_worker":
            del selected_registry["commands"]["reference.decimal_value"]
        elif scenario in {"invalid_json", "wrong_id", "zero_calls", "non_evaluable"}:
            selected_registry["commands"]["reference.decimal_value"] = ["{python}", "{p4_root}/examples/workflow_observation_control.py",
                                                                         "--test-id", "{test_id}", "--mode", scenario]
        elif scenario == "modified_plan":
            selected_plan["seed"] += 1
        elif scenario == "changed_contract":
            selected_contract["relations"]["decimal_value"]["parameters"]["absolute_tolerance"] = 1e-3
        for name, value in (("plan", selected_plan), ("contract", selected_contract), ("registry", selected_registry)):
            save(directory / (name + ".json"), value)
        decisions = {}
        for arm in ("exit_only", "strict_scripted"):
            report = direct_workflow(selected_plan, selected_contract, selected_registry, strict=arm == "strict_scripted")
            save(directory / (arm + ".json"), report)
            decisions[arm] = report["status"]
        report_path = directory / "p4.json"
        if report_path.exists():
            raise ValueError("use a new output directory, not stale results")
        completed = cli("execute", "--contract", str(directory / "contract.json"),
                        "--plan", str(directory / "plan.json"), "--registry", str(directory / "registry.json"),
                        "--artifact-root", str(ROOT), "--output", str(report_path))
        save(directory / "p4_process.json", {"exit_code": completed.returncode,
                                             "stdout": completed.stdout, "stderr": completed.stderr})
        if completed.returncode == 2 and not report_path.exists():
            decisions["p4"] = "REFUSED"
        elif completed.returncode == 0 and report_path.exists():
            decisions["p4"] = json.loads(report_path.read_text())["status"]
        else:
            raise AssertionError("unexpected P4 process result: " + completed.stderr)
        if decisions["p4"] != expected or decisions["strict_scripted"] != expected:
            raise AssertionError((scenario, decisions, expected))
        results.append({"scenario": scenario, "expected": expected, **decisions})
    import p4_testing
    summary = {"schema": "P4-Workflow-Comparison-v1", "scope": "nine disclosed local controls; not framework superiority or detection rates",
               "python": sys.version, "p4_import": p4_testing.__file__,
               "protocol_sha256": digest((ROOT / "docs/SCP_TARGETED_VALIDATION_PROTOCOL.md").read_bytes()),
               "same_plan_and_workers_within_scenario": True,
               "rows": results, "strict_agrees_with_p4": True,
               "worker_file_sha256": digest((ROOT / "examples/expm1_relation_worker.py").read_bytes())}
    save(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output_dir.resolve())
    print(json.dumps(summary, indent=2))
