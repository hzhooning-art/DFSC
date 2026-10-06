"""Run an installed-wheel transfer example; persist reports for review."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    example = ROOT / "contracts/examples"
    contract, registry = example / "ode_decay.json", example / "ode_decay_registry.json"
    def cli(*args):
        completed = subprocess.run([sys.executable, "-m", "p4_testing", *map(str, args)],
                                   cwd=ROOT, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise AssertionError(completed.stderr)
    plan = output / "plan.json"
    cli("compile", "--contract", contract, "--risk-profile", example / "ode_decay_risk.json",
        "--seed", 31, "--max-invocations", 3, "--wall-seconds", 30, "--output", plan)
    biased = json.loads(registry.read_text())
    biased["commands"]["reference.trajectory"] += ["--bias", "0.01"]
    biased_path = output / "biased_registry.json"
    biased_path.write_text(json.dumps(biased, indent=2) + "\n", encoding="utf-8")
    for name, source, expected in (("positive", registry, "CONFORMANT"), ("biased", biased_path, "NONCONFORMANT")):
        report_path = output / (name + ".json")
        cli("execute", "--contract", contract, "--plan", plan, "--registry", source,
            "--artifact-root", ROOT, "--output", report_path)
        report = json.loads(report_path.read_text())
        assert report["status"] == expected and report["selected_relations_executed"]
        assert report["selected_relation_count"] == 3
        assert report["reported_subject_invocations"] == 10
        assert all(row["stdout_sha256"] and row["command_file_sha256"] for row in report["observations"])
        assert sorted(row["outcome"] for row in report["observations"]) == (["PASS"] * 3 if name == "positive" else ["FAIL", "PASS", "PASS"])
    import p4_testing
    summary = {"schema": "P4-ODE-Transfer-v1", "python": sys.version,
               "executable": sys.executable, "p4_import": p4_testing.__file__,
               "p4_version": importlib.metadata.version("dfsc-primitive-protocol"),
               "scipy": importlib.metadata.version("scipy"), "numpy": importlib.metadata.version("numpy"),
               "positive_status": "CONFORMANT", "biased_status": "NONCONFORMANT",
               "relations_per_plan": 3, "reported_solve_ivp_calls_per_run": 10,
               "core_example_specific_changes": False,
               "scope": "author-performed current-API transfer; not independent adoption or a historical defect"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir.resolve()), indent=2))
