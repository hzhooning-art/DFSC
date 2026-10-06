"""Command-line entry points for reproducible P4 reconstruction work."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .benchmark import freeze_manifest, load_benchmark, load_case
from .compiler import Budget, canonical_plan_bytes, compile_contract, compile_plan
from .contracts import RiskProfile, load_contract
from .execution import execute_plan, load_registry
from .metrics import cluster_bootstrap, summarize_results
from .models import Layer, TestPlan, TestSpec
from .properties import check_coverage_certificate
from .replay import evaluate_captured_pair


def _p4_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _plan_payload(plan: TestPlan) -> dict[str, Any]:
    return json.loads(canonical_plan_bytes(plan).decode("utf-8"))


def _load_plan(path: Path) -> TestPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "P4-Test-Plan-v1":
        raise ValueError("unsupported test plan")
    tests = tuple(
        TestSpec(
            test_id=item["test_id"],
            layer=Layer(item["layer"]),
            relation=item["relation"],
            parameters=item["parameters"],
            max_invocations=item["max_invocations"],
            marginal_mechanisms=tuple(item.get("marginal_mechanisms", ())),
            estimated_cost=item.get("estimated_cost", 1.0),
        )
        for item in payload["tests"]
    )
    plan = TestPlan(
        schema=payload["schema"],
        contract_id=payload["contract_id"],
        seed=payload["seed"],
        tests=tests,
        layers=tuple(Layer(item) for item in payload["layers"]),
        max_invocations=payload["max_invocations"],
        wall_seconds=payload["wall_seconds"],
        digest=payload["digest"],
        risk_mechanisms=tuple(payload.get("risk_mechanisms", ())),
        status=payload.get("status", "READY"),
        reason_code=payload.get("reason_code"),
        coverage_certificate=payload.get("coverage_certificate", {}),
    )
    expected = hashlib.sha256(canonical_plan_bytes(plan, include_digest=False)).hexdigest()
    if expected != plan.digest:
        raise ValueError("test plan digest mismatch")
    if plan.risk_mechanisms and not check_coverage_certificate(plan):
        raise ValueError("invalid risk coverage certificate")
    return plan


def _audit_manifest(path: Path) -> tuple[bool, str]:
    if path.name != "frozen.json":
        return False, "manifest is not frozen"
    metadata_path = _p4_root() / "benchmark" / "FREEZE_METADATA.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != metadata["manifest_sha256"]:
            return False, "manifest hash mismatch"
        manifest = load_benchmark(path)
        frozen = freeze_manifest(manifest)
        if frozen.digest != metadata["admitted_manifest_digest"]:
            return False, "admitted manifest hash mismatch"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, f"audit failed: {error}"
    return True, "audit passed"


def _cmd_compile(args: argparse.Namespace) -> int:
    try:
        contract = load_contract(Path(args.contract))
        budget = Budget(args.max_invocations, args.wall_seconds)
        if args.risk_profile:
            profile = RiskProfile.from_dict(json.loads(Path(args.risk_profile).read_text(encoding="utf-8")))
            plan = compile_plan(contract, profile, budget, seed=args.seed)
        else:
            plan = compile_contract(contract, args.seed, budget)
        _write_json(Path(args.output), _plan_payload(plan))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return _error(f"compile failed: {error}")
    print(f"compiled {plan.contract_id}: {plan.status} {plan.digest}")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    try:
        case = load_case(Path(args.case))
        plan = _load_plan(Path(args.plan))
        if plan.status != "READY":
            return _error(f"replay refused: plan status is {plan.status}")
        if not any(test.layer is Layer.REPLAY for test in plan.tests):
            return _error("replay refused: plan selects no historical replay relation")
        result = evaluate_captured_pair(case, plan)
        _write_json(Path(args.output), {
            "complete_pair": result.complete_pair,
            "runner_hash_matches": result.runner_hash_matches,
            "detected": result.pair_result.detected,
            "status": result.pair_result.status,
            "case_id": result.pair_result.case_id,
            "defect_family": result.pair_result.defect_family,
            "project": result.pair_result.project,
            "plan_digest": plan.digest,
            "plan_status": plan.status,
            "coverage_certificate": dict(plan.coverage_certificate),
            "evidence_kind": "captured_pair_validation",
            "selected_relations_executed": False,
        })
    except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as error:
        return _error(f"replay failed: {error}")
    print(f"replayed {case.case_id}: {result.pair_result.status}")
    return 0


def _cmd_execute(args: argparse.Namespace) -> int:
    try:
        contract = load_contract(Path(args.contract))
        plan = _load_plan(Path(args.plan))
        registry = load_registry(Path(args.registry))
        artifact_root = Path(args.artifact_root).resolve() if args.artifact_root else _p4_root()
        if not artifact_root.is_dir():
            raise ValueError(f"artifact root is not a directory: {artifact_root}")
        result = execute_plan(plan, contract, registry, root=artifact_root)
        result["contract_file_sha256"] = hashlib.sha256(Path(args.contract).read_bytes()).hexdigest()
        result["registry_file_sha256"] = hashlib.sha256(Path(args.registry).read_bytes()).hexdigest()
        _write_json(Path(args.output), result)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return _error(f"execute failed: {error}")
    print(f"executed {plan.contract_id}: {result['status']}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        raw = json.loads(Path(args.rows).read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw["rows"]
        summary = summarize_results(rows)
        bootstrap = cluster_bootstrap(rows, seed=args.seed, replicates=args.bootstrap_replicates)
        _write_json(Path(args.output), {
            "summary": asdict(summary),
            "bootstrap": asdict(bootstrap),
        })
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        return _error(f"evaluate failed: {error}")
    print(f"evaluated {summary.families} families across {summary.projects} projects")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    ok, message = _audit_manifest(Path(args.manifest))
    if not ok:
        return _error(message)
    print(message)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p4-testing", description="Auditable P4 reconstruction commands")
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile", help="compile a component contract")
    compile_parser.add_argument("--contract", required=True)
    compile_parser.add_argument("--risk-profile", help="JSON risk profile; enables risk-aware selection and abstention")
    compile_parser.add_argument("--seed", type=int, required=True)
    compile_parser.add_argument("--max-invocations", type=int, required=True,
                                help="maximum selected relation slots during planning; not measured runtime calls")
    compile_parser.add_argument("--wall-seconds", type=float, required=True)
    compile_parser.add_argument("--output", required=True)
    replay_parser = sub.add_parser("replay", help="validate an existing captured buggy/fixed pair; does not execute all selected relations")
    replay_parser.add_argument("--case", required=True)
    replay_parser.add_argument("--plan", required=True)
    replay_parser.add_argument("--output", required=True)
    execute_parser = sub.add_parser("execute", help="run every selected relation using a trusted local command registry")
    execute_parser.add_argument("--contract", required=True)
    execute_parser.add_argument("--plan", required=True)
    execute_parser.add_argument("--registry", required=True)
    execute_parser.add_argument("--artifact-root", help="trusted root for worker paths and working directory; required for repo-relative registries after wheel installation")
    execute_parser.add_argument("--output", required=True)
    evaluate_parser = sub.add_parser("evaluate", help="summarize frozen result rows")
    evaluate_parser.add_argument("--rows", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--seed", type=int, default=20260916)
    evaluate_parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    audit_parser = sub.add_parser("audit", help="audit a frozen benchmark manifest")
    audit_parser.add_argument("--manifest", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    return {
        "compile": _cmd_compile,
        "replay": _cmd_replay,
        "execute": _cmd_execute,
        "evaluate": _cmd_evaluate,
        "audit": _cmd_audit,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
