"""Measure dfsc internal readiness before a public release.

The score intentionally excludes public adoption, hosted-service availability,
and community history. Those are external maturity axes and cannot be created
honestly by a private repository audit.
"""

from __future__ import annotations

import io
import json
import sys
import tomllib
import unittest
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dfsc


def _json(name: str) -> dict[str, Any]:
    path = ROOT / "results" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _unit_test_gate() -> tuple[bool, dict[str, Any]]:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    return result.wasSuccessful() and result.testsRun >= 80, {
        "tests_run": result.testsRun,
        "tests_passed": passed,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }


def _gradient_gate() -> tuple[bool, dict[str, Any]]:
    z = torch.tensor([-0.2, -0.7, -1.1], dtype=torch.float64)
    alpha = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    value = dfsc.mittag_leffler_e(alpha, z, terms=120).sum()
    (gradient,) = torch.autograd.grad(value, alpha)
    step = 1e-5
    plus = dfsc.mittag_leffler_e(alpha.detach() + step, z, terms=120).sum()
    minus = dfsc.mittag_leffler_e(alpha.detach() - step, z, terms=120).sum()
    finite_difference = (plus - minus) / (2 * step)
    relative = float(
        torch.abs(gradient - finite_difference)
        / torch.abs(finite_difference).clamp_min(torch.finfo(torch.float64).eps)
    )
    return relative < 1e-7, {
        "autograd": float(gradient),
        "finite_difference": float(finite_difference),
        "relative_disagreement": relative,
    }


def _beta_gradient_gate() -> tuple[bool, dict[str, Any]]:
    alpha = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
    z = -torch.linspace(0.0, 0.8, 8, dtype=torch.float64)
    evaluation = dfsc.evaluate_mittag_leffler(
        alpha, z, beta=beta, method="series", terms=120
    )
    evaluation.values.square().mean().backward()
    passed = bool(torch.isfinite(alpha.grad) & torch.isfinite(beta.grad))
    return passed, {
        "alpha_gradient": float(alpha.grad),
        "beta_gradient": float(beta.grad),
    }


def _strict_guard_gate() -> tuple[bool, dict[str, Any]]:
    try:
        dfsc.evaluate_mittag_leffler(
            0.8,
            torch.tensor([0.2], dtype=torch.float64),
            method="series",
            strict=True,
        )
    except RuntimeError:
        return True, {"outside_domain_rejected": True}
    return False, {"outside_domain_rejected": False}


def _namespace_gate() -> tuple[bool, dict[str, Any]]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = config["tool"]["setuptools"]["packages"]
    legacy_sources = list((ROOT / "mlsl").glob("*.py")) if (ROOT / "mlsl").exists() else []
    offenders: list[str] = []
    for path in ROOT.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.parts[-2:-1] == ("__pycache__",):
            continue
        text = path.read_text(encoding="utf-8")
        if "from mlsl" in text or "import mlsl" in text:
            offenders.append(str(path.relative_to(ROOT)))
    passed = packages == ["dfsc"] and not legacy_sources and not offenders
    return passed, {
        "distributed_packages": packages,
        "legacy_source_files": [str(path) for path in legacy_sources],
        "legacy_imports": offenders,
    }


def _solution_gate() -> tuple[bool, dict[str, Any]]:
    x, layer = dfsc.build_dirichlet_mlsl_1d(
        num_points=24,
        num_modes=6,
        config=dfsc.MLSLConfig.stable(terms=80),
    )
    problem = dfsc.FractionalSpectralProblem(
        layer,
        torch.sin(torch.pi * x),
        torch.linspace(0.0, 0.01, 3),
        torch.tensor(0.8),
    )
    solution = dfsc.solve(problem)
    summary = solution.summary()
    return solution.success and solution.reliability is not None, summary


def _run_check(name: str, function: Callable[[], tuple[bool, dict[str, Any]]]) -> dict[str, Any]:
    try:
        passed, details = function()
        return {"name": name, "passed": bool(passed), "details": details}
    except Exception as exc:  # audit failures should remain inspectable
        return {"name": name, "passed": False, "details": {"error": repr(exc)}}


def main() -> None:
    torch.set_default_dtype(torch.float64)
    audit = _json("software_artifact_audit_summary.json")
    maturity = _json("maturity_report.json")
    spt = _json("real_spt_evidence_chain_summary.json")
    generated = ROOT / "generated_results"
    brownian = json.loads((generated / "real_brownian_two_population_summary.json").read_text(encoding="utf-8"))
    geomembrane = json.loads((generated / "real_geomembrane_relaxation_summary.json").read_text(encoding="utf-8"))
    geotes = json.loads((generated / "real_geotes_cross_cycle_summary.json").read_text(encoding="utf-8"))
    external_solver = json.loads((generated / "external_fractional_solver_benchmark_summary.json").read_text(encoding="utf-8"))

    checks = [
        _run_check(
            "E_1(z) exponential identity",
            lambda: (
                bool(
                    torch.allclose(
                        dfsc.mittag_leffler_e(
                            1.0, -torch.linspace(0.0, 2.0, 16), terms=100
                        ),
                        torch.exp(-torch.linspace(0.0, 2.0, 16)),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ),
                {},
            ),
        ),
        _run_check(
            "E_2(-x^2) cosine identity",
            lambda: (
                bool(
                    torch.allclose(
                        dfsc.mittag_leffler_e(
                            2.0, -(torch.linspace(0.0, 1.5, 16) ** 2), terms=100
                        ),
                        torch.cos(torch.linspace(0.0, 1.5, 16)),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ),
                {},
            ),
        ),
        _run_check(
            "E_1,1(z) exponential identity",
            lambda: (
                bool(
                    torch.allclose(
                        dfsc.mittag_leffler_e_ab(
                            1.0, 1.0, -torch.linspace(0.0, 2.0, 16), terms=100
                        ),
                        torch.exp(-torch.linspace(0.0, 2.0, 16)),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ),
                {},
            ),
        ),
        _run_check("alpha autograd finite-difference agreement", _gradient_gate),
        _run_check("two-parameter alpha/beta gradients", _beta_gradient_gate),
        _run_check("strict reliability-domain guard", _strict_guard_gate),
        _run_check(
            "dfsc identity and version",
            lambda: (
                dfsc.LIBRARY_NAME == "dfsc" and bool(dfsc.__version__),
                {"version": dfsc.__version__},
            ),
        ),
        _run_check("single distributed namespace", _namespace_gate),
        _run_check(
            "public primitive entrypoints",
            lambda: (
                len(dfsc.primitive_entrypoints()) >= 6,
                {"entrypoints": sorted(dfsc.primitive_entrypoints())},
            ),
        ),
        _run_check("solution reliability contract", _solution_gate),
        _run_check("unit-test gate", _unit_test_gate),
        _run_check(
            "software artifact audit",
            lambda: (audit.get("passed_checks") == audit.get("total_checks") == 38, audit),
        ),
        _run_check(
            "maturity evidence gate",
            lambda: (
                maturity.get("passed") == maturity.get("total")
                and maturity.get("total", 0) >= 21,
                maturity,
            ),
        ),
        _run_check(
            "runtime smoke tests",
            lambda: (
                bool(dfsc.capability_report().get("all_smoke_tests_passed")),
                dfsc.capability_report(),
            ),
        ),
        _run_check(
            "application catalog",
            lambda: (len(dfsc.application_catalog()) >= 4, {"count": len(dfsc.application_catalog())}),
        ),
        _run_check(
            "experimental SPT evidence",
            lambda: (
                spt.get("num_trajectories") == 3112
                and spt.get("direct_mlsl_vs_stretched_late_error_reduction", 0.0) > 0.0,
                spt,
            ),
        ),
        _run_check(
            "multi-domain real evidence",
            lambda: (
                brownian.get("eligible_trajectories_length_gt_40") == 660
                and len(geomembrane.get("tasks", [])) == 4
                and geomembrane.get("license") == "CC BY 4.0",
                {
                    "brownian_conditions": len(brownian.get("population_sizes", {})),
                    "geomembrane_conditions": len(geomembrane.get("tasks", [])),
                    "physical_domains": 2,
                },
            ),
        ),
        _run_check(
            "cross-cycle forced real evidence",
            lambda: (
                geotes.get("license") == "CC BY 4.0"
                and len(geotes.get("summary", [])) == 4
                and (ROOT / "data/external/geotes/manifest.json").exists(),
                {
                    "dataset": geotes.get("dataset"),
                    "models": [row.get("model") for row in geotes.get("summary", [])],
                },
            ),
        ),
        _run_check(
            "external solver breadth",
            lambda: (
                {row.get("task") for row in external_solver.get("rows", [])}
                == {"scalar relaxation", "coupled 2-state relaxation"}
                and {row.get("method") for row in external_solver.get("rows", [])}
                == {"dfsc adaptive direct query", "FDEint predictor-corrector", "pycaputo PECE"},
                {
                    "tasks": sorted({row.get("task") for row in external_solver.get("rows", [])}),
                    "methods": sorted({row.get("method") for row in external_solver.get("rows", [])}),
                },
            ),
        ),
        _run_check(
            "data provenance boundary",
            lambda: (
                "does not state" in spt.get("redistribution_boundary", "")
                and (ROOT / "data/external/anomdiffdb/manifest.json").exists(),
                {"redistribution_boundary": spt.get("redistribution_boundary")},
            ),
        ),
        _run_check(
            "continuous-integration definition",
            lambda: (
                all(
                    token in (ROOT / ".github/workflows/dfsc-ci.yml").read_text(encoding="utf-8")
                    for token in ('"3.10"', '"3.11"', '"3.12"', "unittest discover")
                ),
                {},
            ),
        ),
        _run_check(
            "documentation surface",
            lambda: (
                all((ROOT / path).exists() for path in ("README.md", "docs/API.md", "mkdocs.yml", "CHANGELOG.md", "CONTRIBUTING.md")),
                {},
            ),
        ),
        _run_check(
            "release metadata",
            lambda: (
                all(
                    (ROOT / path).exists()
                    for path in ("LICENSE", "CITATION.cff", ".zenodo.json", "MANIFEST.in", "docs/RELEASE_CHECKLIST.md")
                ),
                {},
            ),
        ),
    ]

    passed = sum(check["passed"] for check in checks)
    total = len(checks)
    score = 100.0 * passed / total
    report = {
        "scope": "pre-release internal readiness",
        "score_percent": score,
        "passed": passed,
        "total": total,
        "gate_threshold_percent": 95.0,
        "gate_passed": score >= 95.0,
        "checks": checks,
        "excluded_external_axes": [
            "public package and source release",
            "hosted documentation availability",
            "independent downstream adopters",
            "public issue and pull-request history",
            "redistribution permission for the current SPT raw data",
        ],
    }
    output = ROOT / "results" / "pre_release_readiness.json"
    try:
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except PermissionError:
        output = ROOT / "results" / "tables" / "pre_release_readiness.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except PermissionError:
            output = ROOT / "pre_release_readiness_latest.json"
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = output.relative_to(ROOT).as_posix()
    print(json.dumps(report, indent=2))

    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
