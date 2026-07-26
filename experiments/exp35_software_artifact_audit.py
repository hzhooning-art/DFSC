"""Software-artifact audit for the dfsc environment.

The audit turns the P1--P3 software claim into machine-readable evidence:
dfsc is not only a numerical formula, but a small reproducible environment with
public APIs, examples, tests, diagnostic entry points, generated result assets,
and paper-generation utilities.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates = [
        path,
        path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}"),
        RESULTS / f"{path.stem}_{int(time.time())}{path.suffix}",
        ROOT / f"{path.stem}_{int(time.time())}{path.suffix}",
    ]
    last_error: PermissionError | None = None
    for candidate in candidates:
        try:
            f = candidate.open("w", newline="", encoding="utf-8")
            path = candidate
            break
        except PermissionError as exc:
            last_error = exc
    else:
        assert last_error is not None
        raise last_error
    with f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def file_exists(rel: str) -> bool:
    return (ROOT / rel).is_file()


def project_file_exists(rel: str) -> bool:
    return (PROJECT / rel).is_file()


def dir_has_files(rel: str, suffix: str | None = None) -> tuple[bool, int]:
    path = ROOT / rel
    if not path.is_dir():
        return False, 0
    files = [p for p in path.rglob("*") if p.is_file() and (suffix is None or p.suffix == suffix)]
    return bool(files), len(files)


def count_result_files() -> tuple[int, int]:
    csv_count = len(list((RESULTS / "tables").glob("*.csv"))) if (RESULTS / "tables").is_dir() else 0
    json_count = len(list(RESULTS.glob("*.json"))) if RESULTS.is_dir() else 0
    return csv_count, json_count


def check_public_api() -> tuple[bool, str]:
    module = importlib.import_module("dfsc")
    required = [
        "MLSLConfig",
        "MittagLefflerSpectralLayer",
        "ForcedMittagLefflerSpectralLayer",
        "build_mlsl",
        "build_dirichlet_mlsl_1d",
        "build_periodic_mlsl_2d",
        "mittag_leffler_e",
        "mittag_leffler_e_ab_hybrid",
        "FNO1D",
        "DeepONet1D",
        "capability_report",
        "component_summary",
        "implemented_components",
        "HybridResidualModel",
        "MittagLefflerResidualRegressor",
        "make_trainable_orders",
        "mlsl_applicability_report",
        "VariableOrderMLSL",
        "DistributedOrderMLSL",
        "validate_dataset_manifest",
        "benchmark_targets",
        "build_operator_mlsl",
        "build_generalized_operator_mlsl",
        "build_graph_mlsl",
        "spectral_decomposition_from_operator",
        "generalized_spectral_decomposition",
        "evaluate_mittag_leffler",
        "choose_algorithm",
        "AutoDFSC",
        "CaputoL1",
        "CaputoL1Problem",
        "caputo_l1_linear_solve",
        "CaputoHistoryProblem",
        "CaputoL1HistoryDirect",
        "CaputoL1HistoryFFT",
        "CaputoHistoryDiagnostics",
        "caputo_l1_derivative_direct",
        "caputo_l1_derivative_fft",
        "caputo_l1_history",
        "MLSLKrylov",
        "KrylovDiagnostics",
        "lanczos_mittag_leffler_action",
        "MLSLArnoldi",
        "ArnoldiDiagnostics",
        "GeneralLinearOperator",
        "GeneralOperatorProblem",
        "arnoldi_mittag_leffler_action",
        "ComplexMittagLefflerEvaluation",
        "evaluate_complex_mittag_leffler",
        "mittag_leffler_e_complex_series",
        "ApplicationCase",
        "application_catalog",
        "anomalous_diffusion_case",
        "assembled_relaxation_case",
        "network_diffusion_case",
        "advection_diffusion_case",
        "periodic_advection_diffusion_operator_1d",
        "SPTTrajectoryDataset",
        "SPTObservables",
        "load_anomdiffdb_mat",
        "split_trajectories",
        "estimate_wave_number",
        "empirical_spt_observables",
        "SelfAdjointLinearOperator",
        "LinearOperatorSpectralProblem",
        "MLSLPicard",
        "PicardDiagnostics",
        "SemilinearSpectralProblem",
        "semilinear_mild_picard",
        "algorithm_registry",
        "FractionalSpectralProblem",
        "OperatorSpectralProblem",
        "GeneralizedOperatorSpectralProblem",
        "GraphSpectralProblem",
        "ForcedSpectralProblem",
        "Solution",
        "solve",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    return not missing, "missing=" + ",".join(missing) if missing else f"{len(required)} public symbols"


def check_dfsc_namespace() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dfsc = importlib.import_module("dfsc")
    summary = dfsc.component_summary()
    checks = [
        (
            "canonical_name",
            getattr(dfsc, "LIBRARY_NAME", "") == "dfsc",
            "dfsc canonical library name",
            "dfsc",
        ),
        (
            "no_compatibility_alias",
            summary.get("compatibility_aliases") == [],
            "single public namespace",
            "no alias package",
        ),
        (
            "component_registry",
            summary.get("implemented_count", 0) >= 8,
            f"{summary.get('implemented_count', 0)} implemented components",
            ">=8 implemented components",
        ),
        (
            "operator_adapter",
            hasattr(dfsc, "build_operator_mlsl")
            and hasattr(dfsc, "build_generalized_operator_mlsl")
            and hasattr(dfsc, "build_graph_mlsl"),
            "standard, generalized, and graph operator adapters importable",
            "assembled-operator adapter entry points",
        ),
        (
            "problem_algorithm_solve",
            hasattr(dfsc, "solve") and hasattr(dfsc, "FractionalSpectralProblem") and hasattr(dfsc, "algorithm_registry"),
            "problem, algorithm, and solve entry points",
            "SciML-style specialized interface",
        ),
        (
            "diagnostic_algorithm_selection",
            hasattr(dfsc, "evaluate_mittag_leffler") and hasattr(dfsc, "choose_algorithm"),
            "diagnostic evaluator and automatic selection importable",
            "inspectable numerical routing",
        ),
        (
            "history_fallback",
            hasattr(dfsc, "CaputoL1Problem") and hasattr(dfsc, "CaputoL1"),
            "constant-order Caputo L1 fallback importable",
            "first history-aware fallback path",
        ),
        (
            "fft_history_operator",
            hasattr(dfsc, "CaputoHistoryProblem") and hasattr(dfsc, "caputo_l1_derivative_fft"),
            "direct and FFT Caputo-L1 trajectory operators importable",
            "differentiable long-trajectory history evaluation",
        ),
        (
            "krylov_and_semilinear",
            hasattr(dfsc, "MLSLKrylov") and hasattr(dfsc, "SemilinearSpectralProblem"),
            "Lanczos matrix-function action and semilinear Picard problem importable",
            "operator-scale and nonlinear extension paths",
        ),
        (
            "sparse_matrix_free_contract",
            hasattr(dfsc, "SelfAdjointLinearOperator") and hasattr(dfsc, "LinearOperatorSpectralProblem"),
            "sparse and matrix-free operator entry points importable",
            "non-dense operator representation path",
        ),
        (
            "complex_general_operator",
            hasattr(dfsc, "MLSLArnoldi") and hasattr(dfsc, "evaluate_complex_mittag_leffler"),
            "controlled complex evaluator and general-operator Arnoldi importable",
            "moderate-radius non-self-adjoint propagation path",
        ),
        (
            "workflow_entrypoints",
            hasattr(dfsc, "HybridResidualModel") and hasattr(dfsc, "make_trainable_orders"),
            "hybrid and inverse-order workflows",
            "workflow helpers importable",
        ),
        (
            "domain_application_templates",
            hasattr(dfsc, "ApplicationCase")
            and len(dfsc.application_catalog()) == 4
            and hasattr(dfsc, "advection_diffusion_case"),
            "four scoped domain templates importable",
            "tested anomalous diffusion, relaxation, graph, and transport entry points",
        ),
        (
            "experimental_spt_evidence_chain",
            hasattr(dfsc, "load_anomdiffdb_mat")
            and hasattr(dfsc, "MittagLefflerResidualRegressor")
            and file_exists("data/external/anomdiffdb/manifest.json")
            and file_exists("results/real_spt_evidence_chain_summary.json"),
            "experimental SPT loader, hybrid regressor, provenance manifest, and result summary",
            "complete local real-data evidence chain",
        ),
    ]
    for case, passed, observed, expected in checks:
        rows.append(
            {
                "axis": "dfsc_ecosystem",
                "case": case,
                "observed": observed,
                "expected": expected,
                "passed": passed,
            }
        )
    return rows


def check_docs_examples_tests() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    checks = [
        ("packaging", "pyproject", file_exists("pyproject.toml"), "editable Python package metadata"),
        ("packaging", "requirements", file_exists("requirements.txt"), "minimal dependency record"),
        ("documentation", "api_doc", file_exists("docs/API.md"), "API reference page"),
        ("documentation", "roadmap_doc", file_exists("docs/ROADMAP.md"), "roadmap separates stable and experimental work"),
        ("documentation", "ecosystem_maturity_doc", file_exists("docs/ECOSYSTEM_MATURITY.md"), "ecosystem maturity comparison notes"),
        ("documentation", "release_checklist", file_exists("docs/RELEASE_CHECKLIST.md"), "release checklist"),
        ("documentation", "status_doc", file_exists("docs/IMPLEMENTATION_STATUS.md"), "implementation status page"),
        ("documentation", "generality_doc", file_exists("docs/GENERALITY_VALIDATION.md"), "generality-validation notes"),
        ("packaging", "license", file_exists("LICENSE"), "open-source license"),
        ("packaging", "citation", file_exists("CITATION.cff"), "citation metadata"),
        ("packaging", "docs_site_config", file_exists("mkdocs.yml"), "documentation-site config"),
        ("packaging", "root_ci", project_file_exists(".github/workflows/dfsc-ci.yml"), "repository-level CI workflow"),
        ("data_contract", "benchmark_manifest_template", file_exists("examples/public_benchmark_manifest_template.json"), "public benchmark manifest template"),
    ]
    examples_ok, example_count = dir_has_files("examples", ".py")
    tests_ok, test_count = dir_has_files("tests", ".py")
    tools_ok, tool_count = dir_has_files("tools", ".py")
    checks.extend(
        [
            ("examples", "example_scripts", examples_ok and example_count >= 3, f"{example_count} Python examples"),
            ("tests", "unit_tests", tests_ok and test_count >= 3, f"{test_count} unit-test files"),
            ("diagnostics", "diagnostic_tools", tools_ok and tool_count >= 3, f"{tool_count} tool scripts"),
        ]
    )
    for axis, case, passed, observed in checks:
        rows.append(
            {
                "axis": axis,
                "case": case,
                "observed": observed,
                "expected": "present",
                "passed": passed,
            }
        )
    return rows


def check_reproducibility_assets() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    csv_count, json_count = count_result_files()
    figure_dir = PROJECT / "paper_assets" / "figures"
    table_dir = PROJECT / "paper_assets" / "tables"
    figure_count = len(list(figure_dir.glob("*.pdf"))) if figure_dir.is_dir() else 0
    table_count = len(list(table_dir.glob("*.tex"))) if table_dir.is_dir() else 0
    checks = [
        ("reproducibility", "validation_gate", file_exists("validate.py"), "validate.py"),
        ("reproducibility", "core_reproduction", file_exists("tools/reproduce_core.py"), "tools/reproduce_core.py"),
        ("reproducibility", "maturity_report", file_exists("tools/maturity_report.py"), "tools/maturity_report.py"),
        ("results", "csv_result_tables", csv_count >= 20, f"{csv_count} CSV result tables"),
        ("results", "json_summaries", json_count >= 10, f"{json_count} JSON summaries"),
        ("paper_assets", "latex_tables", table_count >= 10, f"{table_count} generated LaTeX tables"),
        ("paper_assets", "figures", figure_count >= 3, f"{figure_count} generated PDF figures"),
    ]
    for axis, case, passed, observed in checks:
        rows.append(
            {
                "axis": axis,
                "case": case,
                "observed": observed,
                "expected": "available and regenerable",
                "passed": passed,
            }
        )
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    rows = check_docs_examples_tests()
    rows.extend(check_dfsc_namespace())
    api_ok, api_observed = check_public_api()
    rows.append(
        {
            "axis": "api",
            "case": "public_symbols",
            "observed": api_observed,
            "expected": "layer/evaluator/baseline/diagnostic symbols",
            "passed": api_ok,
        }
    )
    rows.extend(check_reproducibility_assets())

    csv_path = write_csv(TABLES / "software_artifact_audit.csv", rows)
    total = len(rows)
    passed = sum(1 for row in rows if bool(row["passed"]))
    by_axis: dict[str, dict[str, int]] = {}
    for row in rows:
        axis = str(row["axis"])
        by_axis.setdefault(axis, {"passed": 0, "total": 0})
        by_axis[axis]["total"] += 1
        by_axis[axis]["passed"] += int(bool(row["passed"]))

    summary = {
        "total_checks": total,
        "passed_checks": passed,
        "failed_checks": total - passed,
        "pass_rate": passed / total,
        "by_axis": by_axis,
        "status": "software-paper-core-ready" if passed == total else "needs-artifact-hardening",
        "csv_path": csv_path.relative_to(ROOT).as_posix(),
    }
    summary_path = RESULTS / "software_artifact_audit_summary.json"
    try:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except PermissionError:
        summary_path = RESULTS / "tables" / f"software_artifact_audit_summary_{int(time.time())}.json"
        try:
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except PermissionError:
            summary_path = ROOT / f"software_artifact_audit_summary_{int(time.time())}.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        summary["summary_path"] = summary_path.relative_to(ROOT).as_posix()
    except ValueError:
        summary["summary_path"] = str(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
