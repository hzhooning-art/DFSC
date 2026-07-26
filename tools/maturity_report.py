"""Generate a maturity report from current dfsc results."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from importlib import metadata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import ReliabilityReport, capability_report, component_summary


RESULTS = ROOT / "results"
GENERATED_RESULTS = ROOT / "generated_results"


def read_json(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_generated_json(name: str) -> dict[str, Any]:
    path = GENERATED_RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    capability = capability_report()
    try:
        package_version = metadata.version("dfsc")
    except metadata.PackageNotFoundError:
        package_version = None
    generality = read_json("primitive_generality_summary.json")
    gap = read_json("gap_closure_summary.json")
    gpu = read_json("gpu_validation_summary.json")
    two_d = read_json("two_d_nonlinear_extension_summary.json")
    boundary = read_json("boundary_generality_summary.json")
    forcing = read_json("manufactured_forcing_summary.json")
    audit = read_json("software_artifact_audit_summary.json")
    krylov_semilinear = read_json("krylov_semilinear_summary.json")
    sparse_matrix_free = read_json("sparse_matrix_free_summary.json")
    fast_history = read_json("fast_history_summary.json")
    complex_arnoldi = read_json("complex_arnoldi_summary.json")
    application_domains = read_json("application_domain_validation_summary.json")
    real_spt = read_json("real_spt_evidence_chain_summary.json")
    real_brownian = read_generated_json("real_brownian_two_population_summary.json")
    real_geomembrane = read_generated_json("real_geomembrane_relaxation_summary.json")
    real_geotes = read_generated_json("real_geotes_cross_cycle_summary.json")
    external_solver = read_generated_json("external_fractional_solver_benchmark_summary.json")
    adaptive = read_generated_json("adaptive_krylov_calibration_summary.json")
    pre_release = read_json("pre_release_readiness.json")

    checks = {
        "runtime_smoke_tests": bool(capability.get("all_smoke_tests_passed")),
        "editable_package_install": package_version == "0.1.0",
        "software_artifact_audit": audit.get("pass_rate") == 1.0,
        "primitive_generality": generality.get("primitive_generality_pass_rate") == 1.0,
        "boundary_generality": boundary.get("boundary_generality_pass_rate") == 1.0,
        "two_d_boundary_extension": two_d.get("2d_extended_boundary_pass_rate") == 1.0,
        "semilinear_backbone": two_d.get("semilinear_backbone_pass_rate") == 1.0,
        "reaction_diffusion": gap.get("reaction_diffusion_pass_rate") == 1.0,
        "eab_reference_accuracy": gap.get("eab_reference_max_relative_error", 1.0) < 1e-5,
        "manufactured_forcing": forcing.get("manufactured_forcing_all_finite") is True,
        "gpu_validation": gpu.get("gpu_all_finite_checks") is True,
        "krylov_reference_and_gradient": (
            krylov_semilinear.get("krylov_full_dimension_relative_error", 1.0) < 1e-10
            and krylov_semilinear.get("krylov_alpha_gradient_relative_error", 1.0) < 1e-8
        ),
        "semilinear_picard_and_gradients": (
            krylov_semilinear.get("semilinear_retcode") == "success"
            and krylov_semilinear.get("semilinear_alpha_gradient_finite") is True
            and krylov_semilinear.get("semilinear_gamma_gradient_finite") is True
        ),
        "sparse_matrix_free_reference": (
            sparse_matrix_free.get("sparse_relative_error", 1.0) < 1e-10
            and sparse_matrix_free.get("matrix_free_relative_error", 1.0) < 1e-10
            and sparse_matrix_free.get("operator_parameter_gradient_finite") is True
        ),
        "large_matrix_free_and_gpu": (
            sparse_matrix_free.get("large_matrix_free_finite") is True
            and sparse_matrix_free.get("gpu", {}).get("matrix_free_finite") is True
            and sparse_matrix_free.get("gpu", {}).get("alpha_gradient_finite") is True
        ),
        "fft_history_reference_and_gradients": (
            fast_history.get("max_fft_vs_direct_relative_error", 1.0) < 1e-10
            and fast_history.get("alpha_gradient_relative_error", 1.0) < 1e-9
            and fast_history.get("trajectory_gradient_relative_error", 1.0) < 1e-9
        ),
        "fft_history_convergence_and_gpu": (
            fast_history.get("analytic_error_reduction", 0.0) > 10.0
            and fast_history.get("largest_fft_finite") is True
            and fast_history.get("gpu", {}).get("alpha_gradient_finite") is True
            and fast_history.get("gpu", {}).get("trajectory_gradient_finite") is True
        ),
        "complex_mittag_leffler_reference": (
            complex_arnoldi.get("complex_scalar_max_relative_error", 1.0) < 1e-11
            and complex_arnoldi.get("complex_scalar_all_converged") is True
        ),
        "arnoldi_reference_gradient_and_gpu": (
            complex_arnoldi.get("arnoldi_matrix_relative_error", 1.0) < 1e-11
            and complex_arnoldi.get("arnoldi_alpha_gradient_relative_error", 1.0) < 1e-7
            and complex_arnoldi.get("gpu", {}).get("alpha_gradient_finite") is True
        ),
        "scoped_application_domain_validation": (
            application_domains.get("catalog_size") == 4
            and application_domains.get("anomalous_diffusion_alpha_gradient_finite") is True
            and application_domains.get("anomalous_diffusion_beta_gradient_finite") is True
            and application_domains.get("assembled_relaxation_mass_projection") is True
            and application_domains.get("graph_constant_mode_max_error", 1.0) < 1e-12
            and application_domains.get("advection_diffusion_exponential_relative_error", 1.0) < 1e-11
            and application_domains.get("advection_diffusion_diffusivity_gradient_finite") is True
            and application_domains.get("advection_diffusion_velocity_gradient_finite") is True
        ),
        "experimental_spt_evidence_chain": (
            real_spt.get("num_trajectories", 0) == 3112
            and real_spt.get("num_localizations", 0) == 426620
            and real_spt.get("best_scattering_late_error_mean", 1.0) < 0.1
            and real_spt.get("direct_mlsl_vs_stretched_late_error_reduction", -1.0) > 0.0
            and real_spt.get("hybrid_vs_pure_mlp_late_error_reduction", -1.0) > 0.0
            and "model-conditional" in real_spt.get("interpretation_boundary", "")
            and "does not state" in real_spt.get("redistribution_boundary", "")
        ),
        "adaptive_error_control": (
            len(adaptive.get("summary", [])) == 3
            and all(
                row.get("fraction_actual_error_below_10x_rtol") == 1.0
                for row in adaptive.get("summary", [])
            )
        ),
        "external_solver_benchmark": (
            {row.get("task") for row in external_solver.get("rows", [])}
            == {"scalar relaxation", "coupled 2-state relaxation"}
            and {row.get("method") for row in external_solver.get("rows", [])}
            == {"dfsc adaptive direct query", "FDEint predictor-corrector", "pycaputo PECE"}
        ),
        "multi_domain_real_evidence": (
            real_brownian.get("eligible_trajectories_length_gt_40") == 660
            and len(real_geomembrane.get("tasks", [])) == 4
            and real_geomembrane.get("license") == "CC BY 4.0"
            and (ROOT / "data/external/anomdiffdb/manifest_brownian.json").exists()
            and (ROOT / "data/external/geomembrane/manifest.json").exists()
        ),
        "cross_cycle_forced_real_evidence": (
            real_geotes.get("license") == "CC BY 4.0"
            and len(real_geotes.get("summary", [])) == 4
            and next(
                row["cycle2_error_mean"] for row in real_geotes.get("summary", [])
                if row.get("model") == "DFSC"
            )
            < next(
                row["cycle2_error_mean"] for row in real_geotes.get("summary", [])
                if row.get("model") == "Integer propagation"
            )
            and (ROOT / "data/external/geotes/manifest.json").exists()
        ),
        "single_dfsc_namespace": (
            component_summary().get("python_package") == "dfsc"
            and component_summary().get("compatibility_aliases") == []
            and not list((ROOT / "mlsl").glob("*.py"))
        ),
        "numerical_reliability_contract": (
            ReliabilityReport is not None and (ROOT / "dfsc/reliability.py").exists()
        ),
        "pre_release_internal_gate": pre_release.get("score_percent", 0.0) >= 95.0,
    }
    passed = sum(1 for value in checks.values() if value)
    total = len(checks)
    report = {
        "library": component_summary(),
        "installed_package_version": package_version,
        "maturity_checks": checks,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total,
        "suggested_status": "specialized-beta" if passed == total else "needs-more-hardening",
        "artifact_core_gate": "pass" if passed == total else "fail",
        "maturity_axes": {
            "differentiable_spectral_core": "research-grade-tested",
            "problem_algorithm_interface": "pre-release-beta",
            "package_identity": "single-dfsc-distribution",
            "numerical_reliability": "validated-domain-and-empirical-error-contract",
            "history_aware_solver_breadth": "linear-caputo-stepper-plus-fft-trajectory-operator",
            "operator_coverage": "self-adjoint-generalized-plus-controlled-general-complex-arnoldi",
            "nonlinear_coverage": "mild-form-picard-for-known-spectral-backbones",
            "application_coverage": "four-templates-plus-cross-cycle-real-evidence-in-three-domains",
            "external_benchmark_evidence": "FDEint-and-pycaputo-on-scalar-and-coupled-linear-systems",
            "public_release_ecosystem": "not-released",
        },
        "production_release_ready": False,
        "internal_pre_release_ready": pre_release.get("score_percent", 0.0) >= 95.0,
        "pre_release_internal_readiness_percent": pre_release.get("score_percent"),
        "general_fractional_solver_ready": False,
        "software_paper_artifact_readiness": (
            "internally synchronized bilingual manuscript and evidence assets; "
            "public release and independent reproduction remain"
        ),
    }
    out = RESULTS / "maturity_report.json"
    try:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except PermissionError:
        out = RESULTS / "tables" / f"maturity_report_{int(time.time())}.json"
        try:
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except PermissionError:
            out = None
    report["written_to"] = None if out is None else str(out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
