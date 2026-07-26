"""Component registry for the dfsc software environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .factory import MLSLConfig, build_mlsl
from .forced_layer import ForcedMittagLefflerSpectralLayer
from .mittag_leffler import mittag_leffler_e, mittag_leffler_e_ab_hybrid
from .spectral_layer import MittagLefflerSpectralLayer


@dataclass(frozen=True)
class ComponentSpec:
    """Machine-readable description of a dfsc component."""

    name: str
    kind: str
    status: str
    entrypoint: str
    summary: str


COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        name="Mittag-Leffler spectral layer",
        kind="primitive",
        status="implemented",
        entrypoint="dfsc.build_mlsl",
        summary="Differentiable direct-query spectral propagator with trainable alpha/beta.",
    ),
    ComponentSpec(
        name="Forced Mittag-Leffler wrapper",
        kind="wrapper",
        status="implemented",
        entrypoint="dfsc.ForcedMittagLefflerSpectralLayer",
        summary="Two-parameter Duhamel-style wrapper for nonhomogeneous fractional dynamics.",
    ),
    ComponentSpec(
        name="Boundary spectral constructors",
        kind="constructor",
        status="implemented",
        entrypoint="dfsc.build_mlsl",
        summary="1D/2D Dirichlet, Neumann, periodic, and mixed spectral bases.",
    ),
    ComponentSpec(
        name="Operator spectral adapter",
        kind="constructor",
        status="implemented",
        entrypoint="dfsc.build_operator_mlsl",
        summary="Build a dfsc spectral layer from a symmetric PSD discrete operator or graph Laplacian.",
    ),
    ComponentSpec(
        name="Generalized operator spectral adapter",
        kind="constructor",
        status="implemented",
        entrypoint="dfsc.build_generalized_operator_mlsl",
        summary="Build a mass-aware spectral layer from assembled symmetric stiffness and mass matrices.",
    ),
    ComponentSpec(
        name="Lanczos Mittag-Leffler action",
        kind="solver",
        status="implemented",
        entrypoint="dfsc.lanczos_mittag_leffler_action",
        summary="Avoid full eigendecomposition for larger dense symmetric PSD operators.",
    ),
    ComponentSpec(
        name="Controlled complex Mittag-Leffler and Arnoldi action",
        kind="numerical-kernel",
        status="implemented",
        entrypoint="dfsc.arnoldi_mittag_leffler_action",
        summary="Moderate-radius complex evaluation and general-operator matrix-function actions.",
    ),
    ComponentSpec(
        name="Sparse and matrix-free operator contract",
        kind="interface",
        status="implemented",
        entrypoint="dfsc.SelfAdjointLinearOperator",
        summary="Apply Lanczos MLSL through sparse tensors or differentiable matvec callables.",
    ),
    ComponentSpec(
        name="Problem-algorithm-solve interface",
        kind="interface",
        status="implemented",
        entrypoint="dfsc.solve",
        summary="SciML-inspired problem, algorithm, and solution objects scoped to dfsc spectral dynamics.",
    ),
    ComponentSpec(
        name="Diagnostic Mittag-Leffler evaluator",
        kind="numerical-kernel",
        status="implemented",
        entrypoint="dfsc.evaluate_mittag_leffler",
        summary="Autograd-preserving evaluation with branch, finiteness, and embedded-disagreement diagnostics.",
    ),
    ComponentSpec(
        name="Numerical reliability contract",
        kind="interface",
        status="implemented",
        entrypoint="dfsc.ReliabilityReport",
        summary="Validated-domain, convergence, gradient-reliability, and non-rigorous error-estimate metadata.",
    ),
    ComponentSpec(
        name="Automatic algorithm selection",
        kind="diagnostic",
        status="implemented",
        entrypoint="dfsc.choose_algorithm",
        summary="Detached spectral-regime diagnostics selecting direct, stable, operator, graph, or forced workflows.",
    ),
    ComponentSpec(
        name="Caputo L1 history fallback",
        kind="solver",
        status="implemented",
        entrypoint="dfsc.CaputoL1Problem",
        summary="Differentiable implicit L1 time marching for constant-order linear Caputo systems.",
    ),
    ComponentSpec(
        name="FFT Caputo-L1 trajectory operator",
        kind="numerical-kernel",
        status="implemented",
        entrypoint="dfsc.caputo_l1_derivative_fft",
        summary="Differentiable quasi-linear full-trajectory history convolution for residual evaluation.",
    ),
    ComponentSpec(
        name="Semilinear mild-form Picard solver",
        kind="solver",
        status="implemented",
        entrypoint="dfsc.SemilinearSpectralProblem",
        summary="Iterative nonlinear extension using an MLSL backbone and Duhamel quadrature.",
    ),
    ComponentSpec(
        name="Hybrid residual workflow",
        kind="workflow",
        status="implemented",
        entrypoint="dfsc.HybridResidualModel",
        summary="Compose a known fractional backbone with a trainable neural residual head.",
    ),
    ComponentSpec(
        name="Experimental relaxation hybrid",
        kind="workflow",
        status="implemented",
        entrypoint="dfsc.MittagLefflerResidualRegressor",
        summary="Fit measured scalar relaxation or scattering curves with a trainable Mittag-Leffler backbone and gated neural residual.",
    ),
    ComponentSpec(
        name="Inverse-order workflow",
        kind="workflow",
        status="implemented",
        entrypoint="dfsc.make_trainable_orders",
        summary="Bounded trainable alpha/beta parameters for inverse fractional-order recovery.",
    ),
    ComponentSpec(
        name="Variable-order fractional operators",
        kind="experimental-wrapper",
        status="experimental",
        entrypoint="dfsc.VariableOrderMLSL",
        summary="Direct-query wrapper that composes MLSL with query-dependent alpha samples.",
    ),
    ComponentSpec(
        name="Distributed-order fractional operators",
        kind="experimental-wrapper",
        status="experimental",
        entrypoint="dfsc.DistributedOrderMLSL",
        summary="Differentiable quadrature mixture over retained MLSL alpha nodes.",
    ),
    ComponentSpec(
        name="Applicability contract",
        kind="diagnostic",
        status="implemented",
        entrypoint="dfsc.mlsl_applicability_report",
        summary="Machine-readable scope check for retained spectral MLSL workloads.",
    ),
    ComponentSpec(
        name="Public benchmark manifest contract",
        kind="data-contract",
        status="implemented",
        entrypoint="dfsc.validate_dataset_manifest",
        summary="Provenance-first schema for future public physical benchmark integration.",
    ),
    ComponentSpec(
        name="Experimental SPT benchmark workflow",
        kind="data-workflow",
        status="implemented",
        entrypoint="dfsc.load_anomdiffdb_mat",
        summary="Provenance-aware experimental trajectory loading, trajectory-level splitting, and ensemble observables.",
    ),
    ComponentSpec(
        name="Domain application templates",
        kind="application-interface",
        status="implemented",
        entrypoint="dfsc.application_catalog",
        summary="Tested templates for anomalous diffusion, assembled relaxation, graph diffusion, and non-self-adjoint transport.",
    ),
)


def list_components(*, include_planned: bool = True) -> list[dict[str, str]]:
    """Return the registered dfsc components as dictionaries."""

    rows = COMPONENTS if include_planned else tuple(c for c in COMPONENTS if c.status == "implemented")
    return [component.__dict__.copy() for component in rows]


def implemented_components() -> list[dict[str, str]]:
    """Return only implemented components."""

    return list_components(include_planned=False)


def component_summary() -> dict[str, object]:
    """Return aggregate component counts for audit and reporting."""

    implemented = [c for c in COMPONENTS if c.status == "implemented"]
    experimental = [c for c in COMPONENTS if c.status == "experimental"]
    planned = [c for c in COMPONENTS if c.status not in {"implemented", "experimental"}]
    return {
        "library_name": "dfsc",
        "python_package": "dfsc",
        "compatibility_aliases": [],
        "implemented_count": len(implemented),
        "experimental_count": len(experimental),
        "planned_count": len(planned),
        "implemented_components": [c.name for c in implemented],
        "experimental_components": [c.name for c in experimental],
        "planned_components": [c.name for c in planned],
    }


def primitive_entrypoints() -> dict[str, Callable | type]:
    """Return the core importable dfsc entry points."""

    return {
        "MLSLConfig": MLSLConfig,
        "MittagLefflerSpectralLayer": MittagLefflerSpectralLayer,
        "ForcedMittagLefflerSpectralLayer": ForcedMittagLefflerSpectralLayer,
        "build_mlsl": build_mlsl,
        "mittag_leffler_e": mittag_leffler_e,
        "mittag_leffler_e_ab_hybrid": mittag_leffler_e_ab_hybrid,
    }
