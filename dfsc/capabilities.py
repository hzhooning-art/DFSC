"""Applicability contracts for dfsc components.

The checks in this module deliberately describe the current implementation
boundary. They are not a mathematical certificate for every fractional model;
they are lightweight guards that help users decide whether MLSL is the right
primitive for a given spectral workload.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ApplicabilityReport:
    """Result of a dfsc applicability check."""

    supported: bool
    component: str
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    unsupported_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "supported": self.supported,
            "component": self.component,
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "unsupported_reasons": list(self.unsupported_reasons),
        }


def mlsl_applicability_report(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor | None = None,
    *,
    alpha_bounds: tuple[float, float] = (0.0, 2.0),
    beta_bounds: tuple[float, float] = (0.0, 4.0),
    allow_zero_mode: bool = True,
    real_nonnegative_tolerance: float = 1e-12,
) -> ApplicabilityReport:
    """Check whether a spectral problem matches the implemented MLSL scope.

    MLSL expects a diagonal spectral representation in which the retained modal
    rates generate real non-positive Mittag-Leffler arguments
    ``-c^2 lambda_n^(beta/2) t^alpha`` for non-negative times.
    """

    assumptions = (
        "The operator is supplied in a retained diagonal spectral basis.",
        "Query times are non-negative and the Mittag-Leffler propagator is known.",
        "This retained-eigenbasis adapter targets real non-positive spectral arguments.",
        "The layer is a differentiable primitive, not a general fractional solver.",
    )
    warnings: list[str] = []
    unsupported: list[str] = []

    if not torch.is_tensor(eigenvalues):
        eigenvalues = torch.as_tensor(eigenvalues)

    if eigenvalues.ndim != 1:
        unsupported.append("eigenvalues must be a one-dimensional retained spectrum")

    if torch.is_complex(eigenvalues):
        unsupported.append("complex spectra require the controlled GeneralOperatorProblem/MLSLArnoldi path")
    elif not torch.isfinite(eigenvalues).all().item():
        unsupported.append("eigenvalues must be finite")
    else:
        min_eval = float(eigenvalues.min().detach().cpu())
        if min_eval < -real_nonnegative_tolerance:
            unsupported.append("negative Laplacian eigenvalues would leave the tested real non-positive argument regime")
        if min_eval <= real_nonnegative_tolerance and not allow_zero_mode:
            unsupported.append("a zero mode is present but allow_zero_mode=False")
        if min_eval <= real_nonnegative_tolerance and allow_zero_mode:
            warnings.append("zero modes are allowed, but inverse problems may need regularization")

    if eigenvectors is not None:
        if not torch.is_tensor(eigenvectors):
            eigenvectors = torch.as_tensor(eigenvectors)
        if eigenvectors.ndim != 2:
            unsupported.append("eigenvectors must have shape (num_points, num_modes)")
        elif eigenvalues.ndim == 1 and eigenvectors.shape[1] != eigenvalues.shape[0]:
            unsupported.append("eigenvector columns must match the number of eigenvalues")
        if torch.is_complex(eigenvectors):
            unsupported.append("complex eigenvectors are not implemented in this artifact")
        elif not torch.isfinite(eigenvectors).all().item():
            unsupported.append("eigenvectors must be finite")

    alpha_lo, alpha_hi = alpha_bounds
    beta_lo, beta_hi = beta_bounds
    if not (0.0 < alpha_lo < alpha_hi <= 2.0):
        warnings.append("alpha_bounds extend outside the validated 0 < alpha <= 2 regime")
    if not (0.0 < beta_lo < beta_hi):
        warnings.append("beta_bounds should remain positive for lambda^(beta/2)")

    return ApplicabilityReport(
        supported=not unsupported,
        component="Mittag-Leffler spectral layer",
        assumptions=assumptions,
        warnings=tuple(warnings),
        unsupported_reasons=tuple(unsupported),
    )


def ecosystem_gap_report() -> dict[str, object]:
    """Return the current ecosystem maturity gaps in machine-readable form."""

    return {
        "implemented_scope": [
            "PyTorch dfsc primitive with autograd-compatible alpha/beta",
            "1D/2D tensor-product spectral constructors",
            "symmetric PSD operator and graph-Laplacian spectral adapters",
            "domain templates for anomalous diffusion, assembled relaxation, graph diffusion, and controlled advection-diffusion",
            "forced, inverse-order, and hybrid residual workflows",
            "CPU/GPU smoke tests, reproducible synthetic experiments, and one experimental SPT benchmark condition",
        ],
        "active_limitations": [
            "not a general-purpose fractional solver library",
            "the broadest validation remains diagonalizable real spectra; complex/general operators use a moderate-radius Arnoldi path",
            "variable-order and distributed-order support remains experimental",
            "no public PyPI release or hosted API reference yet",
            "real-data evidence currently covers one H-actin SPT condition whose source page does not state a redistribution license",
            "JAX/Julia backends and unstructured meshes are not implemented",
        ],
        "near_term_closure_plan": [
            "publish a versioned Python package and CI-tested source archive",
            "promote experimental variable/distributed-order wrappers after validation",
            "add a second public benchmark with an explicit redistribution license and independent physical modality",
            "build hosted API documentation from the current docs skeleton",
        ],
    }
