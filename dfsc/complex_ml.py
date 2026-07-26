"""Controlled complex-argument Mittag-Leffler evaluation for dfsc."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .reliability import ReliabilityReport


def _real_dtype(tensor: torch.Tensor) -> torch.dtype:
    return tensor.real.dtype if torch.is_complex(tensor) else tensor.dtype


def _real_parameter(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    dtype = _real_dtype(reference)
    parameter = torch.as_tensor(value, dtype=dtype, device=reference.device)
    if parameter.numel() != 1:
        raise ValueError("alpha must be a scalar real parameter")
    return parameter


def mittag_leffler_e_complex_series(
    alpha: torch.Tensor | float,
    z: torch.Tensor,
    *,
    terms: int = 120,
    max_radius: float = 4.0,
    allow_unvalidated: bool = False,
) -> torch.Tensor:
    """Evaluate ``E_alpha(z)`` by a differentiable complex power series.

    The default radius is deliberately conservative. This routine is the first
    complex-spectrum path in dfsc, not a replacement for future contour or
    rational evaluators in large sectors of the complex plane.
    """

    if not torch.is_tensor(z) or not torch.is_complex(z):
        raise TypeError("z must be a complex torch tensor")
    if terms < 2:
        raise ValueError("terms must be >= 2")
    if not bool(torch.isfinite(z.detach()).all().item()):
        raise ValueError("z must be finite")
    radius = float(torch.abs(z.detach()).max().cpu()) if z.numel() else 0.0
    if radius > max_radius and not allow_unvalidated:
        raise ValueError(
            f"complex series radius {radius:.6g} exceeds the validated limit {max_radius:.6g}"
        )
    alpha_t = _real_parameter(alpha, z)
    if not bool(((alpha_t > 0.0) & (alpha_t <= 2.0)).detach().item()):
        raise ValueError("complex Mittag-Leffler evaluation requires 0 < alpha <= 2")

    result = torch.ones_like(z)
    power = torch.ones_like(z)
    for index in range(1, terms):
        power = power * z
        coefficient = torch.exp(-torch.lgamma(alpha_t * index + 1.0))
        result = result + coefficient * power
    return result


@dataclass(frozen=True)
class ComplexMittagLefflerEvaluation:
    """Complex values and conservative embedded-series diagnostics."""

    values: torch.Tensor
    terms: int
    max_radius: float
    observed_radius: float
    finite: bool
    converged: bool
    embedded_absolute_disagreement: float
    embedded_relative_disagreement: float
    reliability: ReliabilityReport

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "complex-series",
            "terms": self.terms,
            "max_radius": self.max_radius,
            "observed_radius": self.observed_radius,
            "finite": self.finite,
            "converged": self.converged,
            "embedded_absolute_disagreement": self.embedded_absolute_disagreement,
            "embedded_relative_disagreement": self.embedded_relative_disagreement,
            "reliability": self.reliability.to_dict(),
            "error_estimate_kind": "embedded-truncation-disagreement-not-a-rigorous-bound",
            "validated_domain": f"complex |z| <= {self.max_radius}",
        }


def evaluate_complex_mittag_leffler(
    alpha: torch.Tensor | float,
    z: torch.Tensor,
    *,
    terms: int = 120,
    max_radius: float = 4.0,
    rtol: float | None = None,
    atol: float | None = None,
    strict: bool = False,
) -> ComplexMittagLefflerEvaluation:
    """Evaluate a controlled complex series with an embedded richer truncation."""

    values = mittag_leffler_e_complex_series(
        alpha,
        z,
        terms=terms,
        max_radius=max_radius,
    )
    richer_terms = max(terms + 24, int(math.ceil(1.25 * terms)))
    with torch.no_grad():
        reference = mittag_leffler_e_complex_series(
            _real_parameter(alpha, z).detach(),
            z.detach(),
            terms=richer_terms,
            max_radius=max_radius,
        )
        delta = torch.abs(values.detach() - reference)
        dtype = _real_dtype(z)
        active_atol = (1e-12 if dtype == torch.float64 else 2e-6) if atol is None else float(atol)
        active_rtol = (1e-10 if dtype == torch.float64 else 5e-5) if rtol is None else float(rtol)
        scale = torch.maximum(torch.abs(reference), torch.full_like(torch.abs(reference), active_atol))
        absolute = float(delta.max().cpu()) if delta.numel() else 0.0
        relative = float((delta / scale).max().cpu()) if delta.numel() else 0.0
        finite = bool(torch.isfinite(values.detach()).all().item())
        converged = finite and absolute <= active_atol + active_rtol * float(torch.abs(reference).max().cpu())
    observed_radius = float(torch.abs(z.detach()).max().cpu()) if z.numel() else 0.0
    within_domain = observed_radius <= max_radius and z.dtype in {torch.complex64, torch.complex128}
    reliability = ReliabilityReport(
        level=(
            "high"
            if converged and within_domain and z.dtype == torch.complex128
            else "moderate"
            if converged and within_domain
            else "low"
        ),
        within_validated_domain=within_domain,
        finite=finite,
        converged=converged,
        gradient_reliability="high" if z.dtype == torch.complex128 else "moderate",
        error_estimate_kind="embedded-truncation-disagreement-not-a-rigorous-bound",
        relative_error_estimate=relative,
        rigorous_error_bound=False,
        messages=("complex evaluation is restricted to the documented radius",),
        metadata={"validated_argument_domain": f"complex |z| <= {max_radius}"},
    )
    if strict and not reliability.trusted:
        raise RuntimeError(
            "complex Mittag-Leffler evaluation did not satisfy the dfsc reliability contract"
        )
    return ComplexMittagLefflerEvaluation(
        values,
        terms,
        max_radius,
        observed_radius,
        finite,
        converged,
        absolute,
        relative,
        reliability,
    )
