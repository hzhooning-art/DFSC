"""Numerical reliability contracts shared by dfsc evaluators and solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class ReliabilityReport:
    """Machine-readable assessment of a numerical result.

    A report distinguishes an empirical estimate from a rigorous error bound.
    The current dfsc kernels provide the former; callers can therefore reject
    unverified regimes without interpreting a finite tensor as certified.
    """

    level: str
    within_validated_domain: bool
    finite: bool
    converged: bool
    gradient_reliability: str
    error_estimate_kind: str
    relative_error_estimate: float | None = None
    rigorous_error_bound: bool = False
    messages: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def trusted(self) -> bool:
        return self.level in {"high", "moderate"} and self.finite and self.converged

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "trusted": self.trusted,
            "within_validated_domain": self.within_validated_domain,
            "finite": self.finite,
            "converged": self.converged,
            "gradient_reliability": self.gradient_reliability,
            "error_estimate_kind": self.error_estimate_kind,
            "relative_error_estimate": self.relative_error_estimate,
            "rigorous_error_bound": self.rigorous_error_bound,
            "messages": list(self.messages),
            **self.metadata,
        }


def assess_mittag_leffler_reliability(
    *,
    alpha: torch.Tensor,
    beta: torch.Tensor | None,
    z: torch.Tensor,
    method: str,
    branch_counts: dict[str, int],
    finite: bool,
    converged: bool,
    relative_error_estimate: float | None,
    error_estimate_kind: str,
) -> ReliabilityReport:
    """Assess the documented real-negative dfsc Mittag-Leffler regime."""

    negative_real = not torch.is_complex(z) and bool(torch.all(z.detach() <= 0).item())
    alpha_valid = bool(torch.all((alpha.detach() > 0) & (alpha.detach() <= 2)).item())
    beta_valid = beta is None or bool(torch.all(beta.detach() > 0).item())
    precision_valid = z.dtype in {torch.float32, torch.float64}
    within_domain = negative_real and alpha_valid and beta_valid and precision_valid

    messages: list[str] = []
    if not negative_real:
        messages.append("arguments are outside the validated non-positive real domain")
    if not precision_valid:
        messages.append("dtype is outside the validated float32/float64 reliability contract")
    if not beta_valid:
        messages.append("beta must be positive in the validated two-parameter regime")

    if not finite:
        level = "invalid"
    elif not within_domain or not converged:
        level = "low"
    elif branch_counts.get("transition", 0) or branch_counts.get("asymptotic", 0):
        level = "moderate"
        messages.append("hybrid branches use empirical disagreement, not a rigorous bound")
    else:
        level = "high"

    if not within_domain or not finite:
        gradient_reliability = "low"
    elif method == "series" and z.dtype == torch.float64:
        gradient_reliability = "high"
    else:
        gradient_reliability = "moderate"

    return ReliabilityReport(
        level=level,
        within_validated_domain=within_domain,
        finite=finite,
        converged=converged,
        gradient_reliability=gradient_reliability,
        error_estimate_kind=error_estimate_kind,
        relative_error_estimate=relative_error_estimate,
        rigorous_error_bound=False,
        messages=tuple(messages),
        metadata={
            "validated_argument_domain": "real z <= 0",
            "validated_alpha_domain": "0 < alpha <= 2",
            "validated_beta_domain": "beta > 0 when supplied",
            "method": method,
        },
    )


def assess_solution_reliability(
    *,
    finite: bool,
    retcode: str,
    diagnostics: dict[str, Any],
    warnings: tuple[str, ...],
) -> ReliabilityReport:
    """Create a conservative, algorithm-independent solution assessment."""

    indicators: dict[str, float] = {}
    for key in (
        "embedded_relative_disagreement",
        "estimated_relative_disagreement",
        "residual",
    ):
        value = diagnostics.get(key)
        if isinstance(value, (float, int)):
            indicators[key] = float(value)

    if not finite:
        level = "invalid"
    elif retcode != "success":
        level = "low"
    else:
        level = "moderate"

    relative_indicator = max(indicators.values()) if indicators else None
    messages = list(warnings)
    if not indicators:
        messages.append("no rigorous global solver error bound is available")

    return ReliabilityReport(
        level=level,
        within_validated_domain=retcode == "success",
        finite=finite,
        converged=retcode == "success",
        gradient_reliability="not-assessed",
        error_estimate_kind=(
            "empirical-algorithm-indicators" if indicators else "no-global-error-estimate"
        ),
        relative_error_estimate=relative_indicator,
        rigorous_error_bound=False,
        messages=tuple(messages),
        metadata={"indicators": indicators},
    )
