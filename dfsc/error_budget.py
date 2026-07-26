"""Composable numerical error budgets for dfsc propagators."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ErrorBudget:
    """Allocate a requested tolerance across numerical error sources."""

    rtol: float = 1e-8
    atol: float = 1e-10
    evaluator_fraction: float = 0.4
    krylov_fraction: float = 0.4
    projection_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.rtol < 0 or self.atol < 0 or self.rtol + self.atol == 0:
            raise ValueError("rtol and atol must be nonnegative and not both zero")
        fractions = (self.evaluator_fraction, self.krylov_fraction, self.projection_fraction)
        if any(value < 0 for value in fractions):
            raise ValueError("error-budget fractions must be nonnegative")
        if abs(sum(fractions) - 1.0) > 1e-12:
            raise ValueError("error-budget fractions must sum to one")

    def absolute_tolerance(self, reference_norm: float) -> float:
        return self.atol + self.rtol * abs(float(reference_norm))

    def allocations(self, reference_norm: float) -> dict[str, float]:
        total = self.absolute_tolerance(reference_norm)
        return {
            "evaluator": total * self.evaluator_fraction,
            "krylov": total * self.krylov_fraction,
            "projection": total * self.projection_fraction,
        }


@dataclass(frozen=True)
class ErrorComponent:
    name: str
    estimate: float | None
    allocation: float
    estimate_kind: str
    rigorous: bool = False

    @property
    def assessed(self) -> bool:
        return self.estimate is not None

    @property
    def satisfied(self) -> bool | None:
        return None if self.estimate is None else self.estimate <= self.allocation

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "estimate": self.estimate,
            "allocation": self.allocation,
            "estimate_kind": self.estimate_kind,
            "assessed": self.assessed,
            "satisfied": self.satisfied,
            "rigorous": self.rigorous,
        }


@dataclass(frozen=True)
class ErrorBudgetReport:
    requested_tolerance: float
    components: tuple[ErrorComponent, ...]

    @property
    def assessed(self) -> bool:
        return all(component.assessed for component in self.components)

    @property
    def satisfied(self) -> bool:
        return self.assessed and all(component.satisfied is True for component in self.components)

    @property
    def rigorous_global_bound(self) -> bool:
        return self.assessed and all(component.rigorous for component in self.components)

    @property
    def summed_estimate(self) -> float | None:
        if not self.assessed:
            return None
        return sum(float(component.estimate) for component in self.components if component.estimate is not None)

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_tolerance": self.requested_tolerance,
            "summed_estimate": self.summed_estimate,
            "assessed": self.assessed,
            "satisfied": self.satisfied,
            "rigorous_global_bound": self.rigorous_global_bound,
            "components": [component.to_dict() for component in self.components],
        }


def alternating_series_remainder_bound(
    alpha: torch.Tensor | float,
    z: torch.Tensor,
    *,
    terms: int,
    beta: torch.Tensor | float = 1.0,
) -> torch.Tensor | None:
    """Bound a negative-real Mittag-Leffler series by its first omitted term.

    The bound is returned only when ``0 < alpha <= 1``, ``beta > 0``, ``z <= 0``,
    and the omitted-term magnitudes have entered their decreasing regime.  In
    that restricted regime the alternating-series remainder theorem applies.
    ``None`` means that this certificate is unavailable, not that evaluation
    failed.
    """

    if terms < 2 or not torch.is_tensor(z) or torch.is_complex(z):
        return None
    alpha_t = torch.as_tensor(alpha, dtype=z.dtype, device=z.device)
    beta_t = torch.as_tensor(beta, dtype=z.dtype, device=z.device)
    if bool(torch.any((alpha_t <= 0) | (alpha_t > 1)).item()):
        return None
    if bool(torch.any(beta_t <= 0).item()) or bool(torch.any(z > 0).item()):
        return None
    k0 = torch.as_tensor(float(terms - 1), dtype=z.dtype, device=z.device)
    k1 = torch.as_tensor(float(terms), dtype=z.dtype, device=z.device)
    log_previous = k0 * torch.log(torch.abs(z).clamp_min(torch.finfo(z.dtype).tiny)) - torch.lgamma(
        alpha_t * k0 + beta_t
    )
    log_omitted = k1 * torch.log(torch.abs(z).clamp_min(torch.finfo(z.dtype).tiny)) - torch.lgamma(
        alpha_t * k1 + beta_t
    )
    previous = torch.where(z == 0, torch.zeros_like(z), torch.exp(log_previous))
    omitted = torch.where(z == 0, torch.zeros_like(z), torch.exp(log_omitted))
    if bool(torch.any(omitted > previous).item()):
        return None
    return omitted


def compose_error_budget_report(
    budget: ErrorBudget,
    *,
    reference_norm: float,
    evaluator_estimate: float | None,
    evaluator_rigorous: bool,
    krylov_estimate: float | None = None,
    projection_estimate: float | None = None,
    projection_rigorous: bool = False,
) -> ErrorBudgetReport:
    """Combine available component estimates without inventing missing terms."""

    allocations = budget.allocations(reference_norm)
    return ErrorBudgetReport(
        requested_tolerance=budget.absolute_tolerance(reference_norm),
        components=(
            ErrorComponent(
                "evaluator",
                evaluator_estimate,
                allocations["evaluator"],
                "alternating-series-bound" if evaluator_rigorous else "embedded-disagreement",
                evaluator_rigorous,
            ),
            ErrorComponent(
                "krylov",
                krylov_estimate,
                allocations["krylov"],
                "successive-subspace-disagreement" if krylov_estimate is not None else "not-assessed",
                False,
            ),
            ErrorComponent(
                "projection",
                projection_estimate,
                allocations["projection"],
                "omitted-mode-norm" if projection_estimate is not None else "not-assessed",
                projection_estimate is not None and projection_rigorous,
            ),
        ),
    )
