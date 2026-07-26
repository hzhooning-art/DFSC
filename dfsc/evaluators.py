"""Diagnostic Mittag-Leffler evaluation for dfsc workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch

from .mittag_leffler import hybrid_switch_region, mittag_leffler_e, mittag_leffler_e_ab
from .error_budget import alternating_series_remainder_bound
from .reliability import ReliabilityReport, assess_mittag_leffler_reliability


@dataclass
class MittagLefflerEvaluation:
    """Values plus inspectable numerical-routing diagnostics.

    ``estimated_*_disagreement`` compares the requested truncation with a
    richer embedded evaluation.  It is useful for diagnostics but is not a
    rigorous approximation-error bound.
    """

    values: torch.Tensor
    method: str
    branch_counts: dict[str, int]
    finite: bool
    converged: bool
    reliability: ReliabilityReport
    estimated_absolute_disagreement: float | None = None
    estimated_relative_disagreement: float | None = None
    error_estimate_kind: str = "not-requested"
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> torch.Tensor:
        return self.values

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "branch_counts": dict(self.branch_counts),
            "finite": self.finite,
            "converged": self.converged,
            "reliability": self.reliability.to_dict(),
            "estimated_absolute_disagreement": self.estimated_absolute_disagreement,
            "estimated_relative_disagreement": self.estimated_relative_disagreement,
            "error_estimate_kind": self.error_estimate_kind,
            "warnings": list(self.warnings),
            **self.metadata,
        }


@dataclass
class AdaptiveMittagLefflerEvaluation:
    """Tolerance-driven evaluation with an inspectable work history.

    Convergence is assessed from successive truncations.  This is an empirical
    a posteriori indicator, not a certified bound on the special-function
    approximation error.  The returned value retains the autograd graph of the
    selected truncation.
    """

    values: torch.Tensor
    converged: bool
    selected_terms: int
    attempted_terms: tuple[int, ...]
    relative_disagreements: tuple[float, ...]
    absolute_disagreements: tuple[float, ...]
    evaluation: MittagLefflerEvaluation

    @property
    def value(self) -> torch.Tensor:
        return self.values

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.evaluation.diagnostics(),
            "adaptive": True,
            "adaptive_converged": self.converged,
            "selected_terms": self.selected_terms,
            "attempted_terms": list(self.attempted_terms),
            "successive_relative_disagreements": list(self.relative_disagreements),
            "successive_absolute_disagreements": list(self.absolute_disagreements),
            "adaptive_estimate_is_error_bound": False,
        }


def _as_parameter(value: torch.Tensor | float, z: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(dtype=z.dtype, device=z.device)
    return torch.as_tensor(value, dtype=z.dtype, device=z.device)


def _auto_method(alpha: torch.Tensor, z: torch.Tensor, threshold: float) -> tuple[str, str]:
    if bool(torch.any(z > 0).item()):
        return "series", "positive arguments require the series path in the current implementation"
    _, _, hi = hybrid_switch_region(alpha, z, threshold=threshold)
    if float(torch.max(torch.abs(z.detach())).cpu()) <= hi:
        return "series", "all arguments remain inside the direct-series region"
    return "hybrid", "at least one argument enters the asymptotic or transition region"


def _branch_counts(alpha: torch.Tensor, z: torch.Tensor, method: str, threshold: float) -> dict[str, int]:
    total = z.numel()
    if method == "series":
        return {"series": total, "transition": 0, "asymptotic": 0}
    _, lo, hi = hybrid_switch_region(alpha, z, threshold=threshold)
    radius = torch.abs(z.detach())
    small = int(torch.count_nonzero(radius <= lo).cpu())
    large = int(torch.count_nonzero(radius >= hi).cpu())
    return {"series": small, "transition": total - small - large, "asymptotic": large}


def evaluate_mittag_leffler(
    alpha: torch.Tensor | float,
    z: torch.Tensor,
    *,
    beta: torch.Tensor | float | None = None,
    method: str = "auto",
    terms: int = 100,
    asymptotic_terms: int = 8,
    threshold: float = 12.0,
    estimate_error: bool = True,
    rtol: float | None = None,
    atol: float | None = None,
    custom_backward: bool = False,
    strict: bool = False,
) -> MittagLefflerEvaluation:
    """Evaluate ``E_alpha`` or ``E_{alpha,beta}`` with diagnostics.

    The returned tensor remains connected to the caller's autograd graph.  The
    optional embedded comparison is detached so diagnostics do not enlarge the
    training graph.
    """

    if not torch.is_tensor(z):
        raise TypeError("z must be a torch.Tensor")
    if torch.is_complex(z):
        raise ValueError("complex arguments are not implemented by the current dfsc evaluator")
    if not torch.is_floating_point(z):
        raise TypeError("z must have a floating-point dtype")
    if terms < 2 or asymptotic_terms < 1:
        raise ValueError("terms must be >= 2 and asymptotic_terms must be >= 1")

    alpha_t = _as_parameter(alpha, z)
    if bool(torch.any((alpha_t <= 0.0) | (alpha_t > 2.0)).item()):
        raise ValueError("the validated dfsc evaluator requires 0 < alpha <= 2")
    beta_t = None if beta is None else _as_parameter(beta, z)
    if beta_t is not None and bool(torch.any(beta_t <= 0.0).item()):
        raise ValueError("the validated two-parameter evaluator requires beta > 0")

    warnings: list[str] = []
    selection_reason = "method selected explicitly"
    selected = method
    if method == "auto":
        selected, selection_reason = _auto_method(alpha_t, z, threshold)
    if selected not in {"series", "hybrid"}:
        raise ValueError("method must be 'auto', 'series', or 'hybrid'")
    if selected == "hybrid" and bool(torch.any(z > 0).item()):
        raise ValueError("the hybrid evaluator supports non-positive real arguments only")
    if bool(torch.any(z > 0).item()):
        warnings.append("positive arguments are outside the validated stable negative-real regime")

    if beta_t is None:
        values = mittag_leffler_e(
            alpha_t,
            z,
            terms=terms,
            custom_backward=custom_backward if selected == "series" else False,
            method=selected,
        )
    else:
        values = mittag_leffler_e_ab(
            alpha_t,
            beta_t,
            z,
            terms=terms,
            method=selected,
            asymptotic_terms=asymptotic_terms,
            threshold=threshold,
        )

    finite = bool(torch.isfinite(values.detach()).all().item())
    default_rtol = 1e-10 if z.dtype == torch.float64 else 5e-5
    default_atol = 1e-12 if z.dtype == torch.float64 else 1e-6
    active_rtol = default_rtol if rtol is None else float(rtol)
    active_atol = default_atol if atol is None else float(atol)
    abs_disagreement: float | None = None
    rel_disagreement: float | None = None
    estimate_kind = "not-requested"
    converged = finite

    if estimate_error:
        richer_terms = max(terms + 16, int(math.ceil(1.25 * terms)))
        richer_asymptotic_terms = asymptotic_terms + 2
        with torch.no_grad():
            alpha_ref = alpha_t.detach()
            z_ref = z.detach()
            if beta_t is None:
                reference = mittag_leffler_e(alpha_ref, z_ref, terms=richer_terms, method=selected)
            else:
                reference = mittag_leffler_e_ab(
                    alpha_ref,
                    beta_t.detach(),
                    z_ref,
                    terms=richer_terms,
                    method=selected,
                    asymptotic_terms=richer_asymptotic_terms,
                    threshold=threshold,
                )
            delta = torch.abs(values.detach() - reference)
            scale = torch.maximum(torch.abs(reference), torch.full_like(reference, active_atol))
            abs_disagreement = float(torch.max(delta).cpu()) if delta.numel() else 0.0
            rel_disagreement = float(torch.max(delta / scale).cpu()) if delta.numel() else 0.0
            converged = finite and abs_disagreement <= active_atol + active_rtol * float(
                torch.max(torch.abs(reference)).cpu()
            )
        estimate_kind = "embedded-truncation-disagreement-not-a-rigorous-bound"
        if not converged:
            warnings.append("embedded evaluations disagree beyond the requested tolerance")

        if selected == "series":
            certified = alternating_series_remainder_bound(
                alpha_t,
                z,
                beta=1.0 if beta_t is None else beta_t,
                terms=terms,
            )
            if certified is not None:
                abs_disagreement = float(torch.max(certified.detach()).cpu()) if certified.numel() else 0.0
                reference_scale = float(torch.max(torch.abs(values.detach())).cpu()) if values.numel() else 0.0
                rel_disagreement = abs_disagreement / max(reference_scale, active_atol)
                converged = finite and abs_disagreement <= active_atol + active_rtol * reference_scale
                estimate_kind = "rigorous-alternating-series-remainder-bound"

    branches = _branch_counts(alpha_t, z, selected, threshold)
    if branches["asymptotic"] or branches["transition"]:
        warnings.append(
            "embedded truncation disagreement does not bound systematic error in the asymptotic model"
        )

    reliability = assess_mittag_leffler_reliability(
        alpha=alpha_t,
        beta=beta_t,
        z=z,
        method=selected,
        branch_counts=branches,
        finite=finite,
        converged=converged,
        relative_error_estimate=rel_disagreement,
        error_estimate_kind=estimate_kind,
    )
    if estimate_kind == "rigorous-alternating-series-remainder-bound":
        reliability = ReliabilityReport(
            **{
                **reliability.__dict__,
                "rigorous_error_bound": True,
            }
        )
    warnings.extend(reliability.messages)
    if strict and not reliability.trusted:
        raise RuntimeError(
            "Mittag-Leffler evaluation did not satisfy the dfsc reliability contract: "
            f"level={reliability.level}, converged={reliability.converged}"
        )

    return MittagLefflerEvaluation(
        values=values,
        method=selected,
        branch_counts=branches,
        finite=finite,
        converged=converged,
        reliability=reliability,
        estimated_absolute_disagreement=abs_disagreement,
        estimated_relative_disagreement=rel_disagreement,
        error_estimate_kind=estimate_kind,
        warnings=tuple(warnings),
        metadata={
            "selection_reason": selection_reason,
            "terms": terms,
            "asymptotic_terms": asymptotic_terms,
            "threshold": threshold,
            "rtol": active_rtol,
            "atol": active_atol,
            "two_parameter": beta_t is not None,
        },
    )


def evaluate_mittag_leffler_adaptive(
    alpha: torch.Tensor | float,
    z: torch.Tensor,
    *,
    beta: torch.Tensor | float | None = None,
    method: str = "auto",
    term_schedule: tuple[int, ...] = (24, 48, 80, 120, 180),
    asymptotic_terms: int = 8,
    threshold: float = 12.0,
    rtol: float | None = None,
    atol: float | None = None,
    strict: bool = False,
) -> AdaptiveMittagLefflerEvaluation:
    """Increase truncation depth until successive evaluations agree.

    The controller is designed for differentiable training loops: routing uses
    detached diagnostics, while ``values`` remains connected to the graph at
    the selected depth.  Consequently, the returned map is differentiable
    inside regions where the selected depth is unchanged; no continuity of the
    derivative is promised across a work-budget switch.  A fixed schedule also
    makes the computational budget reproducible across devices.
    """

    schedule = tuple(int(value) for value in term_schedule)
    if len(schedule) < 2 or any(value < 2 for value in schedule):
        raise ValueError("term_schedule must contain at least two entries >= 2")
    if any(right <= left for left, right in zip(schedule, schedule[1:])):
        raise ValueError("term_schedule must be strictly increasing")

    active_rtol = (1e-10 if z.dtype == torch.float64 else 5e-5) if rtol is None else float(rtol)
    active_atol = (1e-12 if z.dtype == torch.float64 else 1e-6) if atol is None else float(atol)
    previous: torch.Tensor | None = None
    relative_history: list[float] = []
    absolute_history: list[float] = []
    selected: MittagLefflerEvaluation | None = None
    converged = False

    for terms in schedule:
        current = evaluate_mittag_leffler(
            alpha,
            z,
            beta=beta,
            method=method,
            terms=terms,
            asymptotic_terms=asymptotic_terms,
            threshold=threshold,
            estimate_error=False,
            rtol=active_rtol,
            atol=active_atol,
        )
        selected = current
        if previous is not None:
            with torch.no_grad():
                delta = torch.abs(current.values.detach() - previous.detach())
                reference_scale = torch.maximum(
                    torch.abs(current.values.detach()),
                    torch.full_like(current.values.detach(), active_atol),
                )
                absolute = float(torch.max(delta).cpu()) if delta.numel() else 0.0
                relative = float(torch.max(delta / reference_scale).cpu()) if delta.numel() else 0.0
                tolerance = active_atol + active_rtol * float(
                    torch.max(torch.abs(current.values.detach())).cpu()
                )
            absolute_history.append(absolute)
            relative_history.append(relative)
            if current.finite and absolute <= tolerance:
                converged = True
                break
        previous = current.values

    assert selected is not None
    selected.converged = converged
    selected.estimated_absolute_disagreement = absolute_history[-1] if absolute_history else None
    selected.estimated_relative_disagreement = relative_history[-1] if relative_history else None
    selected.error_estimate_kind = "successive-truncation-disagreement-not-a-rigorous-bound"
    alpha_t = _as_parameter(alpha, z)
    beta_t = None if beta is None else _as_parameter(beta, z)
    selected.reliability = assess_mittag_leffler_reliability(
        alpha=alpha_t,
        beta=beta_t,
        z=z,
        method=selected.method,
        branch_counts=selected.branch_counts,
        finite=selected.finite,
        converged=converged,
        relative_error_estimate=selected.estimated_relative_disagreement,
        error_estimate_kind=selected.error_estimate_kind,
    )
    selected.metadata.update(
        {
            "rtol": active_rtol,
            "atol": active_atol,
            "terms": selected.metadata.get("terms"),
        }
    )
    if not converged:
        selected.warnings = (*selected.warnings, "adaptive term schedule exhausted before tolerance was met")
    if strict and not converged:
        raise RuntimeError("adaptive Mittag-Leffler term schedule exhausted before tolerance was met")
    attempted = schedule[: schedule.index(int(selected.metadata["terms"])) + 1]
    return AdaptiveMittagLefflerEvaluation(
        values=selected.values,
        converged=converged,
        selected_terms=int(selected.metadata["terms"]),
        attempted_terms=attempted,
        relative_disagreements=tuple(relative_history),
        absolute_disagreements=tuple(absolute_history),
        evaluation=selected,
    )
