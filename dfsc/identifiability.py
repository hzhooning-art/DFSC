"""Local identifiability diagnostics for differentiable inverse problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class IdentifiabilityReport:
    hessian: torch.Tensor
    eigenvalues: torch.Tensor
    condition_number: float
    covariance: torch.Tensor
    standard_errors: torch.Tensor
    correlation: torch.Tensor
    locally_identifiable: bool
    rank: int

    def to_dict(self) -> dict[str, object]:
        return {
            "hessian": self.hessian.detach().cpu().tolist(),
            "eigenvalues": self.eigenvalues.detach().cpu().tolist(),
            "condition_number": self.condition_number,
            "covariance": self.covariance.detach().cpu().tolist(),
            "standard_errors": self.standard_errors.detach().cpu().tolist(),
            "correlation": self.correlation.detach().cpu().tolist(),
            "locally_identifiable": self.locally_identifiable,
            "rank": self.rank,
        }


def local_identifiability(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    parameters: torch.Tensor,
    *,
    noise_variance: float = 1.0,
    rcond: float = 1e-10,
) -> IdentifiabilityReport:
    """Assess local curvature using an autograd Hessian and pseudo-inverse.

    This is a local, model-conditional diagnostic.  It does not establish
    global identifiability or correct model specification.
    """

    if parameters.ndim != 1:
        raise ValueError("parameters must be a one-dimensional tensor")
    if noise_variance < 0:
        raise ValueError("noise_variance must be nonnegative")
    hessian = torch.autograd.functional.hessian(loss_fn, parameters, create_graph=False)
    hessian = 0.5 * (hessian + hessian.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(hessian)
    scale = float(torch.max(torch.abs(eigenvalues)).detach().cpu()) if eigenvalues.numel() else 0.0
    threshold = rcond * max(scale, 1.0)
    positive = eigenvalues > threshold
    rank = int(torch.count_nonzero(positive).detach().cpu())
    positive_values = eigenvalues[positive]
    condition = (
        float((positive_values.max() / positive_values.min()).detach().cpu())
        if rank == parameters.numel() and rank > 0
        else float("inf")
    )
    covariance = float(noise_variance) * torch.linalg.pinv(hessian, rcond=rcond)
    variances = torch.diagonal(covariance).clamp_min(0.0)
    standard_errors = torch.sqrt(variances)
    denominator = standard_errors[:, None] * standard_errors[None, :]
    correlation = torch.where(denominator > 0, covariance / denominator, torch.zeros_like(covariance))
    return IdentifiabilityReport(
        hessian=hessian,
        eigenvalues=eigenvalues,
        condition_number=condition,
        covariance=covariance,
        standard_errors=standard_errors,
        correlation=correlation,
        locally_identifiable=rank == parameters.numel() and bool(torch.all(eigenvalues > 0).item()),
        rank=rank,
    )
