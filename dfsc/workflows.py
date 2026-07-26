"""Reusable workflow helpers for dfsc."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .mittag_leffler import mittag_leffler_e


@dataclass
class TrainableOrders:
    """Bounded trainable fractional orders for inverse problems."""

    raw_alpha: torch.Tensor
    raw_beta: torch.Tensor | None = None
    alpha_bounds: tuple[float, float] = (0.2, 1.8)
    beta_bounds: tuple[float, float] = (0.2, 2.5)

    @property
    def alpha(self) -> torch.Tensor:
        lo, hi = self.alpha_bounds
        return lo + (hi - lo) * torch.sigmoid(self.raw_alpha)

    @property
    def beta(self) -> torch.Tensor | None:
        if self.raw_beta is None:
            return None
        lo, hi = self.beta_bounds
        return lo + (hi - lo) * torch.sigmoid(self.raw_beta)

    def parameters(self) -> list[torch.Tensor]:
        params = [self.raw_alpha]
        if self.raw_beta is not None:
            params.append(self.raw_beta)
        return params


def make_trainable_orders(
    *,
    alpha_init: float = 0.9,
    beta_init: float | None = 1.5,
    alpha_bounds: tuple[float, float] = (0.2, 1.8),
    beta_bounds: tuple[float, float] = (0.2, 2.5),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> TrainableOrders:
    """Create bounded trainable alpha/beta parameters.

    The raw variables are unconstrained tensors. The public ``alpha`` and
    ``beta`` properties map them into physically meaningful intervals.
    """

    def inverse_sigmoid(value: float, bounds: tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        clipped = min(max((value - lo) / (hi - lo), 1e-6), 1.0 - 1e-6)
        raw = torch.logit(torch.tensor(clipped, dtype=dtype, device=device))
        return raw.detach().clone().requires_grad_(True)

    raw_alpha = inverse_sigmoid(alpha_init, alpha_bounds)
    raw_beta = None if beta_init is None else inverse_sigmoid(beta_init, beta_bounds)
    return TrainableOrders(
        raw_alpha=raw_alpha,
        raw_beta=raw_beta,
        alpha_bounds=alpha_bounds,
        beta_bounds=beta_bounds,
    )


class HybridResidualModel(nn.Module):
    """Compose a known dfsc backbone with a trainable residual head."""

    def __init__(self, backbone: nn.Module, residual_head: nn.Module, *, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.residual_head = residual_head
        self.residual_scale = residual_scale

    def forward(
        self,
        u0: torch.Tensor,
        times: torch.Tensor,
        alpha: torch.Tensor,
        *,
        beta: torch.Tensor | None = None,
        residual_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.backbone(u0, times, alpha, beta=beta)
        if residual_features is None:
            residual_features = base
        residual = self.residual_head(residual_features)
        return base + self.residual_scale * residual


class MittagLefflerResidualRegressor(nn.Module):
    """Scalar Mittag-Leffler relaxation with a gated neural residual."""

    def __init__(
        self,
        residual_head: nn.Module,
        *,
        alpha_init: float = 0.9,
        rate_init: float = 0.1,
        alpha_bounds: tuple[float, float] = (0.2, 1.6),
        residual_scale: float = 0.35,
        terms: int = 120,
    ) -> None:
        super().__init__()
        alpha_lo, alpha_hi = alpha_bounds
        if not alpha_lo < alpha_init < alpha_hi:
            raise ValueError("alpha_init must lie strictly inside alpha_bounds")
        if rate_init <= 0.0:
            raise ValueError("rate_init must be positive")
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be nonnegative")
        fraction = (alpha_init - alpha_lo) / (alpha_hi - alpha_lo)
        self.raw_alpha = nn.Parameter(torch.logit(torch.tensor(fraction)))
        self.raw_rate = nn.Parameter(torch.log(torch.expm1(torch.tensor(rate_init))))
        self.residual_head = residual_head
        self.alpha_bounds = alpha_bounds
        self.residual_scale = float(residual_scale)
        self.terms = int(terms)

    @property
    def alpha(self) -> torch.Tensor:
        lo, hi = self.alpha_bounds
        return lo + (hi - lo) * torch.sigmoid(self.raw_alpha)

    @property
    def rate(self) -> torch.Tensor:
        return F.softplus(self.raw_rate)

    def base_prediction(self, times: torch.Tensor) -> torch.Tensor:
        z = -self.rate * times.pow(self.alpha)
        return mittag_leffler_e(
            self.alpha,
            z,
            terms=self.terms,
            custom_backward=False,
            method="hybrid",
        )

    def forward(self, times: torch.Tensor, residual_features: torch.Tensor) -> torch.Tensor:
        base = self.base_prediction(times)
        correction = self.residual_head(residual_features)
        if correction.shape[-1:] == (1,):
            correction = correction.squeeze(-1)
        if correction.shape != base.shape:
            raise ValueError("residual head output must match the time shape")
        gated = (1.0 - base) * self.residual_scale * torch.tanh(correction)
        return base + gated


def relative_l2_error(prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-14) -> torch.Tensor:
    """Return relative L2 error with a small denominator guard."""

    return torch.linalg.norm(prediction - target) / torch.linalg.norm(target).clamp_min(eps)
