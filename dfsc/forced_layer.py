"""Forced extensions of the dfsc Mittag-Leffler spectral layer."""

from __future__ import annotations

import torch
from torch import nn

from .mittag_leffler import mittag_leffler_e_ab
from .spectral_layer import MittagLefflerSpectralLayer


class ForcedMittagLefflerSpectralLayer(nn.Module):
    """Spectral layer for linear fractional dynamics with additive forcing.

    It evaluates a midpoint quadrature approximation of

    ``u(t) = S_alpha(t)u0 + integral_0^t (t-s)^(alpha-1)
    E_{alpha,alpha}(-mu (t-s)^alpha) f(s) ds``.
    """

    def __init__(
        self,
        base_layer: MittagLefflerSpectralLayer,
        *,
        quadrature_points: int = 64,
        forcing_terms: int = 80,
        ml_method: str = "hybrid",
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.quadrature_points = int(quadrature_points)
        self.forcing_terms = int(forcing_terms)
        self.ml_method = ml_method

    def forward(
        self,
        u0: torch.Tensor,
        t: torch.Tensor,
        alpha: torch.Tensor | float,
        forcing_values: torch.Tensor,
        forcing_times: torch.Tensor,
        *,
        beta: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Evaluate the forced layer.

        ``forcing_values`` may have shape ``(Q, num_points)`` for time-invariant
        forcing samples or ``(T, Q, num_points)`` for query-time-specific samples
        such as ``f(t_i * forcing_times_q, x)``.
        """

        if t.ndim != 1:
            raise ValueError("forced layer currently expects vector query times")
        if forcing_values.ndim not in (2, 3):
            raise ValueError("forcing_values must have shape (Q, N) or (T, Q, N)")
        if forcing_times.ndim != 1:
            raise ValueError("forcing_times must have shape (Q,)")
        if forcing_values.ndim == 2 and forcing_values.shape[0] != forcing_times.numel():
            raise ValueError("forcing_values and forcing_times disagree")
        if forcing_values.ndim == 3 and (
            forcing_values.shape[0] != t.numel() or forcing_values.shape[1] != forcing_times.numel()
        ):
            raise ValueError("time-specific forcing_values must have shape (T, Q, N)")

        homogeneous = self.base_layer(u0, t, alpha, beta=beta)
        if not torch.is_tensor(alpha):
            alpha = torch.as_tensor(alpha, dtype=u0.dtype, device=u0.device)
        else:
            alpha = alpha.to(dtype=u0.dtype, device=u0.device)

        phi = self.base_layer.eigenvectors.to(dtype=u0.dtype, device=u0.device)
        eigenvalues = self.base_layer.eigenvalues.to(dtype=u0.dtype, device=u0.device)
        mu = self.base_layer.modal_rates(beta, eigenvalues)
        forcing_times = forcing_times.to(dtype=u0.dtype, device=u0.device)
        forcing_values = forcing_values.to(dtype=u0.dtype, device=u0.device)

        forced_states = []
        for time_index, time_value in enumerate(t):
            if float(time_value.detach()) == 0.0:
                forced_states.append(torch.zeros_like(u0))
                continue
            values = forcing_values if forcing_values.ndim == 2 else forcing_values[time_index]
            f_hat = torch.matmul(values, phi)
            tau = time_value * (1.0 - forcing_times)
            weights = time_value / forcing_times.numel()
            kernel = tau.clamp_min(torch.finfo(u0.dtype).tiny).pow(alpha - 1.0)
            z = -mu[None, :] * tau[:, None].pow(alpha)
            eaa = mittag_leffler_e_ab(
                alpha,
                alpha,
                z,
                terms=self.forcing_terms,
                method=self.ml_method,
            )
            modal = weights * torch.sum(kernel[:, None] * eaa * f_hat, dim=0)
            forced_states.append(torch.matmul(modal, phi.transpose(-1, -2)))

        return homogeneous + torch.stack(forced_states, dim=0)


def modal_forced_reference(
    base_layer: MittagLefflerSpectralLayer,
    u0: torch.Tensor,
    t: torch.Tensor,
    alpha: torch.Tensor,
    forcing_values: torch.Tensor,
    forcing_times: torch.Tensor,
    *,
    beta: torch.Tensor | float | None = None,
    forcing_terms: int = 120,
) -> torch.Tensor:
    """Convenience wrapper used by experiments for high-quadrature references."""

    layer = ForcedMittagLefflerSpectralLayer(
        base_layer,
        quadrature_points=forcing_times.numel(),
        forcing_terms=forcing_terms,
        ml_method="hybrid",
    )
    return layer(u0, t, alpha, forcing_values, forcing_times, beta=beta)
