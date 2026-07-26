"""Differentiable Mittag-Leffler spectral layer for dfsc."""

from __future__ import annotations

import torch
from torch import nn

from .mittag_leffler import mittag_leffler_e


class MittagLefflerSpectralLayer(nn.Module):
    """History-free spectral layer for homogeneous fractional dynamics.

    The layer computes

    ``u(t) = Phi diag(E_alpha(-mu_n t^alpha)) Phi.T u0``

    where ``mu_n = c^2 lambda_n^(beta/2)``.
    """

    def __init__(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        *,
        projection_vectors: torch.Tensor | None = None,
        terms: int = 80,
        wave_speed: float = 1.0,
        beta: float = 2.0,
        custom_backward: bool = True,
        ml_method: str = "series",
    ) -> None:
        super().__init__()
        if eigenvectors.ndim != 2:
            raise ValueError("eigenvectors must have shape (num_points, num_modes)")
        if eigenvalues.ndim != 1:
            raise ValueError("eigenvalues must have shape (num_modes,)")
        if eigenvectors.shape[1] != eigenvalues.shape[0]:
            raise ValueError("eigenvectors columns must match eigenvalues")
        if projection_vectors is not None and projection_vectors.shape != eigenvectors.shape:
            raise ValueError("projection_vectors must match eigenvectors")

        self.register_buffer("eigenvalues", eigenvalues.detach().clone())
        self.register_buffer("eigenvectors", eigenvectors.detach().clone())
        projection = eigenvectors if projection_vectors is None else projection_vectors
        self.register_buffer("projection_vectors", projection.detach().clone())
        self.terms = int(terms)
        self.wave_speed = float(wave_speed)
        self.beta = float(beta)
        self.custom_backward = bool(custom_backward)
        self.ml_method = ml_method

    def modal_rates(
        self,
        beta: torch.Tensor | float | None = None,
        eigenvalues: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``mu_n = c^2 lambda_n^(beta/2)``."""

        if eigenvalues is None:
            eigenvalues = self.eigenvalues

        if beta is None:
            beta = torch.as_tensor(
                self.beta,
                dtype=eigenvalues.dtype,
                device=eigenvalues.device,
            )
        elif not torch.is_tensor(beta):
            beta = torch.as_tensor(
                beta,
                dtype=eigenvalues.dtype,
                device=eigenvalues.device,
            )
        else:
            beta = beta.to(dtype=eigenvalues.dtype, device=eigenvalues.device)

        return (self.wave_speed**2) * eigenvalues.pow(beta / 2.0)

    def forward(
        self,
        u0: torch.Tensor,
        t: torch.Tensor | float,
        alpha: torch.Tensor | float,
        *,
        beta: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Evaluate the layer.

        Parameters
        ----------
        u0:
            Tensor with shape ``(..., num_points)``.
        t:
            Scalar or tensor of query times. If a vector of length ``T`` is
            passed and ``u0`` is one field, the output has shape
            ``(T, num_points)``.
        alpha:
            Fractional order. May require gradients.
        beta:
            Optional spatial order. May require gradients.
        """

        phi = self.eigenvectors.to(dtype=u0.dtype, device=u0.device)
        projection = self.projection_vectors.to(dtype=u0.dtype, device=u0.device)
        eigenvalues = self.eigenvalues.to(dtype=u0.dtype, device=u0.device)
        mu = self.modal_rates(beta, eigenvalues)

        if not torch.is_tensor(alpha):
            alpha = torch.as_tensor(alpha, dtype=u0.dtype, device=u0.device)
        else:
            alpha = alpha.to(dtype=u0.dtype, device=u0.device)

        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=u0.dtype, device=u0.device)
        else:
            t = t.to(dtype=u0.dtype, device=u0.device)

        coeffs = torch.matmul(u0, projection)
        tiny = torch.finfo(t.dtype).tiny
        t_safe = t.clamp_min(tiny)
        t_alpha = torch.where(t > 0, t_safe.pow(alpha), torch.zeros_like(t))
        z = -mu * t_alpha.unsqueeze(-1) if t.ndim > 0 else -mu * t_alpha
        kernel = mittag_leffler_e(
            alpha,
            z,
            terms=self.terms,
            custom_backward=self.custom_backward,
            method=self.ml_method,
        )

        evolved = coeffs.unsqueeze(-2) * kernel if t.ndim > 0 else coeffs * kernel
        return torch.matmul(evolved, phi.transpose(-1, -2))
