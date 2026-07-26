"""Mild-form Picard iteration for semilinear dfsc spectral problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .factory import MLSLConfig
from .forced_layer import ForcedMittagLefflerSpectralLayer
from .spectral_layer import MittagLefflerSpectralLayer


@dataclass(frozen=True)
class PicardDiagnostics:
    """Convergence information for a semilinear mild-form iteration."""

    iterations: int
    residual: float
    converged: bool
    residual_history: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "picard_iterations": self.iterations,
            "picard_residual": self.residual,
            "picard_converged": self.converged,
            "picard_residual_history": list(self.residual_history),
        }


def _interpolate_states(times: torch.Tensor, states: torch.Tensor, query_times: torch.Tensor) -> torch.Tensor:
    if times.numel() == 1:
        return states[0].expand(query_times.numel(), -1)
    indices = torch.searchsorted(times.detach(), query_times.detach(), right=True)
    right = indices.clamp(1, times.numel() - 1)
    left = right - 1
    t_left = times[left]
    t_right = times[right]
    weight = (query_times - t_left) / (t_right - t_left).clamp_min(torch.finfo(times.dtype).tiny)
    return states[left] + weight[:, None] * (states[right] - states[left])


def semilinear_mild_picard(
    layer: MittagLefflerSpectralLayer,
    u0: torch.Tensor,
    times: torch.Tensor,
    alpha: torch.Tensor | float,
    nonlinearity: Callable[[torch.Tensor], torch.Tensor],
    *,
    beta: torch.Tensor | float | None = None,
    max_iterations: int = 30,
    tolerance: float = 1e-7,
    relaxation: float = 1.0,
    quadrature_points: int = 48,
    forcing_terms: int = 100,
    config: MLSLConfig | None = None,
) -> tuple[torch.Tensor, PicardDiagnostics]:
    """Solve a semilinear fractional evolution by mild-form Picard iteration.

    This first implementation is intentionally scoped to one unbatched state,
    monotonically increasing query times beginning at zero, and a known MLSL
    spectral backbone. It is an iterative nonlinear extension, not a general
    convergence guarantee for arbitrary nonlinearities.
    """

    if u0.ndim != 1:
        raise ValueError("semilinear Picard currently expects one unbatched initial state")
    if times.ndim != 1 or times.numel() < 1:
        raise ValueError("times must be a nonempty one-dimensional tensor")
    times = times.to(dtype=u0.dtype, device=u0.device)
    if float(times[0].detach().cpu()) != 0.0 or bool(torch.any(times[1:] <= times[:-1]).item()):
        raise ValueError("times must be strictly increasing and begin at zero")
    if max_iterations < 1 or quadrature_points < 1:
        raise ValueError("max_iterations and quadrature_points must be positive")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must lie in (0, 1]")

    cfg = MLSLConfig.stable(terms=layer.terms) if config is None else config
    base_layer = MittagLefflerSpectralLayer(
        layer.eigenvalues,
        layer.eigenvectors,
        projection_vectors=layer.projection_vectors,
        terms=cfg.terms,
        wave_speed=cfg.wave_speed,
        beta=cfg.beta,
        custom_backward=cfg.custom_backward,
        ml_method=cfg.ml_method,
    ).to(device=u0.device, dtype=u0.dtype)
    forced_layer = ForcedMittagLefflerSpectralLayer(
        base_layer,
        quadrature_points=quadrature_points,
        forcing_terms=forcing_terms,
        ml_method=cfg.ml_method,
    )
    normalized_nodes = (torch.arange(quadrature_points, dtype=u0.dtype, device=u0.device) + 0.5) / quadrature_points
    state = base_layer(u0, times, alpha, beta=beta)
    residual_history: list[float] = []
    converged = False

    for _ in range(max_iterations):
        forcing_rows = []
        for time_value in times:
            sampled = _interpolate_states(times, state, time_value * normalized_nodes)
            nonlinear_values = nonlinearity(sampled)
            if nonlinear_values.shape != sampled.shape:
                raise ValueError("nonlinearity must preserve the state shape")
            forcing_rows.append(nonlinear_values)
        forcing_values = torch.stack(forcing_rows, dim=0)
        candidate = forced_layer(
            u0,
            times,
            alpha,
            forcing_values,
            normalized_nodes,
            beta=beta,
        )
        updated = relaxation * candidate + (1.0 - relaxation) * state
        denominator = torch.linalg.vector_norm(updated.detach()).clamp_min(torch.finfo(u0.dtype).eps)
        residual_t = torch.linalg.vector_norm((updated - state).detach()) / denominator
        residual = float(residual_t.cpu())
        residual_history.append(residual)
        state = updated
        if residual <= tolerance:
            converged = True
            break

    diagnostics = PicardDiagnostics(len(residual_history), residual_history[-1], converged, tuple(residual_history))
    return state, diagnostics
