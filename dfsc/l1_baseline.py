"""Differentiable L1 time-marching baseline for Caputo relaxation.

This module is intentionally small and transparent. It is not meant to be the
fastest fractional solver; it is a baseline that exposes the history-dependent
computation graph that MLSL avoids.
"""

from __future__ import annotations

import torch


def l1_caputo_relaxation(
    u0: torch.Tensor | float,
    *,
    alpha: torch.Tensor | float,
    mu: torch.Tensor | float,
    final_time: float,
    num_steps: int,
) -> torch.Tensor:
    """Solve ``D_t^alpha u + mu u = 0`` with an implicit L1 scheme.

    This implementation targets ``0 < alpha < 1``. It returns all states
    ``u_0, ..., u_N`` and keeps the full history in the computation graph.
    """

    if num_steps < 1:
        raise ValueError("num_steps must be positive")

    if not torch.is_tensor(u0):
        u0 = torch.as_tensor(u0, dtype=torch.float64)
    dtype = u0.dtype
    device = u0.device

    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha, dtype=dtype, device=device)
    else:
        alpha = alpha.to(dtype=dtype, device=device)

    if not torch.is_tensor(mu):
        mu = torch.as_tensor(mu, dtype=dtype, device=device)
    else:
        mu = mu.to(dtype=dtype, device=device)

    dt = torch.as_tensor(final_time / num_steps, dtype=dtype, device=device)
    j = torch.arange(num_steps, dtype=dtype, device=device)
    weights = (j + 1.0).pow(1.0 - alpha) - j.pow(1.0 - alpha)
    scale = dt.pow(-alpha) / torch.exp(torch.lgamma(2.0 - alpha))

    states = [u0]
    for n in range(1, num_steps + 1):
        if n == 1:
            history = torch.zeros_like(u0)
        else:
            terms = [
                weights[j_idx] * (states[n - j_idx] - states[n - j_idx - 1])
                for j_idx in range(1, n)
            ]
            history = torch.stack(terms).sum(dim=0)

        numerator = scale * states[-1] - scale * history
        states.append(numerator / (scale + mu))

    return torch.stack(states, dim=0)


def l1_caputo_derivative_uniform(
    values: torch.Tensor,
    *,
    alpha: torch.Tensor,
    final_time: float,
) -> torch.Tensor:
    """Approximate the Caputo derivative on a uniform grid with L1 weights.

    Parameters
    ----------
    values:
        Tensor with shape ``(N + 1,)`` containing ``u(t_n)``.
    alpha:
        Fractional order in ``(0, 1)``.
    final_time:
        End time of the uniform grid.

    Returns
    -------
    Tensor with shape ``(N,)`` approximating ``D_t^alpha u(t_n)`` for
    ``n = 1, ..., N``.
    """

    if values.ndim != 1:
        raise ValueError("values must be a 1D tensor")
    num_steps = values.numel() - 1
    if num_steps < 1:
        raise ValueError("values must contain at least two time points")

    alpha = alpha.to(dtype=values.dtype, device=values.device)
    dt = torch.as_tensor(final_time / num_steps, dtype=values.dtype, device=values.device)
    j = torch.arange(num_steps, dtype=values.dtype, device=values.device)
    weights = (j + 1.0).pow(1.0 - alpha) - j.pow(1.0 - alpha)
    scale = dt.pow(-alpha) / torch.exp(torch.lgamma(2.0 - alpha))

    derivatives = []
    for n in range(1, num_steps + 1):
        increments = torch.stack([values[n - j_idx] - values[n - j_idx - 1] for j_idx in range(n)])
        derivatives.append(scale * torch.sum(weights[:n] * increments))
    return torch.stack(derivatives)
