"""History-aware fallback algorithms for dfsc."""

from __future__ import annotations

import torch


def caputo_l1_linear_solve(
    operator: torch.Tensor,
    u0: torch.Tensor,
    *,
    alpha: torch.Tensor | float,
    final_time: float,
    num_steps: int,
    forcing: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``D_t^alpha u + A u = f`` with the implicit L1 scheme.

    This differentiable fallback targets constant-order Caputo systems with
    ``0 < alpha < 1`` on a uniform time grid.  It stores the full history and
    is intentionally distinct from the direct-query MLSL path.
    """

    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    if not torch.is_tensor(operator):
        operator = torch.as_tensor(operator, dtype=u0.dtype, device=u0.device)
    operator = operator.to(dtype=u0.dtype, device=u0.device)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be a square matrix")
    if u0.ndim < 1 or u0.shape[-1] != operator.shape[0]:
        raise ValueError("u0 last dimension must match the operator size")
    if not torch.isfinite(operator).all().item() or not torch.isfinite(u0).all().item():
        raise ValueError("operator and u0 must be finite")

    alpha_t = torch.as_tensor(alpha, dtype=u0.dtype, device=u0.device)
    if alpha_t.numel() != 1:
        raise ValueError("CaputoL1 currently requires a scalar constant alpha")
    if not bool(((alpha_t > 0.0) & (alpha_t < 1.0)).item()):
        raise ValueError("CaputoL1 requires 0 < alpha < 1")
    if forcing is not None:
        forcing = forcing.to(dtype=u0.dtype, device=u0.device)
        expected = u0.shape[:-1] + (num_steps + 1, operator.shape[0])
        unbatched_expected = (num_steps + 1, operator.shape[0])
        if forcing.shape not in {expected, unbatched_expected}:
            raise ValueError("forcing must have shape (..., num_steps + 1, state_size)")

    times = torch.linspace(0.0, final_time, num_steps + 1, dtype=u0.dtype, device=u0.device)
    dt = torch.as_tensor(final_time / num_steps, dtype=u0.dtype, device=u0.device)
    j = torch.arange(num_steps, dtype=u0.dtype, device=u0.device)
    weights = (j + 1.0).pow(1.0 - alpha_t) - j.pow(1.0 - alpha_t)
    scale = dt.pow(-alpha_t) / torch.exp(torch.lgamma(2.0 - alpha_t))
    system = scale * torch.eye(operator.shape[0], dtype=u0.dtype, device=u0.device) + operator

    states = [u0]
    for n in range(1, num_steps + 1):
        if n == 1:
            history = torch.zeros_like(u0)
        else:
            increments = torch.stack(
                [states[n - offset] - states[n - offset - 1] for offset in range(1, n)],
                dim=-2,
            )
            weight_shape = (1,) * (increments.ndim - 2) + (n - 1, 1)
            history = torch.sum(weights[1:n].reshape(weight_shape) * increments, dim=-2)
        rhs = scale * states[-1] - scale * history
        if forcing is not None:
            rhs = rhs + forcing[..., n, :]
        next_state = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
        states.append(next_state)
    return torch.stack(states, dim=-2), times
