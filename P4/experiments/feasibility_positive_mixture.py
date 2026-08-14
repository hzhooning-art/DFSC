"""Minimal P4 feasibility test for a constrained learnable memory kernel."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def relative_l2(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y)).detach())


class PositiveExponentialMixture(torch.nn.Module):
    """Causal non-negative kernel with positive normalized spectral weights."""

    def __init__(self, modes: int, init_rates: torch.Tensor) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(modes, device=init_rates.device))
        self.log_rates = torch.nn.Parameter(torch.log(init_rates))

    def kernel(self, t: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.logits, dim=0)
        rates = torch.nn.functional.softplus(self.log_rates)
        return (weights[:, None] * rates[:, None] * torch.exp(-rates[:, None] * t[None, :])).sum(0)

    def convolve(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dt = t[1] - t[0]
        lag = t[:, None] - t[None, :]
        # Midpoint sampling keeps the learned kernel and reference quadrature
        # on the same discrete causal convention.
        causal_kernel = self._kernel_at((lag.clamp_min(0.0) + 0.5 * dt)) * (lag >= 0).to(t.dtype)
        return dt * (causal_kernel @ u)

    def _kernel_at(self, t: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.logits, dim=0)
        rates = torch.nn.functional.softplus(self.log_rates)
        return (weights[:, None, None] * rates[:, None, None] * torch.exp(-rates[:, None, None] * t[None, :, :])).sum(0)


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 256
    t = torch.linspace(0.0, 40.0, n, device=device)
    dt = t[1] - t[0]
    tau = t + 0.5 * dt
    alpha = 0.62
    temper = 0.045
    target = tau.pow(-alpha) * torch.exp(-temper * tau)
    target_mass = target.sum() * dt
    target = target / target_mass
    u = torch.sin(0.55 * t) + 0.35 * torch.cos(0.13 * t) + torch.exp(-0.5 * ((t - 2.0) / 0.25) ** 2)
    lag = t[:, None] - t[None, :]
    lag_pos = lag.clamp_min(0.0)
    y_target = dt * ((lag >= 0).to(t.dtype) * (lag_pos + 0.5 * dt).pow(-alpha) * torch.exp(-temper * (lag_pos + 0.5 * dt)) / target_mass @ u)

    rates = torch.logspace(math.log10(0.03), math.log10(3.0), 12, device=device)
    model = PositiveExponentialMixture(modes=12, init_rates=rates).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.035)
    train = t <= 28.0
    for _ in range(1000):
        opt.zero_grad(set_to_none=True)
        pred = model.kernel(tau)
        loss = torch.mean((pred[train] - target[train]) ** 2)
        loss.backward()
        opt.step()

    learned_kernel = model.kernel(tau)
    learned_output = model.convolve(u, t)
    holdout = t > 28.0
    positivity = float(learned_kernel.detach().min())
    normalization = float((learned_kernel.sum() * dt).detach())
    grad_probe = torch.autograd.grad(learned_output[-1], tuple(model.parameters()), retain_graph=False)
    finite_grad = all(torch.isfinite(g).all().item() for g in grad_probe)
    result = {
        "device": str(device),
        "modes": 12,
        "target_alpha": alpha,
        "target_tempering": temper,
        "kernel_relative_l2_full": relative_l2(learned_kernel, target),
        "kernel_relative_l2_holdout": relative_l2(learned_kernel[holdout], target[holdout]),
        "convolution_relative_l2_full": relative_l2(learned_output, y_target),
        "convolution_relative_l2_holdout": relative_l2(learned_output[holdout], y_target[holdout]),
        "learned_kernel_min": positivity,
        "learned_kernel_mass": normalization,
        "finite_endpoint_gradients": bool(finite_grad),
        "feasibility": bool(finite_grad and positivity >= -1e-7),
        "interpretation": "Route-feasibility evidence only; not a complete P4 validation.",
    }
    out = Path(__file__).parents[1] / "results" / "p4_feasibility_positive_mixture.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
