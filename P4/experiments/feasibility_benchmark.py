"""P4 practicality probe: constrained kernel learning versus simple baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def rel(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y)).detach())


class PositiveMixture(torch.nn.Module):
    def __init__(self, modes: int, device: torch.device) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(modes, device=device))
        self.log_rates = torch.nn.Parameter(torch.linspace(math.log(0.02), math.log(4.0), modes, device=device))

    def __call_kernel(self, t: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.logits, dim=0)
        r = torch.nn.functional.softplus(self.log_rates)
        return (w[:, None] * r[:, None] * torch.exp(-r[:, None] * t[None, :])).sum(0)

    def kernel(self, t: torch.Tensor) -> torch.Tensor:
        return self.__call_kernel(t)


def causal_apply(kernel_values: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    n = u.numel()
    lag = torch.arange(n, device=u.device)[:, None] - torch.arange(n, device=u.device)[None, :]
    mask = (lag >= 0).to(u.dtype)
    return dt * ((mask * kernel_values[lag.clamp_min(0)]).tril() @ u)


def target_kernel(tau: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "tempered_power_law":
        raw = tau.pow(-0.62) * torch.exp(-0.045 * tau)
    elif kind == "two_timescale":
        raw = 0.75 * 0.12 * torch.exp(-0.12 * tau) + 0.25 * 2.5 * torch.exp(-2.5 * tau)
    else:
        raise ValueError(kind)
    return raw / (raw.sum() * (tau[1] - tau[0]))


def fit_case(kind: str, seed: int, device: torch.device) -> dict[str, float | str | bool]:
    torch.manual_seed(seed)
    n = 192
    t = torch.linspace(0.0, 32.0, n, device=device)
    dt = t[1] - t[0]
    tau = t + 0.5 * dt
    clean = target_kernel(tau, kind)
    noise = 0.025 * clean.max() * torch.randn_like(clean)
    observed = clean + noise
    observed = observed.clamp_min(0.0)
    mask = (t <= 20.0) & ((torch.arange(n, device=device) % 2) == 0)
    model = PositiveMixture(12, device)
    opt = torch.optim.Adam(model.parameters(), lr=0.04)
    for _ in range(700):
        opt.zero_grad(set_to_none=True)
        pred = model.kernel(tau)
        loss = ((pred[mask] - observed[mask]) ** 2).mean()
        loss.backward()
        opt.step()
    learned = model.kernel(tau)

    rate = torch.nn.Parameter(torch.tensor(math.log(0.3), device=device))
    opt_single = torch.optim.Adam([rate], lr=0.04)
    for _ in range(700):
        opt_single.zero_grad(set_to_none=True)
        single = torch.nn.functional.softplus(rate) * torch.exp(-torch.nn.functional.softplus(rate) * tau)
        single = single / (single.sum() * dt)
        (((single[mask] - observed[mask]) ** 2).mean()).backward()
        opt_single.step()
    single = torch.nn.functional.softplus(rate) * torch.exp(-torch.nn.functional.softplus(rate) * tau)
    single = single / (single.sum() * dt)

    u = torch.sin(0.45 * t) + 0.3 * torch.cos(0.17 * t) + torch.exp(-0.5 * ((t - 3.0) / 0.3) ** 2)
    y_ref = causal_apply(clean, u, dt)
    y_mix = causal_apply(learned, u, dt)
    y_single = causal_apply(single, u, dt)
    hold = t > 20.0
    grad = torch.autograd.grad(y_mix[-1], tuple(model.parameters()), retain_graph=False)
    return {
        "kind": kind,
        "seed": seed,
        "kernel_error_mixture": rel(learned, clean),
        "kernel_error_single": rel(single, clean),
        "kernel_holdout_error_mixture": rel(learned[hold], clean[hold]),
        "kernel_holdout_error_single": rel(single[hold], clean[hold]),
        "convolution_error_mixture": rel(y_mix, y_ref),
        "convolution_error_single": rel(y_single, y_ref),
        "convolution_holdout_error_mixture": rel(y_mix[hold], y_ref[hold]),
        "convolution_holdout_error_single": rel(y_single[hold], y_ref[hold]),
        "positive_min": float(learned.detach().min()),
        "finite_gradients": bool(all(torch.isfinite(g).all().item() for g in grad)),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [fit_case(kind, seed, device) for kind in ("tempered_power_law", "two_timescale") for seed in (3, 7, 11)]
    summary = {}
    for kind in ("tempered_power_law", "two_timescale"):
        group = [r for r in rows if r["kind"] == kind]
        summary[kind] = {}
        for key in rows[0]:
            if key in {"kind", "seed"}:
                continue
            values = torch.tensor([float(r[key]) for r in group])
            summary[kind][key + "_mean"] = float(values.mean())
            summary[kind][key + "_std"] = float(values.std(unbiased=True))
    out = Path(__file__).parents[1] / "results" / "p4_practicality_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
