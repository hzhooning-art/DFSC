"""P4 strong-baseline probe with fixed fractional, signed, and constrained kernels."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def rel(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y)).detach())


def causal_apply(kernel: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    n = u.numel()
    idx = torch.arange(n, device=u.device)
    lag = idx[:, None] - idx[None, :]
    return dt * ((lag >= 0).to(u.dtype) * kernel[lag.clamp_min(0)]).tril() @ u


def normalized(raw: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    return raw / (raw.sum() * dt).clamp_min(1e-12)


def target(tau: torch.Tensor, kind: str, dt: torch.Tensor) -> torch.Tensor:
    if kind == "tempered_power_law":
        raw = tau.pow(-0.62) * torch.exp(-0.045 * tau)
    else:
        raw = 0.75 * 0.12 * torch.exp(-0.12 * tau) + 0.25 * 2.5 * torch.exp(-2.5 * tau)
    return normalized(raw, dt)


def fit_constrained(tau: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    modes = 12
    logits = torch.nn.Parameter(torch.zeros(modes, device=tau.device))
    log_rates = torch.nn.Parameter(torch.linspace(math.log(0.02), math.log(4.0), modes, device=tau.device))
    opt = torch.optim.Adam([logits, log_rates], lr=0.04)
    for _ in range(700):
        opt.zero_grad(set_to_none=True)
        w = torch.softmax(logits, 0)
        r = torch.nn.functional.softplus(log_rates)
        pred = (w[:, None] * r[:, None] * torch.exp(-r[:, None] * tau[None, :])).sum(0)
        ((pred[mask] - obs[mask]) ** 2).mean().backward()
        opt.step()
    w = torch.softmax(logits, 0)
    r = torch.nn.functional.softplus(log_rates)
    return (w[:, None] * r[:, None] * torch.exp(-r[:, None] * tau[None, :])).sum(0)


def fit_signed(tau: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    rates = torch.logspace(math.log10(0.02), math.log10(4.0), 12, device=tau.device)
    features = torch.exp(-rates[:, None] * tau[None, :]).T
    coeff = torch.linalg.lstsq(features[mask], obs[mask, None]).solution[:, 0]
    return features @ coeff


def fit_signed_ridge(t: torch.Tensor, tau: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Prefix-selected ridge signed mixture, avoiding an unregularized strawman."""
    rates = torch.logspace(math.log10(0.02), math.log10(4.0), 12, device=tau.device)
    features = torch.exp(-rates[:, None] * tau[None, :]).T
    train = mask & (t <= 14.0)
    valid = mask & (t > 14.0) & (t <= 20.0)
    candidates = (1e-8, 1e-6, 1e-4, 1e-2, 1e0)
    best = None
    best_score = None
    eye = torch.eye(features.shape[1], device=tau.device)
    for lam in candidates:
        lhs = features[train].T @ features[train] + lam * eye
        rhs = features[train].T @ obs[train]
        coeff = torch.linalg.solve(lhs, rhs)
        score = ((features[valid] @ coeff - obs[valid]) ** 2).mean()
        if best_score is None or score < best_score:
            best_score = score
            best = coeff
    assert best is not None
    return features @ best


def fit_fractional(tau: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    raw_alpha = torch.nn.Parameter(torch.tensor(0.0, device=tau.device))
    raw_temper = torch.nn.Parameter(torch.tensor(math.log(0.1), device=tau.device))
    opt = torch.optim.Adam([raw_alpha, raw_temper], lr=0.04)
    for _ in range(700):
        opt.zero_grad(set_to_none=True)
        alpha = 0.1 + 0.85 * torch.sigmoid(raw_alpha)
        temper = torch.nn.functional.softplus(raw_temper)
        pred = normalized(tau.pow(-alpha) * torch.exp(-temper * tau), dt)
        ((pred[mask] - obs[mask]) ** 2).mean().backward()
        opt.step()
    alpha = 0.1 + 0.85 * torch.sigmoid(raw_alpha)
    temper = torch.nn.functional.softplus(raw_temper)
    return normalized(tau.pow(-alpha) * torch.exp(-temper * tau), dt)


def one(kind: str, seed: int, device: torch.device) -> dict[str, float | str | int]:
    torch.manual_seed(seed)
    n = 192
    t = torch.linspace(0.0, 32.0, n, device=device)
    dt = t[1] - t[0]
    tau = t + 0.5 * dt
    truth = target(tau, kind, dt)
    obs = (truth + 0.025 * truth.max() * torch.randn_like(truth)).clamp_min(0.0)
    mask = (t <= 20.0) & ((torch.arange(n, device=device) % 2) == 0)
    learned = {
        "constrained_mixture": fit_constrained(tau, obs, mask, dt),
        "signed_mixture": fit_signed(tau, obs, mask, dt),
        "signed_ridge_mixture": fit_signed_ridge(t, tau, obs, mask, dt),
        "fractional_tempered": fit_fractional(tau, obs, mask, dt),
    }
    u = torch.sin(0.45 * t) + 0.3 * torch.cos(0.17 * t) + torch.exp(-0.5 * ((t - 3.0) / 0.3) ** 2)
    y_ref = causal_apply(truth, u, dt)
    hold = t > 20.0
    out: dict[str, float | str | int] = {"kind": kind, "seed": seed}
    for name, kernel in learned.items():
        y = causal_apply(kernel, u, dt)
        out[f"{name}_kernel_error"] = rel(kernel, truth)
        out[f"{name}_holdout_error"] = rel(kernel[hold], truth[hold])
        out[f"{name}_convolution_error"] = rel(y, y_ref)
        out[f"{name}_convolution_holdout_error"] = rel(y[hold], y_ref[hold])
        out[f"{name}_minimum"] = float(kernel.detach().min())
        out[f"{name}_negative_fraction"] = float((kernel.detach() < 0).float().mean())
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [one(kind, seed, device) for kind in ("tempered_power_law", "two_timescale") for seed in (3, 7, 11)]
    summary = {}
    methods = ("constrained_mixture", "signed_mixture", "signed_ridge_mixture", "fractional_tempered")
    for kind in ("tempered_power_law", "two_timescale"):
        group = [r for r in rows if r["kind"] == kind]
        summary[kind] = {}
        for method in methods:
            for metric in ("kernel_error", "holdout_error", "convolution_error", "convolution_holdout_error", "minimum", "negative_fraction"):
                vals = torch.tensor([float(r[f"{method}_{metric}"]) for r in group])
                summary[kind][f"{method}_{metric}_mean"] = float(vals.mean())
                summary[kind][f"{method}_{metric}_std"] = float(vals.std(unbiased=True))
    out = Path(__file__).parents[1] / "results" / "p4_strong_baseline_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
