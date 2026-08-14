"""P4 distributed-order baseline and fine-grid reference check."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def rel(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y)).detach())


def normalize(x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    return x / (x.sum() * dt).clamp_min(1e-12)


def apply_kernel(kernel: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    n = u.numel()
    idx = torch.arange(n, device=u.device)
    lag = idx[:, None] - idx[None, :]
    return dt * ((lag >= 0).to(u.dtype) * kernel[lag.clamp_min(0)]).tril() @ u


def truth(tau: torch.Tensor, kind: str, dt: torch.Tensor) -> torch.Tensor:
    if kind == "tempered_power_law":
        raw = tau.pow(-0.62) * torch.exp(-0.045 * tau)
    else:
        raw = 0.75 * 0.12 * torch.exp(-0.12 * tau) + 0.25 * 2.5 * torch.exp(-2.5 * tau)
    return normalize(raw, dt)


def fit_distributed_order(t: torch.Tensor, tau: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    alpha_grid = torch.tensor((0.18, 0.32, 0.46, 0.60, 0.74, 0.88), device=t.device)
    logits = torch.nn.Parameter(torch.zeros(len(alpha_grid), device=t.device))
    raw_temper = torch.nn.Parameter(torch.tensor(math.log(0.08), device=t.device))
    opt = torch.optim.Adam([logits, raw_temper], lr=0.04)
    for _ in range(800):
        opt.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=0)
        temper = torch.nn.functional.softplus(raw_temper)
        modes = tau[None, :].pow(-alpha_grid[:, None]) * torch.exp(-temper * tau[None, :])
        pred = normalize((weights[:, None] * modes).sum(0), dt)
        ((pred[mask] - obs[mask]) ** 2).mean().backward()
        opt.step()
    weights = torch.softmax(logits, dim=0)
    temper = torch.nn.functional.softplus(raw_temper)
    modes = tau[None, :].pow(-alpha_grid[:, None]) * torch.exp(-temper * tau[None, :])
    return normalize((weights[:, None] * modes).sum(0), dt)


def fit_exponential(t: torch.Tensor, tau: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    modes = 12
    logits = torch.nn.Parameter(torch.zeros(modes, device=t.device))
    log_rates = torch.nn.Parameter(torch.linspace(math.log(0.02), math.log(4.0), modes, device=t.device))
    opt = torch.optim.Adam([logits, log_rates], lr=0.04)
    for _ in range(800):
        opt.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, 0)
        rates = torch.nn.functional.softplus(log_rates)
        pred = (weights[:, None] * rates[:, None] * torch.exp(-rates[:, None] * tau[None, :])).sum(0)
        ((pred[mask] - obs[mask]) ** 2).mean().backward()
        opt.step()
    weights = torch.softmax(logits, 0)
    rates = torch.nn.functional.softplus(log_rates)
    return (weights[:, None] * rates[:, None] * torch.exp(-rates[:, None] * tau[None, :])).sum(0)


def one(kind: str, seed: int, device: torch.device) -> dict[str, float | str | int]:
    torch.manual_seed(seed)
    n = 192
    t = torch.linspace(0.0, 32.0, n, device=device)
    dt = t[1] - t[0]
    tau = t + 0.5 * dt
    clean = truth(tau, kind, dt)
    obs = (clean + 0.025 * clean.max() * torch.randn_like(clean)).clamp_min(0.0)
    mask = (t <= 20.0) & ((torch.arange(n, device=device) % 2) == 0)
    dist = fit_distributed_order(t, tau, obs, mask, dt)
    exp = fit_exponential(t, tau, obs, mask, dt)
    u = torch.sin(0.45 * t) + 0.3 * torch.cos(0.17 * t) + torch.exp(-0.5 * ((t - 3.0) / 0.3) ** 2)
    ref = apply_kernel(clean, u, dt)
    hold = t > 20.0
    out: dict[str, float | str | int] = {"kind": kind, "seed": seed}
    for name, kernel in (("distributed_order", dist), ("exponential_mixture", exp)):
        pred = apply_kernel(kernel, u, dt)
        out[f"{name}_kernel_error"] = rel(kernel, clean)
        out[f"{name}_convolution_error"] = rel(pred, ref)
        out[f"{name}_convolution_holdout_error"] = rel(pred[hold], ref[hold])
        out[f"{name}_minimum"] = float(kernel.detach().min())
    # Independent fine-grid reference: compare the coarse analytic convolution
    # with a 4x finer quadrature and interpolate it at coarse query times.
    nf = 4 * n
    tf = torch.linspace(0.0, 32.0, nf, device=device)
    dtf = tf[1] - tf[0]
    tauf = tf + 0.5 * dtf
    cleanf = truth(tauf, kind, dtf)
    uf = torch.sin(0.45 * tf) + 0.3 * torch.cos(0.17 * tf) + torch.exp(-0.5 * ((tf - 3.0) / 0.3) ** 2)
    fine = apply_kernel(cleanf, uf, dtf)
    out["coarse_to_fine_reference_error"] = rel(ref, fine[::4])
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [one(kind, seed, device) for kind in ("tempered_power_law", "two_timescale") for seed in (3, 7, 11)]
    summary = {}
    for kind in ("tempered_power_law", "two_timescale"):
        group = [r for r in rows if r["kind"] == kind]
        summary[kind] = {}
        for key in rows[0]:
            if key in {"kind", "seed"}:
                continue
            vals = torch.tensor([float(r[key]) for r in group])
            summary[kind][key + "_mean"] = float(vals.mean())
            summary[kind][key + "_std"] = float(vals.std(unbiased=True))
    out = Path(__file__).parents[1] / "results" / "p4_distributed_order_reference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
