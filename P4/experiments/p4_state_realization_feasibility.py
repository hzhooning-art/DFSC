"""Feasibility test for a differentiable irregular-query fractional state."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from p4_spectral_compression_feasibility import ml_family_batch  # noqa: E402


def cheb2(x, degree, lo, hi):
    z = 2.0 * (x - lo) / (hi - lo) - 1.0
    out = [torch.ones_like(z)]
    if degree > 1:
        out.append(z)
    for _ in range(2, degree):
        out.append(2.0 * z * out[-1] - out[-2])
    return torch.stack(out, dim=-1)


def fit_state_surface(device, state_count):
    alpha_grid = torch.linspace(0.65, 0.91, 12, dtype=torch.float64, device=device)
    rate_grid = torch.linspace(0.40, 0.90, 12, dtype=torch.float64, device=device)
    train_t = torch.linspace(0.05, 2.0, 64, dtype=torch.float64, device=device)
    rates = torch.logspace(torch.log10(torch.tensor(0.2, device=device)), torch.log10(torch.tensor(20.0, device=device)), state_count, dtype=torch.float64, device=device)
    design = torch.exp(-train_t[:, None] * rates[None, :])
    aa, rr = torch.meshgrid(alpha_grid, rate_grid, indexing="ij")
    flat_a, flat_r = aa.reshape(-1), rr.reshape(-1)
    coeff_rows = []
    for a, r in zip(flat_a, flat_r):
        target = ml_family_batch(a.expand_as(train_t), r.expand_as(train_t), train_t, terms=100)
        coeff_rows.append(torch.linalg.lstsq(design, target[:, None]).solution[:, 0])
    coeff = torch.stack(coeff_rows).reshape(len(alpha_grid), len(rate_grid), state_count)
    pa = cheb2(flat_a, 6, 0.65, 0.91)
    pr = cheb2(flat_r, 6, 0.40, 0.90)
    features = torch.einsum("ni,nj->nij", pa, pr).reshape(-1, 36)
    surface_coeff = torch.linalg.lstsq(features, coeff.reshape(-1, state_count)).solution
    return rates, surface_coeff


def state_value(alpha, rate, times, rates, surface_coeff):
    pa = cheb2(alpha, 6, 0.65, 0.91)
    pr = cheb2(rate, 6, 0.40, 0.90)
    feat = torch.einsum("ni,nj->nij", pa, pr).reshape(alpha.shape[0], 36)
    coeff = feat @ surface_coeff
    return torch.sum(coeff * torch.exp(-times[:, None] * rates[None, :]), dim=-1)


def state_rollout(alpha, rate, query_times, rates, surface_coeff):
    pa = cheb2(alpha, 6, 0.65, 0.91)
    pr = cheb2(rate, 6, 0.40, 0.90)
    coeff = torch.einsum("ni,nj->nij", pa, pr).reshape(alpha.shape[0], 36) @ surface_coeff
    state = torch.ones((alpha.shape[0], len(rates)), dtype=torch.float64, device=query_times.device)
    previous = torch.zeros(alpha.shape[0], dtype=torch.float64, device=query_times.device)
    outputs = []
    for i in range(query_times.shape[1]):
        dt = (query_times[:, i] - previous).clamp_min(0.0)
        state = state * torch.exp(-dt[:, None] * rates[None, :])
        outputs.append(torch.sum(coeff * state, dim=-1))
        previous = query_times[:, i]
    return torch.stack(outputs, dim=1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(50000)
    n_test, horizon = 256, 1000
    alpha = 0.65 + 0.26 * torch.rand(n_test, dtype=torch.float64, device=device)
    rate = 0.40 + 0.50 * torch.rand(n_test, dtype=torch.float64, device=device)
    query = 0.05 + 9.95 * torch.rand(n_test, horizon, dtype=torch.float64, device=device)
    query, _ = torch.sort(query, dim=1)
    reference = ml_family_batch(alpha[:, None].expand(-1, horizon).reshape(-1), rate[:, None].expand(-1, horizon).reshape(-1), query.reshape(-1), terms=100).reshape(n_test, horizon)
    reference_last = reference[:, -1]
    rows = []
    start = time.perf_counter()
    direct = ml_family_batch(alpha[:, None].expand(-1, horizon).reshape(-1), rate[:, None].expand(-1, horizon).reshape(-1), query.reshape(-1), terms=100).reshape(n_test, horizon)
    if query.is_cuda:
        torch.cuda.synchronize()
    direct_elapsed = time.perf_counter() - start
    rows.append({
        "state_count": None,
        "model": "direct_mlsl_reference",
        "state_bytes": 0,
        "forward_rmse_all_queries": 0.0,
        "forward_rmse_final_query": 0.0,
        "gradient_rmse_final_query": 0.0,
        "mean_rollout_seconds": direct_elapsed,
        "gradient_finite": True,
    })
    for state_count in [4, 8, 16, 32]:
        rates, surface_coeff = fit_state_surface(device, state_count)
        a = alpha.detach().clone().requires_grad_(True)
        r = rate.detach().clone().requires_grad_(True)
        start = time.perf_counter()
        output = state_rollout(a, r, query, rates, surface_coeff)
        if query.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        grad_a, grad_r = torch.autograd.grad(output[:, -1].sum(), (a, r))
        ref_a = alpha.detach().clone().requires_grad_(True)
        ref_r = rate.detach().clone().requires_grad_(True)
        ref_last = ml_family_batch(ref_a, ref_r, query[:, -1], terms=100)
        ref_ga, ref_gr = torch.autograd.grad(ref_last.sum(), (ref_a, ref_r))
        rows.append({
            "state_count": state_count,
            "state_bytes": int(state_count * 8),
            "forward_rmse_all_queries": float(torch.sqrt(torch.mean((output.detach() - reference) ** 2)).cpu()),
            "forward_rmse_final_query": float(torch.sqrt(torch.mean((output[:, -1].detach() - reference_last) ** 2)).cpu()),
            "gradient_rmse_final_query": float(torch.sqrt(torch.mean((grad_a.detach() - ref_ga.detach()) ** 2 + (grad_r.detach() - ref_gr.detach()) ** 2)).cpu()),
            "mean_rollout_seconds": elapsed,
            "gradient_finite": bool(torch.isfinite(grad_a).all() and torch.isfinite(grad_r).all()),
        })
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "test_batch": n_test,
        "query_horizon": horizon,
        "query_time_domain": [0.05, 10.0],
        "reference_terms": 100,
        "results": rows,
        "interpretation": "controlled feasibility only; state coefficients and envelope are not frozen for publication",
    }
    out = ROOT / "P4" / "results" / "p4_state_realization_feasibility.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
