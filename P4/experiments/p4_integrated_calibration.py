"""Batched differentiable calibration for the integrated DFSC workflow."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from p4_spectral_compression_feasibility import ml_family_batch  # noqa: E402


def bounded(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(51000)
    tasks, observations, steps = 32, 32, 120
    times = torch.linspace(0.05, 1.5, observations, dtype=torch.float64, device=device)
    true_alpha = 0.68 + 0.20 * torch.rand(tasks, dtype=torch.float64, device=device)
    true_rate = 0.45 + 0.40 * torch.rand(tasks, dtype=torch.float64, device=device)
    sigma = 0.01
    target = ml_family_batch(true_alpha[:, None].expand(-1, observations).reshape(-1), true_rate[:, None].expand(-1, observations).reshape(-1), times.repeat(tasks), terms=100).reshape(tasks, observations)
    target = target + sigma * torch.randn_like(target)

    raw_a = torch.nn.Parameter(torch.zeros(tasks, dtype=torch.float64, device=device))
    raw_r = torch.nn.Parameter(torch.zeros(tasks, dtype=torch.float64, device=device))
    opt = torch.optim.Adam([raw_a, raw_r], lr=0.08)
    start = time.perf_counter()
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        a = bounded(raw_a, 0.60, 0.95)
        r = bounded(raw_r, 0.30, 1.00)
        pred = ml_family_batch(a[:, None].expand(-1, observations).reshape(-1), r[:, None].expand(-1, observations).reshape(-1), times.repeat(tasks), terms=16).reshape(tasks, observations)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    grad_elapsed = time.perf_counter() - start
    fit_a, fit_r = bounded(raw_a.detach(), 0.60, 0.95), bounded(raw_r.detach(), 0.30, 1.00)

    # Matched grid-search baseline over the same bounded parameter domain.
    grid_a = torch.linspace(0.60, 0.95, 41, dtype=torch.float64, device=device)
    grid_r = torch.linspace(0.30, 1.00, 41, dtype=torch.float64, device=device)
    ga, gr = torch.meshgrid(grid_a, grid_r, indexing="ij")
    grid_a_flat, grid_r_flat = ga.reshape(-1), gr.reshape(-1)
    start = time.perf_counter()
    grid_rows = []
    for task in range(tasks):
        aa = grid_a_flat[:, None].expand(-1, observations).reshape(-1)
        rr = grid_r_flat[:, None].expand(-1, observations).reshape(-1)
        tt = times.repeat(len(grid_a_flat))
        pred = ml_family_batch(aa, rr, tt, terms=100).reshape(len(grid_a_flat), observations)
        score = torch.mean((pred - target[task][None, :]) ** 2, dim=1)
        idx = torch.argmin(score)
        grid_rows.append((grid_a_flat[idx], grid_r_flat[idx]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    grid_elapsed = time.perf_counter() - start
    grid_fit_a = torch.stack([x[0] for x in grid_rows])
    grid_fit_r = torch.stack([x[1] for x in grid_rows])

    def metrics(a, r):
        return {
            "alpha_abs_error_mean": float(torch.mean((a - true_alpha).abs()).cpu()),
            "rate_abs_error_mean": float(torch.mean((r - true_rate).abs()).cpu()),
            "joint_error_mean": float(torch.mean(torch.sqrt((a - true_alpha) ** 2 + (r - true_rate) ** 2)).cpu()),
        }

    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tasks": tasks,
        "observations": observations,
        "noise_sigma": sigma,
        "optimizer_steps": steps,
        "differentiable_calibration": {**metrics(fit_a, fit_r), "elapsed_seconds": grad_elapsed, "final_loss": losses[-1], "gradient_finite": bool(torch.isfinite(raw_a.grad).all() if raw_a.grad is not None else True)},
        "grid_search": {**metrics(grid_fit_a, grid_fit_r), "elapsed_seconds": grid_elapsed, "grid_size": len(grid_a_flat)},
        "interpretation": "first integrated workflow gate; calibration budget and baselines are not yet publication-frozen",
    }
    out = ROOT / "P4" / "results" / "p4_integrated_calibration.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
