"""Calibrate the EAFI numerical-error inflation factor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from eafi_feasibility import grid_predictions, ml_family  # noqa: E402
sys.path.insert(0, str(ROOT / "P1" / "paper1_mlsl"))


def evaluate_config(device, low_terms, multiplier, sigma, replicates=20):
    times = torch.linspace(0.05, 1.5, 32, dtype=torch.float64, device=device)
    true_a, true_r = 0.78, 0.65
    alpha_grid = torch.linspace(0.65, 0.91, 27, dtype=torch.float64, device=device)
    rate_grid = torch.linspace(0.40, 0.90, 26, dtype=torch.float64, device=device)
    ref = ml_family(torch.tensor(true_a, dtype=torch.float64, device=device), torch.tensor(true_r, dtype=torch.float64, device=device), times, terms=100)
    high = grid_predictions(alpha_grid, rate_grid, times, terms=100)
    low = grid_predictions(alpha_grid, rate_grid, times, terms=low_terms)
    floor = torch.mean((high - low) ** 2, dim=-1).max()
    ia = torch.argmin(torch.abs(alpha_grid - true_a))
    ir = torch.argmin(torch.abs(rate_grid - true_r))
    covered = []
    widths = []
    for seed in range(replicates):
        torch.manual_seed(30000 + seed)
        y = ref + sigma * torch.randn_like(ref)
        loss = torch.sum((low - y[None, None, :]) ** 2, dim=-1)
        threshold = 2.0 * times.numel() * sigma**2 + multiplier * times.numel() * floor
        keep = loss <= loss.min() + threshold
        covered.append(bool(keep[ia, ir].item()))
        widths.append(int(keep.sum().item()))
    return {
        "low_terms": low_terms,
        "multiplier": multiplier,
        "sigma": sigma,
        "floor_mse": float(floor.cpu()),
        "coverage": sum(covered) / len(covered),
        "mean_width": sum(widths) / len(widths),
        "width_std": float(torch.tensor(widths, dtype=torch.float64).std(unbiased=True)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for low_terms in [4, 6, 8, 12]:
        for multiplier in [0.25, 0.5, 1.0, 2.0, 3.0]:
            for sigma in [0.005, 0.01, 0.02]:
                rows.append(evaluate_config(device, low_terms, multiplier, sigma))
    candidates = [r for r in rows if 0.90 <= r["coverage"] <= 1.0]
    candidates.sort(key=lambda r: (abs(r["coverage"] - 0.95), r["mean_width"]))
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_configs": len(rows),
        "candidate_count": len(candidates),
        "recommended_candidates": candidates[:10],
        "all_results": rows,
        "interpretation": "calibration grid for empirical coverage/width trade-off; no threshold is frozen yet",
    }
    out = ROOT / "P4" / "results" / "p4_eafi_uncertainty_calibration.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "all_results"}, indent=2))


if __name__ == "__main__":
    main()
