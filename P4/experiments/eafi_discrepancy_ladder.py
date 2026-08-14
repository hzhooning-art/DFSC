"""Calibrate a model-discrepancy statistic and test a mismatch ladder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from eafi_feasibility import grid_predictions, ml_family  # noqa: E402


def best_normalized_loss(prediction, y, sigma):
    loss = torch.sum((prediction - y[None, None, :]) ** 2, dim=-1)
    return loss.min() / (y.numel() * sigma**2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    times = torch.linspace(0.05, 1.5, 32, dtype=torch.float64, device=device)
    alpha_grid = torch.linspace(0.65, 0.91, 27, dtype=torch.float64, device=device)
    rate_grid = torch.linspace(0.40, 0.90, 26, dtype=torch.float64, device=device)
    true_alpha, true_rate, sigma = 0.773, 0.637, 0.01
    reference = ml_family(torch.tensor(true_alpha, dtype=torch.float64, device=device), torch.tensor(true_rate, dtype=torch.float64, device=device), times, terms=100)
    prediction = grid_predictions(alpha_grid, rate_grid, times, terms=100)

    null_scores = []
    for seed in range(45000, 45100):
        torch.manual_seed(seed)
        y = reference + sigma * torch.randn_like(reference)
        null_scores.append(float(best_normalized_loss(prediction, y, sigma).cpu()))
    null_sorted = sorted(null_scores)
    # Conservative empirical 95% threshold from the null calibration sample.
    threshold = null_sorted[int(0.95 * len(null_sorted)) - 1]

    stretched = torch.exp(-0.95 * times.pow(0.55))
    rows = []
    for fraction in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]:
        scores = []
        for seed in range(46000, 46100):
            torch.manual_seed(seed)
            clean = reference + sigma * torch.randn_like(reference)
            y = (1.0 - fraction) * clean + fraction * (stretched + sigma * torch.randn_like(reference))
            score = best_normalized_loss(prediction, y, sigma)
            scores.append(float(score.cpu()))
        rows.append({
            "contamination_fraction": fraction,
            "replicates": len(scores),
            "mean_score": sum(scores) / len(scores),
            "median_score": sorted(scores)[len(scores) // 2],
            "reject_rate": sum(s > threshold for s in scores) / len(scores),
        })
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "evaluator_terms": 100,
        "truth": {"alpha": true_alpha, "rate": true_rate, "sigma": sigma},
        "null_calibration": {
            "replicates": len(null_scores),
            "threshold_95": threshold,
            "null_mean": sum(null_scores) / len(null_scores),
            "null_reject_rate": sum(s > threshold for s in null_scores) / len(null_scores),
        },
        "mismatch_ladder": rows,
        "exit_rule": "retain route only if rejection rises above 0.50 by 20% contamination while null rejection remains <= 0.10",
        "interpretation": "empirical discrepancy screening; threshold is calibrated for this controlled family and is not a universal test",
    }
    out = ROOT / "P4" / "results" / "p4_eafi_discrepancy_ladder.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
