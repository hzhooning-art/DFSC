"""Matched-budget sweep for the MLSL physics-consistency weight."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from p4_physics_consistency_reuse import PhysicsAwareModel, make_data, train_one
from p4_spectral_compression_feasibility import ml_family_batch

ROOT = Path(__file__).resolve().parents[2]


def evaluate(model, context, target, target_times, true_alpha, true_rate):
    trajectory, alpha, rate = model(context)
    reference = ml_family_batch(
        alpha[:, None].expand(-1, len(target_times)).reshape(-1),
        rate[:, None].expand(-1, len(target_times)).reshape(-1),
        target_times.expand(len(context), -1).reshape(-1),
        terms=16,
    ).reshape_as(trajectory)
    return {
        "test_rmse": float(torch.sqrt(torch.mean((trajectory - target) ** 2)).detach().cpu()),
        "alpha_error": float(torch.mean((alpha - true_alpha).abs()).detach().cpu()),
        "rate_error": float(torch.mean((rate - true_rate).abs()).detach().cpu()),
        "physics_residual": float(torch.sqrt(torch.mean((trajectory - reference) ** 2)).detach().cpu()),
        "gradient_finite": all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_times = torch.linspace(0.05, 0.40, 8, dtype=torch.float64, device=device)
    target_times = torch.linspace(0.50, 1.50, 16, dtype=torch.float64, device=device)
    weights = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]
    rows = []
    for seed in [54100, 54101, 54102]:
        train_context, train_target, _, _ = make_data(
            device, 512, context_times, target_times, (0.68, 0.86), (0.45, 0.80), seed
        )
        test_context, test_target, test_alpha, test_rate = make_data(
            device, 256, context_times, target_times, (0.65, 0.90), (0.40, 0.90), seed + 100
        )
        for weight in weights:
            torch.manual_seed(seed + int(weight * 1000) + 17)
            model = PhysicsAwareModel(8, 16).to(device).double()
            elapsed = train_one(model, train_context, train_target, target_times, weight, steps=600)
            metrics = evaluate(model, test_context, test_target, target_times, test_alpha, test_rate)
            rows.append({"seed": seed, "physics_weight": weight, "elapsed_seconds": elapsed, **metrics})

    summary = {}
    for weight in weights:
        group = [row for row in rows if row["physics_weight"] == weight]
        summary[str(weight)] = {}
        for key in ("test_rmse", "alpha_error", "rate_error", "physics_residual", "elapsed_seconds"):
            values = torch.tensor([row[key] for row in group], dtype=torch.float64)
            summary[str(weight)][key + "_mean"] = float(values.mean())
            summary[str(weight)][key + "_std"] = float(values.std(unbiased=True))
        summary[str(weight)]["all_gradients_finite"] = all(row["gradient_finite"] for row in group)

    # Predeclare a selection rule: choose the lowest OOD RMSE among settings
    # whose physics residual is no worse than the data-only setting.
    baseline_residual = summary["0.0"]["physics_residual_mean"]
    eligible = {
        weight: values
        for weight, values in summary.items()
        if values["physics_residual_mean"] <= baseline_residual
    }
    selected_weight = min(eligible, key=lambda weight: eligible[weight]["test_rmse_mean"])
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "weights": weights,
        "seeds": [54100, 54101, 54102],
        "summary": summary,
        "selection_rule": "minimize OOD RMSE subject to mean physics residual no worse than weight 0.0",
        "selected_weight": float(selected_weight),
        "interpretation": "matched-budget sensitivity audit; not a new PINN architecture claim",
        "rows": rows,
    }
    out = ROOT / "P4" / "results" / "p4_physics_weight_sweep.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "summary"}}, indent=2))
    print(json.dumps({"selected": selected_weight, "selected_summary": summary[selected_weight]}, indent=2))


if __name__ == "__main__":
    main()
