"""Real-data inversion, traditional baselines, and neural hybrid comparison.

The experimental H-actin trajectories are described as fractional Brownian
motion by their source publication. MLSL instead represents a time-fractional
Mittag-Leffler relaxation. Its fitted alpha is therefore reported as an
effective model-conditional order, not as unique mechanism identification.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dfsc


DATA = ROOT / "data" / "external" / "anomdiffdb" / "750nm_mesh_size.mat"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
SEEDS = (0, 1, 2, 3, 4)
TRAIN_MAX_LAG = 20
MAX_LAG = 40


def relative_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1e-14))


def bounded_alpha(raw: torch.Tensor) -> torch.Tensor:
    return 0.2 + 1.4 * torch.sigmoid(raw)


def mlsl_curve(alpha: torch.Tensor, rate: torch.Tensor, lags: torch.Tensor) -> torch.Tensor:
    z = -rate * lags.pow(alpha)
    return dfsc.mittag_leffler_e(alpha, z, terms=120, custom_backward=False, method="hybrid")


class ScatteringMLP(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x.unsqueeze(-1)).squeeze(-1))


def fit_power_law(lags: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.log(lags), np.log(np.clip(values, 1e-12, None)), 1)
    return float(slope), float(np.exp(intercept))


def fit_stretched_exponential(lags: np.ndarray, scattering: np.ndarray) -> tuple[float, float]:
    transformed = np.log(-np.log(np.clip(scattering, 1e-8, 1.0 - 1e-8)))
    alpha, log_rate = np.polyfit(np.log(lags), transformed, 1)
    return float(alpha), float(np.exp(log_rate))


def fit_direct_mlsl(lags: torch.Tensor, target: torch.Tensor, alpha_init: float, rate_init: float) -> tuple[np.ndarray, float, float, int, float]:
    alpha_fraction = np.clip((alpha_init - 0.2) / 1.4, 1e-4, 1 - 1e-4)
    raw_alpha = torch.tensor(float(np.log(alpha_fraction / (1 - alpha_fraction))), requires_grad=True)
    raw_rate = torch.tensor(float(np.log(np.expm1(max(rate_init, 1e-6)))), requires_grad=True)
    optimizer = torch.optim.Adam([raw_alpha, raw_rate], lr=0.03)
    started = time.perf_counter()
    for _ in range(700):
        optimizer.zero_grad()
        prediction = mlsl_curve(bounded_alpha(raw_alpha), F.softplus(raw_rate), lags)
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    alpha = bounded_alpha(raw_alpha)
    rate = F.softplus(raw_rate)
    with torch.no_grad():
        prediction = mlsl_curve(alpha, rate, torch.arange(1, MAX_LAG + 1, dtype=torch.float64))
    return prediction.numpy(), float(alpha.detach()), float(rate.detach()), 2, elapsed


def fit_neural(lags: torch.Tensor, target: torch.Tensor, *, seed: int) -> tuple[np.ndarray, int, float]:
    torch.manual_seed(seed)
    model = ScatteringMLP().to(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
    scale = math.log1p(MAX_LAG)
    x = torch.log1p(lags) / scale
    started = time.perf_counter()
    for _ in range(1000):
        optimizer.zero_grad()
        loss = torch.mean((model(x) - target) ** 2)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    full_x = torch.log1p(torch.arange(1, MAX_LAG + 1, dtype=torch.float64)) / scale
    with torch.no_grad():
        prediction = model(full_x)
    return prediction.numpy(), sum(parameter.numel() for parameter in model.parameters()), elapsed


def fit_hybrid(lags: torch.Tensor, target: torch.Tensor, alpha_init: float, rate_init: float, *, seed: int) -> tuple[np.ndarray, float, float, int, float, float]:
    torch.manual_seed(seed)
    residual_head = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
    model = dfsc.MittagLefflerResidualRegressor(
        residual_head,
        alpha_init=alpha_init,
        rate_init=rate_init,
        alpha_bounds=(0.2, 1.6),
        residual_scale=0.35,
    ).to(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
    scale = math.log1p(MAX_LAG)
    x = torch.log1p(lags) / scale
    started = time.perf_counter()
    for _ in range(1000):
        optimizer.zero_grad()
        prediction = model(lags, x.unsqueeze(-1))
        base_regularizer = sum(parameter.square().mean() for parameter in model.residual_head.parameters())
        loss = torch.mean((prediction - target) ** 2) + 1e-7 * base_regularizer
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    full_lags = torch.arange(1, MAX_LAG + 1, dtype=torch.float64)
    full_x = torch.log1p(full_lags) / scale
    with torch.no_grad():
        prediction = model(full_lags, full_x.unsqueeze(-1))
        alpha, rate = model.alpha, model.rate
        base = model.base_prediction(full_lags)
        residual_fraction = torch.linalg.vector_norm(prediction - base) / torch.linalg.vector_norm(prediction)
    return (
        prediction.numpy(),
        float(alpha),
        float(rate),
        sum(parameter.numel() for parameter in model.parameters()),
        elapsed,
        float(residual_fraction),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError("run tools/fetch_anomdiffdb.py before this experiment")
    torch.set_default_dtype(torch.float64)
    dataset = dfsc.load_anomdiffdb_mat(DATA)
    wave_number = dfsc.estimate_wave_number(dataset.trajectories)
    lags_np = np.arange(1, MAX_LAG + 1)
    early = slice(0, TRAIN_MAX_LAG)
    late = slice(TRAIN_MAX_LAG, MAX_LAG)
    rows: list[dict[str, object]] = []
    observable_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        train_tracks, test_tracks = dfsc.split_trajectories(dataset.trajectories, seed=seed)
        train_obs = dfsc.empirical_spt_observables(train_tracks, lags_np, wave_number=wave_number)
        test_obs = dfsc.empirical_spt_observables(test_tracks, lags_np, wave_number=wave_number)
        for index, lag in enumerate(lags_np):
            observable_rows.append({
                "seed": seed,
                "lag_frames": int(lag),
                "train_scattering": train_obs.scattering[index],
                "test_scattering": test_obs.scattering[index],
                "train_msd": train_obs.msd[index],
                "test_msd": test_obs.msd[index],
                "test_displacement_count": int(test_obs.sample_counts[index]),
            })

        msd_alpha, msd_scale = fit_power_law(lags_np[early], train_obs.msd[early])
        msd_prediction = msd_scale * lags_np**msd_alpha
        rows.append({
            "seed": seed, "model": "MSD power law", "target": "MSD", "inferred_alpha": msd_alpha,
            "inferred_rate": msd_scale, "heldout_all_relative_error": relative_error(msd_prediction, test_obs.msd),
            "late_lag_relative_error": relative_error(msd_prediction[late], test_obs.msd[late]),
            "parameter_count": 2, "training_seconds": 0.0, "residual_fraction": 0.0,
        })

        stretch_alpha, stretch_rate = fit_stretched_exponential(lags_np[early], train_obs.scattering[early])
        stretch_prediction = np.exp(-stretch_rate * lags_np**stretch_alpha)
        rows.append({
            "seed": seed, "model": "Stretched exponential", "target": "scattering", "inferred_alpha": stretch_alpha,
            "inferred_rate": stretch_rate, "heldout_all_relative_error": relative_error(stretch_prediction, test_obs.scattering),
            "late_lag_relative_error": relative_error(stretch_prediction[late], test_obs.scattering[late]),
            "parameter_count": 2, "training_seconds": 0.0, "residual_fraction": 0.0,
        })

        train_lags = torch.tensor(lags_np[early], dtype=torch.float64)
        train_target = torch.tensor(train_obs.scattering[early], dtype=torch.float64)
        direct_prediction, direct_alpha, direct_rate, direct_parameters, direct_seconds = fit_direct_mlsl(
            train_lags, train_target, stretch_alpha, stretch_rate
        )
        rows.append({
            "seed": seed, "model": "Direct MLSL inverse", "target": "scattering", "inferred_alpha": direct_alpha,
            "inferred_rate": direct_rate, "heldout_all_relative_error": relative_error(direct_prediction, test_obs.scattering),
            "late_lag_relative_error": relative_error(direct_prediction[late], test_obs.scattering[late]),
            "parameter_count": direct_parameters, "training_seconds": direct_seconds, "residual_fraction": 0.0,
        })

        neural_prediction, neural_parameters, neural_seconds = fit_neural(train_lags, train_target, seed=seed)
        rows.append({
            "seed": seed, "model": "Pure MLP", "target": "scattering", "inferred_alpha": None,
            "inferred_rate": None, "heldout_all_relative_error": relative_error(neural_prediction, test_obs.scattering),
            "late_lag_relative_error": relative_error(neural_prediction[late], test_obs.scattering[late]),
            "parameter_count": neural_parameters, "training_seconds": neural_seconds, "residual_fraction": 1.0,
        })

        hybrid_prediction, hybrid_alpha, hybrid_rate, hybrid_parameters, hybrid_seconds, residual_fraction = fit_hybrid(
            train_lags, train_target, direct_alpha, direct_rate, seed=seed
        )
        rows.append({
            "seed": seed, "model": "MLSL + residual MLP", "target": "scattering", "inferred_alpha": hybrid_alpha,
            "inferred_rate": hybrid_rate, "heldout_all_relative_error": relative_error(hybrid_prediction, test_obs.scattering),
            "late_lag_relative_error": relative_error(hybrid_prediction[late], test_obs.scattering[late]),
            "parameter_count": hybrid_parameters, "training_seconds": hybrid_seconds, "residual_fraction": residual_fraction,
        })

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    aggregate_rows: list[dict[str, object]] = []
    for model, model_rows in grouped.items():
        aggregate_rows.append({
            "model": model,
            "target": model_rows[0]["target"],
            "seeds": len(model_rows),
            "heldout_all_error_mean": float(np.mean([row["heldout_all_relative_error"] for row in model_rows])),
            "heldout_all_error_std": float(np.std([row["heldout_all_relative_error"] for row in model_rows], ddof=1)),
            "late_lag_error_mean": float(np.mean([row["late_lag_relative_error"] for row in model_rows])),
            "late_lag_error_std": float(np.std([row["late_lag_relative_error"] for row in model_rows], ddof=1)),
            "inferred_alpha_mean": float(np.mean([row["inferred_alpha"] for row in model_rows])) if model != "Pure MLP" else None,
            "parameter_count": model_rows[0]["parameter_count"],
        })

    write_csv(TABLES / "real_spt_model_comparison_raw.csv", rows)
    write_csv(TABLES / "real_spt_model_comparison_summary.csv", aggregate_rows)
    write_csv(TABLES / "real_spt_observables.csv", observable_rows)
    scattering_rows = [row for row in aggregate_rows if row["target"] == "scattering"]
    best = min(scattering_rows, key=lambda row: row["late_lag_error_mean"])
    by_model = {str(row["model"]): row for row in aggregate_rows}
    stretched_late = float(by_model["Stretched exponential"]["late_lag_error_mean"])
    direct_late = float(by_model["Direct MLSL inverse"]["late_lag_error_mean"])
    neural_late = float(by_model["Pure MLP"]["late_lag_error_mean"])
    hybrid_late = float(by_model["MLSL + residual MLP"]["late_lag_error_mean"])
    summary = {
        "dataset": dataset.condition,
        "source": dataset.source,
        "citation": dataset.citation,
        "num_trajectories": dataset.num_trajectories,
        "num_localizations": dataset.num_localizations,
        "wave_number_inverse_source_units": wave_number,
        "trajectory_split": "70/30, seeds 0/1/2/3/4",
        "training_lags_frames": [1, TRAIN_MAX_LAG],
        "extrapolation_lags_frames": [TRAIN_MAX_LAG + 1, MAX_LAG],
        "best_scattering_model_by_late_error": best["model"],
        "best_scattering_late_error_mean": best["late_lag_error_mean"],
        "direct_mlsl_vs_stretched_late_error_reduction": 1.0 - direct_late / stretched_late,
        "hybrid_vs_pure_mlp_late_error_reduction": 1.0 - hybrid_late / neural_late,
        "hybrid_improves_over_direct_mlsl": hybrid_late < direct_late,
        "model_comparison": aggregate_rows,
        "interpretation_boundary": "MLSL alpha is an effective model-conditional order; the source describes this condition as FBM.",
        "redistribution_boundary": "source page does not state a dataset license; raw MAT is excluded from release artifacts",
    }
    (RESULTS / "real_spt_evidence_chain_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
