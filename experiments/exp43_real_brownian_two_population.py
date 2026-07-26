"""Two-population Brownian bead inversion from AnomDiffDB trajectories."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dfsc
from experiments.exp41_real_spt_evidence_chain import (
    MAX_LAG,
    SEEDS,
    TRAIN_MAX_LAG,
    fit_direct_mlsl,
    fit_hybrid,
    fit_neural,
    fit_stretched_exponential,
    relative_error,
)


DATA = ROOT / "data" / "external" / "anomdiffdb" / "brownian_beads.mat"
RESULTS = ROOT / "generated_results"


def trajectory_diffusivity(track: np.ndarray) -> float:
    squared_steps = np.sum(np.diff(track, axis=0) ** 2, axis=1)
    return float(np.median(squared_steps) / 4.0)


def two_means_log_partition(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_values = np.log(values)
    centers = np.quantile(log_values, [0.25, 0.75])
    for _ in range(100):
        labels = np.abs(log_values[:, None] - centers[None, :]).argmin(axis=1)
        updated = np.asarray([log_values[labels == index].mean() for index in range(2)])
        if np.allclose(updated, centers, rtol=0.0, atol=1e-12):
            break
        centers = updated
    order = np.argsort(centers)
    remap = np.empty_like(order)
    remap[order] = np.arange(2)
    return remap[labels], np.exp(centers[order])


def fit_exponential(lags: np.ndarray, values: np.ndarray) -> tuple[float, np.ndarray]:
    rate = float(-np.dot(lags, np.log(np.clip(values, 1e-10, 1.0))) / np.dot(lags, lags))
    return rate, np.exp(-rate * np.arange(1, MAX_LAG + 1))


def append_model_rows(
    rows: list[dict[str, object]],
    condition: str,
    trajectories: tuple[np.ndarray, ...],
) -> None:
    wave_number = dfsc.estimate_wave_number(trajectories)
    lags = np.arange(1, MAX_LAG + 1)
    early = slice(0, TRAIN_MAX_LAG)
    late = slice(TRAIN_MAX_LAG, MAX_LAG)
    for seed in SEEDS:
        train_tracks, test_tracks = dfsc.split_trajectories(trajectories, seed=seed)
        train = dfsc.empirical_spt_observables(train_tracks, lags, wave_number=wave_number)
        test = dfsc.empirical_spt_observables(test_tracks, lags, wave_number=wave_number)
        exponential_rate, exponential_prediction = fit_exponential(lags[early], train.scattering[early])
        candidates: list[tuple[str, np.ndarray, float | None, float | None, int, float, float]] = [
            ("Exponential", exponential_prediction, 1.0, exponential_rate, 1, 0.0, 0.0)
        ]

        stretch_alpha, stretch_rate = fit_stretched_exponential(lags[early], train.scattering[early])
        stretch_prediction = np.exp(-stretch_rate * lags**stretch_alpha)
        candidates.append(("Stretched exponential", stretch_prediction, stretch_alpha, stretch_rate, 2, 0.0, 0.0))
        train_lags = torch.tensor(lags[early], dtype=torch.float64)
        train_target = torch.tensor(train.scattering[early], dtype=torch.float64)
        direct, direct_alpha, direct_rate, direct_parameters, direct_seconds = fit_direct_mlsl(
            train_lags, train_target, stretch_alpha, stretch_rate
        )
        candidates.append(("Direct MLSL inverse", direct, direct_alpha, direct_rate, direct_parameters, direct_seconds, 0.0))
        neural, neural_parameters, neural_seconds = fit_neural(train_lags, train_target, seed=seed)
        candidates.append(("Pure MLP", neural, None, None, neural_parameters, neural_seconds, 1.0))
        hybrid, hybrid_alpha, hybrid_rate, hybrid_parameters, hybrid_seconds, residual_fraction = fit_hybrid(
            train_lags, train_target, direct_alpha, direct_rate, seed=seed
        )
        candidates.append(("MLSL + residual MLP", hybrid, hybrid_alpha, hybrid_rate, hybrid_parameters, hybrid_seconds, residual_fraction))

        for model, prediction, alpha, rate, parameters, seconds, residual in candidates:
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "model": model,
                    "inferred_alpha": alpha,
                    "inferred_rate": rate,
                    "heldout_relative_error": relative_error(prediction, test.scattering),
                    "late_lag_relative_error": relative_error(prediction[late], test.scattering[late]),
                    "parameter_count": parameters,
                    "training_seconds": seconds,
                    "residual_fraction": residual,
                }
            )


def main() -> None:
    torch.set_default_dtype(torch.float64)
    RESULTS.mkdir(parents=True, exist_ok=True)
    dataset = dfsc.load_anomdiffdb_mat(DATA, condition="water-glycerol Brownian beads")
    eligible = tuple(track for track in dataset.trajectories if track.shape[0] > MAX_LAG)
    diffusivities = np.asarray([trajectory_diffusivity(track) for track in eligible])
    labels, centers = two_means_log_partition(diffusivities)
    groups = {
        "Brownian slow population": tuple(track for track, label in zip(eligible, labels) if label == 0),
        "Brownian fast population": tuple(track for track, label in zip(eligible, labels) if label == 1),
    }
    rows: list[dict[str, object]] = []
    for condition, tracks in groups.items():
        append_model_rows(rows, condition, tracks)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["model"]))].append(row)
    summary_rows: list[dict[str, object]] = []
    for (condition, model), model_rows in grouped.items():
        alpha_values = [float(row["inferred_alpha"]) for row in model_rows if row["inferred_alpha"] is not None]
        summary_rows.append(
            {
                "condition": condition,
                "model": model,
                "seeds": len(model_rows),
                "heldout_error_mean": float(np.mean([row["heldout_relative_error"] for row in model_rows])),
                "heldout_error_std": float(np.std([row["heldout_relative_error"] for row in model_rows], ddof=1)),
                "late_error_mean": float(np.mean([row["late_lag_relative_error"] for row in model_rows])),
                "late_error_std": float(np.std([row["late_lag_relative_error"] for row in model_rows], ddof=1)),
                "alpha_mean": float(np.mean(alpha_values)) if alpha_values else None,
                "alpha_std": float(np.std(alpha_values, ddof=1)) if alpha_values else None,
            }
        )

    raw_path = RESULTS / "real_brownian_two_population_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = RESULTS / "real_brownian_two_population_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "dataset": dataset.condition,
        "source": dataset.source,
        "citation": dataset.citation,
        "sha256": "1bc05922899172053126df80b28e43a645411d8b6919b1575a9a70ef4aa9d8e6",
        "eligible_trajectories_length_gt_40": len(eligible),
        "population_definition": "deterministic two-means clustering of log median one-frame diffusivity",
        "population_centers_source_units2_per_frame": centers.tolist(),
        "population_sizes": {name: len(tracks) for name, tracks in groups.items()},
        "interpretation_boundary": "clusters define empirical conditions; they are not claimed as ground-truth bead labels",
        "summary": summary_rows,
    }
    (RESULTS / "real_brownian_two_population_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
