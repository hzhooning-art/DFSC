"""Add a strong optimizer-free trajectory baseline to the mechanism comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import UnivariateSpline

from probe_mechanism_vs_trajectory_baselines import NOISE_STD, RESULTS, make_dataset, rmse


SOURCE = RESULTS / "mechanism_vs_trajectory_baselines.json"


def fit_smoothing_spline(data: dict) -> tuple[np.ndarray, dict]:
    times = data["times"].detach().cpu().numpy()
    target = data["observations"].detach().cpu().numpy()
    train_idx = data["train_idx"].detach().cpu().numpy()
    prediction = np.empty_like(target)
    smoothing_budget = len(train_idx) * NOISE_STD**2
    residual_sums = []
    for channel in range(target.shape[1]):
        spline = UnivariateSpline(
            times[train_idx],
            target[train_idx, channel],
            k=3,
            s=smoothing_budget,
            ext=0,
        )
        prediction[:, channel] = spline(times)
        residual_sums.append(float(spline.get_residual()))
    return prediction, {
        "family": "cubic_smoothing_spline",
        "smoothing_rule": "n_train * known_noise_variance per channel",
        "uses_validation_or_extrapolation_for_tuning": False,
        "channel_residual_sums": residual_sums,
        "note": "The spline estimates trajectories only and has no memory-rank output.",
    }


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = []
    for old in source["records"]:
        data = make_dataset(old["case"], old["repeat"])
        prediction, metadata = fit_smoothing_spline(data)
        clean = data["clean"].detach().cpu().numpy()
        train_idx = data["train_idx"].detach().cpu().numpy()
        interpolation_idx = data["interpolation_idx"].detach().cpu().numpy()
        extrapolation_idx = data["extrapolation_idx"].detach().cpu().numpy()
        spline_metrics = {
            **metadata,
            "train_rmse_to_clean": rmse(prediction[train_idx], clean[train_idx]),
            "interpolation_rmse_to_clean": rmse(
                prediction[interpolation_idx], clean[interpolation_idx]
            ),
            "extrapolation_rmse_to_clean": rmse(
                prediction[extrapolation_idx], clean[extrapolation_idx]
            ),
        }
        records.append({
            "case": old["case"],
            "repeat": old["repeat"],
            "seed": old["seed"],
            "positive_real_memory": {
                key: old["methods"]["positive_real_memory"][key]
                for key in (
                    "selected_rank",
                    "rank_recovered",
                    "quality_failure",
                    "train_rmse_to_clean",
                    "interpolation_rmse_to_clean",
                    "extrapolation_rmse_to_clean",
                )
            },
            "smoothing_spline": spline_metrics,
        })

    rows = []
    for case in source["design"]["cases"]:
        group = [record for record in records if record["case"] == case]
        row = {"case": case, "trials": len(group), "methods": {}}
        for method in ("positive_real_memory", "smoothing_spline"):
            row["methods"][method] = {
                metric: float(np.median([record[method][metric] for record in group]))
                for metric in (
                    "train_rmse_to_clean",
                    "interpolation_rmse_to_clean",
                    "extrapolation_rmse_to_clean",
                )
            }
        row["memory_rank_recoveries"] = sum(
            record["positive_real_memory"]["rank_recovered"] for record in group
        )
        rows.append(row)

    ratios = np.array([
        row["methods"]["positive_real_memory"]["extrapolation_rmse_to_clean"]
        / row["methods"]["smoothing_spline"]["extrapolation_rmse_to_clean"]
        for row in rows
    ])
    checks = {
        "spline_training_rmse_reaches_2p5_noise_std_in_each_case": all(
            row["methods"]["smoothing_spline"]["train_rmse_to_clean"] <= 2.5 * NOISE_STD
            for row in rows
        ),
        "spline_interpolation_rmse_within_5_noise_std_in_each_case": all(
            row["methods"]["smoothing_spline"]["interpolation_rmse_to_clean"] <= 5.0 * NOISE_STD
            for row in rows
        ),
        "mechanism_extrapolation_at_least_25_percent_better_than_spline_on_median": (
            float(np.median(ratios)) <= 0.75
        ),
        "memory_rank_recovered_in_at_least_7_of_9_trials": sum(
            row["memory_rank_recoveries"] for row in rows
        ) >= 7,
    }
    payload = {
        "experiment": "unconstrained_spline_baseline",
        "source_artifact": str(SOURCE),
        "design": {
            "comparison": "positive-real memory model versus cubic smoothing spline",
            "fairness_boundary": (
                "Prediction errors are comparable; the spline does not identify memory rank."
            ),
            "smoothing_rule": "n_train * known_noise_variance per channel",
        },
        "records": records,
        "summary": rows,
        "assessment": {
            "checks": checks,
            "route_pass": all(checks.values()),
            "median_extrapolation_ratio_mechanism_to_spline": float(np.median(ratios)),
        },
    }
    (RESULTS / "unconstrained_spline_baseline.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Unconstrained smoothing-spline baseline",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        "- Smoothing was fixed from the known noise variance; no held-out values tuned it.",
        "- Spline complexity and memory rank are not treated as equivalent quantities.",
        "",
        "| Case | Memory extrap. | Spline train | Spline interp. | Spline extrap. |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        memory = row["methods"]["positive_real_memory"]
        spline = row["methods"]["smoothing_spline"]
        lines.append(
            f"| {row['case']} | {memory['extrapolation_rmse_to_clean']:.4e} | "
            f"{spline['train_rmse_to_clean']:.4e} | "
            f"{spline['interpolation_rmse_to_clean']:.4e} | "
            f"{spline['extrapolation_rmse_to_clean']:.4e} |"
        )
    lines.extend(["", "## Prespecified checks", ""])
    lines.extend(f"- {name}: **{value}**" for name, value in checks.items())
    (RESULTS / "unconstrained_spline_baseline.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["assessment"], indent=2))


if __name__ == "__main__":
    main()
