"""Compare mechanism-constrained memory fitting with trajectory baselines.

The comparison deliberately separates predictive metrics from mechanism claims:
the damped-modal (Prony-like) and MLP baselines do not estimate memory-kernel rank.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from probe_memory_rank import DEVICE, DTYPE, FitResult, fit_rank, lifted_response


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPEATS = 3
HORIZON = 16.0
NUM_POINTS = 129
CHANNELS = 8
NOISE_STD = 8.0e-4
TRAIN_END_FRACTION = 0.60
TRAIN_COUNT = 48
CASES = {
    "rank1_separated": [0.24],
    "rank2_separated": [0.16, 1.25],
    "rank3_separated": [0.10, 0.58, 2.40],
}


def prediction_from_fit(times: torch.Tensor, fit: FitResult) -> torch.Tensor:
    weights = torch.tensor(fit.weights, dtype=DTYPE, device=DEVICE)
    rates = torch.tensor(fit.rates, dtype=DTYPE, device=DEVICE)
    return lifted_response(times, weights, rates)


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def make_dataset(case: str, repeat: int) -> dict:
    seed = 431000 + 1009 * list(CASES).index(case) + 101 * repeat
    rng = np.random.default_rng(seed)
    times = torch.linspace(0.0, HORIZON, NUM_POINTS, dtype=DTYPE, device=DEVICE)
    rates = CASES[case]
    rank = len(rates)
    channel_scale = np.linspace(0.28, 0.82, CHANNELS)[:, None]
    pole_scale = np.linspace(0.78, 1.22, rank)[None, :]
    weights = torch.tensor(channel_scale * pole_scale / rank, dtype=DTYPE, device=DEVICE)
    clean = lifted_response(
        times,
        weights,
        torch.tensor(rates, dtype=DTYPE, device=DEVICE),
    )
    observations = clean + NOISE_STD * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )

    split = int(round(TRAIN_END_FRACTION * (NUM_POINTS - 1)))
    pool = np.arange(1, split + 1)
    sampled = np.sort(rng.choice(pool, size=TRAIN_COUNT - 1, replace=False))
    train_np = np.concatenate(([0], sampled))
    interpolation_np = np.setdiff1d(np.arange(split + 1), train_np)
    extrapolation_np = np.arange(split + 1, NUM_POINTS)
    return {
        "seed": seed,
        "times": times,
        "clean": clean,
        "observations": observations,
        "train_idx": torch.tensor(train_np, dtype=torch.long, device=DEVICE),
        "interpolation_idx": torch.tensor(interpolation_np, dtype=torch.long, device=DEVICE),
        "extrapolation_idx": torch.tensor(extrapolation_np, dtype=torch.long, device=DEVICE),
        "true_rank": rank,
    }


def fit_mechanism(data: dict) -> tuple[dict, np.ndarray]:
    candidates: list[FitResult] = []
    all_starts: list[dict] = []
    for rank in (1, 2, 3):
        starts = [
            fit_rank(
                data["times"],
                data["observations"],
                data["train_idx"],
                data["interpolation_idx"],
                rank=rank,
                seed=data["seed"] * 100 + 17 * rank + start,
                adam_steps=170,
                lbfgs_steps=45,
            )
            for start in range(2)
        ]
        all_starts.extend(asdict(item) for item in starts)
        candidates.append(min(starts, key=lambda item: item.bic))
    winner = min(candidates, key=lambda item: item.bic)
    prediction = prediction_from_fit(data["times"], winner).detach().cpu().numpy()
    quality_failure = (
        winner.jacobian_condition > 1.0e8
        or winner.train_rmse > max(4.0 * NOISE_STD, 3.0e-3)
    )
    return {
        "selected_rank": winner.rank,
        "rank_recovered": winner.rank == data["true_rank"],
        "bic": winner.bic,
        "condition": winner.jacobian_condition,
        "quality_failure": quality_failure,
        "candidate_fits": [asdict(item) for item in candidates],
        "all_starts": all_starts,
    }, prediction


def modal_design(times: np.ndarray, decay: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    columns = [np.ones_like(times)]
    for local_decay, local_frequency in zip(decay, frequency):
        envelope = np.exp(-local_decay * times)
        columns.append(envelope * np.cos(local_frequency * times))
        columns.append(envelope * np.sin(local_frequency * times))
    return np.column_stack(columns)


def ridge_coefficients(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def fit_modal_rank(
    times: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    rank: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    train_times = times[train_idx]
    train_target = target[train_idx]
    ridge = 1.0e-6

    def objective(vector: np.ndarray) -> float:
        decay = np.exp(vector[:rank])
        frequency = np.exp(vector[rank:])
        design = modal_design(train_times, decay, frequency)
        coefficients = ridge_coefficients(design, train_target, ridge)
        residual = design @ coefficients - train_target
        return float(np.mean(residual**2) + ridge * np.mean(coefficients[1:] ** 2))

    best = None
    base_decay = np.geomspace(0.05, 0.8, rank)
    base_frequency = np.geomspace(0.15, 1.8, rank)
    bounds = [(math.log(5.0e-3), math.log(3.0))] * rank + [
        (math.log(2.0e-2), math.log(4.0))
    ] * rank
    for start in range(3):
        initial = np.log(np.concatenate([base_decay, base_frequency]))
        initial += rng.normal(0.0, 0.35, size=2 * rank)
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 140, "ftol": 1.0e-12, "gtol": 1.0e-8},
        )
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    decay = np.exp(best.x[:rank])
    frequency = np.exp(best.x[rank:])
    train_design = modal_design(train_times, decay, frequency)
    coefficients = ridge_coefficients(train_design, train_target, ridge)
    prediction = modal_design(times, decay, frequency) @ coefficients
    residual = prediction[train_idx] - target[train_idx]
    n = residual.size
    parameter_count = 2 * rank + target.shape[1] * (1 + 2 * rank)
    rss = max(float(np.sum(residual**2)), 1.0e-30)
    bic = n * math.log(rss / n) + parameter_count * math.log(n)
    return {
        "rank": rank,
        "bic": bic,
        "decay": decay.tolist(),
        "frequency": frequency.tolist(),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "prediction": prediction,
    }


def fit_modal_baseline(data: dict) -> tuple[dict, np.ndarray]:
    times = data["times"].detach().cpu().numpy()
    target = data["observations"].detach().cpu().numpy()
    train_idx = data["train_idx"].detach().cpu().numpy()
    candidates = [
        fit_modal_rank(times, target, train_idx, rank, data["seed"] + 307 * rank)
        for rank in (1, 2, 3)
    ]
    winner = min(candidates, key=lambda item: item["bic"])
    prediction = winner.pop("prediction")
    for candidate in candidates:
        candidate.pop("prediction", None)
    return {
        "selected_modal_rank": winner["rank"],
        "note": "Damped trajectory-mode rank is not memory-kernel rank.",
        "candidate_fits": candidates,
    }, prediction


class TrajectoryMLP(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, 48),
            torch.nn.Tanh(),
            torch.nn.Linear(48, 48),
            torch.nn.Tanh(),
            torch.nn.Linear(48, channels),
        )

    def forward(self, time_input: torch.Tensor) -> torch.Tensor:
        return self.network(time_input)


def fit_mlp_baseline(data: dict) -> tuple[dict, np.ndarray]:
    normalized = (2.0 * data["times"] / HORIZON - 1.0).unsqueeze(1)
    target = data["observations"]
    best = None
    records = []
    for start in range(2):
        torch.manual_seed(data["seed"] + 701 * start)
        model = TrajectoryMLP(CHANNELS).to(DEVICE, dtype=DTYPE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-3, weight_decay=1.0e-6)
        for _ in range(1200):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(normalized)
            loss = (prediction[data["train_idx"]] - target[data["train_idx"]]).square().mean()
            loss.backward()
            optimizer.step()
        refiner = torch.optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=80,
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            refiner.zero_grad(set_to_none=True)
            local_prediction = model(normalized)
            local_loss = (
                local_prediction[data["train_idx"]] - target[data["train_idx"]]
            ).square().mean()
            local_loss.backward()
            return local_loss

        refiner.step(closure)
        with torch.no_grad():
            prediction = model(normalized)
            train_loss = float(
                (prediction[data["train_idx"]] - target[data["train_idx"]]).square().mean().cpu()
            )
        records.append({"start": start, "train_mse": train_loss})
        if best is None or train_loss < best[0]:
            best = (train_loss, prediction.detach().cpu().numpy())
    assert best is not None
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selection": "lowest training MSE across two fixed starts",
        "starts": records,
        "note": "The MLP estimates trajectories only and has no memory-rank output.",
    }, best[1]


def evaluate_case(case: str, repeat: int) -> dict:
    started = time.perf_counter()
    data = make_dataset(case, repeat)
    target = data["clean"].detach().cpu().numpy()
    indices = {
        "train": data["train_idx"].detach().cpu().numpy(),
        "interpolation": data["interpolation_idx"].detach().cpu().numpy(),
        "extrapolation": data["extrapolation_idx"].detach().cpu().numpy(),
    }
    methods = {}
    for name, fitter in (
        ("positive_real_memory", fit_mechanism),
        ("regularized_damped_modal", fit_modal_baseline),
        ("trajectory_mlp", fit_mlp_baseline),
    ):
        metadata, prediction = fitter(data)
        methods[name] = {
            **metadata,
            "train_rmse_to_clean": rmse(prediction[indices["train"]], target[indices["train"]]),
            "interpolation_rmse_to_clean": rmse(
                prediction[indices["interpolation"]], target[indices["interpolation"]]
            ),
            "extrapolation_rmse_to_clean": rmse(
                prediction[indices["extrapolation"]], target[indices["extrapolation"]]
            ),
        }
    return {
        "case": case,
        "repeat": repeat,
        "seed": data["seed"],
        "true_memory_rank": data["true_rank"],
        "train_points": len(indices["train"]),
        "interpolation_points": len(indices["interpolation"]),
        "extrapolation_points": len(indices["extrapolation"]),
        "methods": methods,
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(records: list[dict]) -> dict:
    methods = ("positive_real_memory", "regularized_damped_modal", "trajectory_mlp")
    rows = []
    for case in CASES:
        group = [record for record in records if record["case"] == case]
        row = {
            "case": case,
            "true_memory_rank": len(CASES[case]),
            "trials": len(group),
            "methods": {},
        }
        for method in methods:
            row["methods"][method] = {
                metric: float(np.median([record["methods"][method][metric] for record in group]))
                for metric in (
                    "train_rmse_to_clean",
                    "interpolation_rmse_to_clean",
                    "extrapolation_rmse_to_clean",
                )
            }
        row["memory_rank_recoveries"] = sum(
            record["methods"]["positive_real_memory"]["rank_recovered"] for record in group
        )
        row["mechanism_quality_failures"] = sum(
            record["methods"]["positive_real_memory"]["quality_failure"] for record in group
        )
        rows.append(row)
    return {"by_case": rows}


def assess(summary: dict) -> dict:
    rows = summary["by_case"]
    mechanism_extrapolation = np.array([
        row["methods"]["positive_real_memory"]["extrapolation_rmse_to_clean"] for row in rows
    ])
    modal_extrapolation = np.array([
        row["methods"]["regularized_damped_modal"]["extrapolation_rmse_to_clean"] for row in rows
    ])
    mlp_extrapolation = np.array([
        row["methods"]["trajectory_mlp"]["extrapolation_rmse_to_clean"] for row in rows
    ])
    checks = {
        "memory_rank_recovered_in_at_least_7_of_9_trials": sum(
            row["memory_rank_recoveries"] for row in rows
        ) >= 7,
        "mechanism_median_extrapolation_at_least_25_percent_better_than_mlp": (
            float(np.median(mechanism_extrapolation / mlp_extrapolation)) <= 0.75
        ),
        "mechanism_not_more_than_25_percent_worse_than_modal_on_median": (
            float(np.median(mechanism_extrapolation / modal_extrapolation)) <= 1.25
        ),
        "each_mlp_median_training_rmse_within_2p5_noise_std": all(
            row["methods"]["trajectory_mlp"]["train_rmse_to_clean"] <= 2.5 * NOISE_STD
            for row in rows
        ),
        "no_mechanism_quality_failures": sum(row["mechanism_quality_failures"] for row in rows) == 0,
    }
    return {
        "checks": checks,
        "route_pass": all(checks.values()),
        "median_extrapolation_ratio_mechanism_to_modal": float(
            np.median(mechanism_extrapolation / modal_extrapolation)
        ),
        "median_extrapolation_ratio_mechanism_to_mlp": float(
            np.median(mechanism_extrapolation / mlp_extrapolation)
        ),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "mechanism_vs_trajectory_baselines.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Mechanism model versus trajectory baselines",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        "- The Prony-like modal rank and MLP capacity are not interpreted as memory rank.",
        "- All metrics use the same sparse early-time training observations.",
        "",
        "| Case | True rank | Memory rank recovery | Memory extrap. | Modal extrap. | MLP extrap. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["by_case"]:
        methods = row["methods"]
        lines.append(
            f"| {row['case']} | {row['true_memory_rank']} | "
            f"{row['memory_rank_recoveries']}/{row['trials']} | "
            f"{methods['positive_real_memory']['extrapolation_rmse_to_clean']:.4e} | "
            f"{methods['regularized_damped_modal']['extrapolation_rmse_to_clean']:.4e} | "
            f"{methods['trajectory_mlp']['extrapolation_rmse_to_clean']:.4e} |"
        )
    lines.extend([
        "",
        "## Prespecified checks",
        "",
    ])
    for name, value in payload["assessment"]["checks"].items():
        lines.append(f"- {name}: **{value}**")
    lines.extend([
        "",
        "This is a feasibility comparison, not a calibrated claim of universal",
        "superiority. The direct modal baseline is a trajectory model, while the",
        "positive-real model carries a memory-mechanism contract and can estimate",
        "memory rank on its declared domain.",
    ])
    (RESULTS / "mechanism_vs_trajectory_baselines.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    records = []
    for case in CASES:
        for repeat in range(REPEATS):
            record = evaluate_case(case, repeat)
            records.append(record)
            print(
                f"{case} repeat={repeat} elapsed={record['elapsed_seconds']:.1f}s "
                f"rank={record['methods']['positive_real_memory']['selected_rank']}"
            )
    summary = summarize(records)
    payload = {
        "experiment": "mechanism_vs_trajectory_baselines",
        "design": {
            "cases": CASES,
            "repeats_per_case": REPEATS,
            "horizon": HORIZON,
            "num_points": NUM_POINTS,
            "channels": CHANNELS,
            "noise_std": NOISE_STD,
            "train_end_fraction": TRAIN_END_FRACTION,
            "train_count": TRAIN_COUNT,
            "comparison_boundary": (
                "Prediction errors are comparable; modal rank and MLP capacity are not memory rank."
            ),
        },
        "records": records,
        "summary": summary,
        "assessment": assess(summary),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_outputs(payload)
    print(json.dumps(payload["assessment"], indent=2))
    print(f"elapsed_seconds={payload['elapsed_seconds']:.1f}")


if __name__ == "__main__":
    main()
