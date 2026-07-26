"""Training-budget sensitivity for the diagnostic FNO and DeepONet controls.

This experiment keeps the architecture, data, optimizer, and initialization
fixed and snapshots each neural baseline at 1x, 2x, and 4x the update budget
used in the main hybrid-composition experiment.  It is a fairness diagnostic,
not a claim that these compact controls represent state-of-the-art tuning.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dfsc import DeepONet1D, FNO1D, MLSLConfig, build_dirichlet_mlsl_1d
from exp29_hybrid_backbone_baseline import flatten_hybrid_dataset, make_random_initial_conditions, rel


RESULTS = ROOT / "revision_results"
SEEDS = (0, 1, 2)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def dataset(seed: int):
    torch.manual_seed(7300 + seed)
    x, layer = build_dirichlet_mlsl_1d(
        num_points=40,
        num_modes=16,
        config=MLSLConfig.stable(terms=120),
    )
    alpha = torch.tensor(1.25)
    gamma = 2.5
    train_times = torch.linspace(0.0, 0.055, 7)
    test_times = torch.linspace(0.12, 0.32, 7)
    train_u0 = make_random_initial_conditions(x, 18)
    test_u0 = make_random_initial_conditions(x, 6)
    train_u0_rows, train_t_rows, _, train_y = flatten_hybrid_dataset(
        train_u0, train_times, layer, alpha, gamma
    )
    test_u0_rows, test_t_rows, _, test_y = flatten_hybrid_dataset(
        test_u0, test_times, layer, alpha, gamma
    )
    return x, alpha, train_u0_rows, train_t_rows, train_y, test_u0_rows, test_t_rows, test_y


def fno_rows(seed: int, data) -> list[dict[str, object]]:
    x, _, train_u0, train_t, train_y, test_u0, test_t, test_y = data
    torch.manual_seed(9100 + seed)
    model = FNO1D(modes=12, width=32, layers=3).to(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    snapshots = {90: "1x", 180: "2x", 360: "4x"}
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for step in range(1, max(snapshots) + 1):
        optimizer.zero_grad()
        prediction = model(train_u0, train_t)
        loss = torch.mean((prediction - train_y) ** 2)
        loss.backward()
        optimizer.step()
        if step in snapshots:
            with torch.no_grad():
                train_prediction = model(train_u0, train_t)
                test_prediction = model(test_u0, test_t)
            rows.append(
                {
                    "seed": seed,
                    "model": "FNO1D",
                    "budget": snapshots[step],
                    "updates": step,
                    "train_samples": train_u0.shape[0],
                    "parameters": parameter_count(model),
                    "learning_rate": 2e-3,
                    "train_relative_error": rel(train_prediction, train_y),
                    "long_time_relative_error": rel(test_prediction, test_y),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    return rows


def deeponet_rows(seed: int, data) -> list[dict[str, object]]:
    x, alpha, train_u0, train_t, train_y, test_u0, test_t, test_y = data
    torch.manual_seed(10100 + seed)
    x_train = x[None, :].expand(train_u0.shape[0], -1)
    t_train = train_t[:, None].expand(-1, x.numel())
    x_test = x[None, :].expand(test_u0.shape[0], -1)
    t_test = test_t[:, None].expand(-1, x.numel())
    alpha_train = alpha.expand(train_u0.shape[0])
    alpha_test = alpha.expand(test_u0.shape[0])
    y_mean = train_y.mean()
    y_std = train_y.std().clamp_min(1e-12)
    target = (train_y - y_mean) / y_std

    model = DeepONet1D(num_points=x.numel(), latent=64, hidden=96).to(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    snapshots = {120: "1x", 240: "2x", 480: "4x"}
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for step in range(1, max(snapshots) + 1):
        optimizer.zero_grad()
        prediction = model(train_u0, x_train, t_train, alpha_train)
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        optimizer.step()
        if step in snapshots:
            with torch.no_grad():
                train_prediction = model(train_u0, x_train, t_train, alpha_train) * y_std + y_mean
                test_prediction = model(test_u0, x_test, t_test, alpha_test) * y_std + y_mean
            rows.append(
                {
                    "seed": seed,
                    "model": "DeepONet1D",
                    "budget": snapshots[step],
                    "updates": step,
                    "train_samples": train_u0.shape[0],
                    "parameters": parameter_count(model),
                    "learning_rate": 1e-3,
                    "train_relative_error": rel(train_prediction, train_y),
                    "long_time_relative_error": rel(test_prediction, test_y),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for model in ("FNO1D", "DeepONet1D"):
        for budget in ("1x", "2x", "4x"):
            selected = [row for row in rows if row["model"] == model and row["budget"] == budget]
            test_values = [float(row["long_time_relative_error"]) for row in selected]
            train_values = [float(row["train_relative_error"]) for row in selected]
            test_mean = sum(test_values) / len(test_values)
            summary.append(
                {
                    "model": model,
                    "budget": budget,
                    "updates": selected[0]["updates"],
                    "train_samples": selected[0]["train_samples"],
                    "parameters": selected[0]["parameters"],
                    "train_error_mean": sum(train_values) / len(train_values),
                    "long_time_error_mean": test_mean,
                    "long_time_error_std": math.sqrt(
                        sum((value - test_mean) ** 2 for value in test_values) / max(len(test_values) - 1, 1)
                    ),
                }
            )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        data = dataset(seed)
        rows.extend(fno_rows(seed, data))
        rows.extend(deeponet_rows(seed, data))
    summary = summarize(rows)
    write_csv(RESULTS / "neural_budget_sensitivity_raw.csv", rows)
    write_csv(RESULTS / "neural_budget_sensitivity_summary.csv", summary)
    payload = {
        "scope": "diagnostic controls on the matched known-propagator task",
        "fairness_boundary": (
            "The sweep tests sensitivity to update count while holding architecture, optimizer, and data fixed; "
            "it is not an exhaustive hyperparameter search or a claim of general neural-operator superiority."
        ),
        "summary": summary,
    }
    (RESULTS / "neural_budget_sensitivity_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
