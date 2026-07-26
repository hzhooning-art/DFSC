"""Real geomembrane stress-relaxation benchmark from a CC BY 4.0 dataset."""

from __future__ import annotations

import csv
import io
import json
import math
import sys
import time
import zipfile
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


DATA = ROOT / "data" / "external" / "geomembrane" / "Effects-of-viscoelasticity.zip"
RESULTS = ROOT / "generated_results"
SEEDS = (0, 1, 2)
STRETCH_RATIOS = (1.001, 1.003, 1.005, 1.008)
NUM_POINTS = 120
TRAIN_FRACTION = 0.60


def load_relaxation(test_index: int) -> tuple[np.ndarray, np.ndarray]:
    member = f"Effects-of-viscoelasticity/Stress-relaxation/series0-test{test_index}.txt"
    with zipfile.ZipFile(DATA) as archive:
        lines = archive.read(member).decode("utf-8", "replace").splitlines()
    header = next(index for index, line in enumerate(lines) if line.startswith("X_Value"))
    values = np.genfromtxt(
        io.StringIO("\n".join(lines[header + 1 :])),
        delimiter=",",
        usecols=range(16),
        invalid_raise=False,
    )
    time_values = values[:, 0]
    stress = np.sum(values[:, 7:10], axis=1)
    finite = np.isfinite(time_values) & np.isfinite(stress)
    time_values, stress = time_values[finite], stress[finite]
    peak = int(np.argmax(stress[: min(stress.size, 300)]))
    time_values, stress = time_values[peak:] - time_values[peak], stress[peak:]

    # Ten-second medians suppress load-cell noise without changing the
    # multi-hour relaxation scale documented by the source dataset.
    bin_index = np.floor(time_values / 10.0).astype(int)
    unique_bins = np.unique(bin_index)
    binned_time = np.asarray([np.median(time_values[bin_index == index]) for index in unique_bins])
    binned_stress = np.asarray([np.median(stress[bin_index == index]) for index in unique_bins])
    sample_indices = np.unique(np.linspace(0, binned_time.size - 1, NUM_POINTS).round().astype(int))
    sampled_time, sampled_stress = binned_time[sample_indices], binned_stress[sample_indices]
    train_count = int(TRAIN_FRACTION * sampled_time.size)
    initial = float(np.median(sampled_stress[:3]))
    train_floor = float(np.median(sampled_stress[max(0, train_count - 8) : train_count]))
    scale = max(abs(initial - train_floor), 1e-8)
    normalized = (sampled_stress - train_floor) / scale
    return sampled_time, normalized


def bounded_alpha(raw: torch.Tensor) -> torch.Tensor:
    return 0.2 + torch.sigmoid(raw)


class CurveMLP(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(-1)).squeeze(-1)


def relative_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target).clamp_min(1e-12))


def fit_parametric(
    time_tensor: torch.Tensor,
    target: torch.Tensor,
    full_time: torch.Tensor,
    *,
    family: str,
    seed: int,
) -> tuple[torch.Tensor, float, float | None, int, float]:
    torch.manual_seed(seed)
    raw_rate = nn.Parameter(torch.tensor(-1.0 + 0.2 * seed))
    offset = nn.Parameter(torch.tensor(0.0))
    amplitude = nn.Parameter(torch.tensor(1.0))
    parameters: list[nn.Parameter] = [raw_rate, offset, amplitude]
    raw_alpha: nn.Parameter | None = None
    if family != "exponential":
        raw_alpha = nn.Parameter(torch.tensor(0.4 + 0.1 * seed))
        parameters.append(raw_alpha)
    optimizer = torch.optim.Adam(parameters, lr=0.025)

    def predict(t: torch.Tensor) -> torch.Tensor:
        rate = F.softplus(raw_rate)
        if family == "exponential":
            kernel = torch.exp(-rate * t)
        elif family == "stretched":
            alpha = bounded_alpha(raw_alpha)
            kernel = torch.exp(-rate * t.pow(alpha))
        elif family == "mlsl":
            alpha = bounded_alpha(raw_alpha)
            kernel = dfsc.mittag_leffler_e(alpha, -rate * t.pow(alpha), terms=120, method="hybrid")
        else:
            raise ValueError(family)
        return offset + amplitude * kernel

    started = time.perf_counter()
    for _ in range(650):
        optimizer.zero_grad()
        loss = torch.mean((predict(time_tensor) - target) ** 2)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        prediction = predict(full_time)
        alpha_value = 1.0 if raw_alpha is None else float(bounded_alpha(raw_alpha))
        rate_value = float(F.softplus(raw_rate))
    return prediction, rate_value, alpha_value, len(parameters), elapsed


def fit_neural(
    x_train: torch.Tensor,
    target: torch.Tensor,
    x_full: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, int, float]:
    torch.manual_seed(seed)
    model = CurveMLP().to(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
    started = time.perf_counter()
    for _ in range(900):
        optimizer.zero_grad()
        loss = torch.mean((model(x_train) - target) ** 2)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        prediction = model(x_full)
    return prediction, sum(parameter.numel() for parameter in model.parameters()), elapsed


def fit_hybrid(
    time_train: torch.Tensor,
    x_train: torch.Tensor,
    target: torch.Tensor,
    time_full: torch.Tensor,
    x_full: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, float, float, int, float, float]:
    torch.manual_seed(seed)
    raw_alpha = nn.Parameter(torch.tensor(0.4))
    raw_rate = nn.Parameter(torch.tensor(-1.0))
    offset = nn.Parameter(torch.tensor(0.0))
    amplitude = nn.Parameter(torch.tensor(1.0))
    residual = CurveMLP(hidden=16).to(dtype=torch.float64)
    parameters = [raw_alpha, raw_rate, offset, amplitude, *residual.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=0.01, weight_decay=1e-7)

    def predict(t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = bounded_alpha(raw_alpha)
        rate = F.softplus(raw_rate)
        base = offset + amplitude * dfsc.mittag_leffler_e(
            alpha, -rate * t.pow(alpha), terms=120, method="hybrid"
        )
        return base + 0.15 * residual(x), base

    started = time.perf_counter()
    for _ in range(900):
        optimizer.zero_grad()
        prediction, base = predict(time_train, x_train)
        loss = torch.mean((prediction - target) ** 2) + 2e-4 * torch.mean((prediction - base) ** 2)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        prediction, base = predict(time_full, x_full)
        residual_fraction = float(torch.linalg.vector_norm(prediction - base) / torch.linalg.vector_norm(prediction))
    return (
        prediction,
        float(bounded_alpha(raw_alpha).detach()),
        float(F.softplus(raw_rate).detach()),
        len(parameters),
        elapsed,
        residual_fraction,
    )


def main() -> None:
    torch.set_default_dtype(torch.float64)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for test_index, stretch_ratio in enumerate(STRETCH_RATIOS, start=1):
        times, normalized_stress = load_relaxation(test_index)
        count = len(times)
        train_count = int(TRAIN_FRACTION * count)
        time_scale = max(times[train_count - 1], 1.0)
        dimensionless_time = torch.tensor(times / time_scale, dtype=torch.float64)
        target = torch.tensor(normalized_stress, dtype=torch.float64)
        x = torch.log1p(dimensionless_time) / math.log1p(float(dimensionless_time[-1]))
        for seed in SEEDS:
            candidates: list[tuple[str, torch.Tensor, float | None, float | None, int, float, float]] = []
            for family, name in (
                ("exponential", "Exponential"),
                ("stretched", "Stretched exponential"),
                ("mlsl", "Fractional Zener / MLSL"),
            ):
                prediction, rate, alpha, parameters, elapsed = fit_parametric(
                    dimensionless_time[:train_count], target[:train_count], dimensionless_time, family=family, seed=seed
                )
                candidates.append((name, prediction, alpha, rate, parameters, elapsed, 0.0))
            prediction, parameters, elapsed = fit_neural(x[:train_count], target[:train_count], x, seed=seed)
            candidates.append(("Pure MLP", prediction, None, None, parameters, elapsed, 1.0))
            prediction, alpha, rate, parameters, elapsed, residual_fraction = fit_hybrid(
                dimensionless_time[:train_count], x[:train_count], target[:train_count], dimensionless_time, x, seed=seed
            )
            candidates.append(("MLSL + residual MLP", prediction, alpha, rate, parameters, elapsed, residual_fraction))
            for model, prediction, alpha, rate, parameters, elapsed, residual_fraction in candidates:
                rows.append(
                    {
                        "test": test_index,
                        "stretch_ratio": stretch_ratio,
                        "seed": seed,
                        "model": model,
                        "train_relative_error": relative_error(prediction[:train_count], target[:train_count]),
                        "extrapolation_relative_error": relative_error(prediction[train_count:], target[train_count:]),
                        "alpha": alpha,
                        "rate_per_train_horizon": rate,
                        "parameter_count": parameters,
                        "training_seconds": elapsed,
                        "residual_fraction": residual_fraction,
                    }
                )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    summary_rows = []
    for model, model_rows in grouped.items():
        errors = np.asarray([float(row["extrapolation_relative_error"]) for row in model_rows])
        summary_rows.append(
            {
                "model": model,
                "runs": len(model_rows),
                "tasks": len({row["test"] for row in model_rows}),
                "extrapolation_error_mean": float(np.mean(errors)),
                "extrapolation_error_std": float(np.std(errors, ddof=1)),
                "extrapolation_error_median": float(np.median(errors)),
            }
        )
    for filename, output_rows in (
        ("real_geomembrane_relaxation_raw.csv", rows),
        ("real_geomembrane_relaxation_summary.csv", summary_rows),
    ):
        with (RESULTS / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    payload = {
        "dataset": "Effects of viscoelastic properties on geomembrane mechanics and deformations",
        "source": "https://doi.org/10.5281/zenodo.14329961",
        "license": "CC BY 4.0",
        "md5": "557d7f5b4996234ff0c37493fceb91fd",
        "tasks": [f"stress relaxation at stretch ratio {ratio}" for ratio in STRETCH_RATIOS],
        "protocol": "10-second median bins; 120 uniform-time samples; first 60% train, final 40% extrapolation",
        "summary": summary_rows,
    }
    (RESULTS / "real_geomembrane_relaxation_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
