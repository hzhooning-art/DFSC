"""Cross-cycle identification on public pilot-scale GeoTES measurements.

The first measured temperature channel is treated as an observed thermal
driver. The remaining three channels are modeled jointly. The protocol fits
the first heating/cooling cycle and evaluates transfer to the second cycle.
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


DATA = ROOT / "data" / "external" / "geotes" / "geotes_thermocouples.csv"
RESULTS = ROOT / "generated_results"
SEEDS = (0, 1, 2)
SAMPLES_PER_CYCLE = 56
STEPS = 320


def load_cycles() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    hours = np.asarray([float(row["elapsed_hours"]) for row in rows])
    values = np.asarray(
        [[float(row[f"Temperature{i}_celsius"]) for i in range(1, 5)] for row in rows]
    )
    rise = np.diff(values[:, 0], prepend=values[0, 0])
    candidates = np.flatnonzero((rise > 20.0) & (np.arange(rise.size) > 80))
    if candidates.size == 0:
        raise RuntimeError("second heating cycle was not detected")
    second_start = int(candidates[0])
    while second_start > 0 and rise[second_start - 1] > 4.0:
        second_start -= 1

    def sample(start: int, stop: int) -> dict[str, np.ndarray]:
        indices = np.unique(np.linspace(start, stop - 1, SAMPLES_PER_CYCLE).round().astype(int))
        t = hours[indices] - hours[indices[0]]
        temperature = values[indices]
        ambient = float(np.median(temperature[:3, 3]))
        scale = max(float(np.max(temperature[:, 0]) - ambient), 1.0)
        return {
            "time": t / max(float(t[-1]), 1e-12),
            "driver": (temperature[:, 0] - ambient) / scale,
            "response": (temperature[:, 1:] - ambient) / scale,
            "ambient_celsius": np.asarray(ambient),
            "scale_celsius": np.asarray(scale),
            "raw_time_hours": t,
        }

    return sample(0, second_start), sample(second_start, len(rows))


def interpolate_driver(query: torch.Tensor, times: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    query = query.clamp(times[0], times[-1])
    right = torch.searchsorted(times, query, right=True).clamp(1, times.numel() - 1)
    left = right - 1
    fraction = (query - times[left]) / (times[right] - times[left]).clamp_min(1e-12)
    return values[left] + fraction * (values[right] - values[left])


def forcing_tensor(times: torch.Tensor, driver: torch.Tensor, quadrature: torch.Tensor) -> torch.Tensor:
    physical = times[:, None] * quadrature[None, :]
    sampled = interpolate_driver(physical.reshape(-1), times, driver).reshape(times.numel(), -1)
    forcing = torch.zeros(times.numel(), quadrature.numel(), 3, dtype=times.dtype, device=times.device)
    forcing[:, :, 0] = sampled
    return forcing


class ResidualHead(nn.Module):
    def __init__(self, hidden: int = 20) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))

    def forward(self, time: torch.Tensor, driver: torch.Tensor) -> torch.Tensor:
        return self.net(torch.stack((time, driver), dim=-1))


class PureMLP(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.head = ResidualHead(hidden)

    def forward(self, time: torch.Tensor, driver: torch.Tensor) -> torch.Tensor:
        return self.head(time, driver)


class ForcedPropagation(nn.Module):
    def __init__(self, *, fractional: bool, hybrid: bool) -> None:
        super().__init__()
        path_laplacian = torch.tensor([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
        operator = path_laplacian + 0.20 * torch.eye(3)
        base = dfsc.build_operator_mlsl(operator, config=dfsc.MLSLConfig.stable(terms=72))
        self.layer = dfsc.ForcedMittagLefflerSpectralLayer(base, forcing_terms=72, ml_method="hybrid")
        self.raw_gain = nn.Parameter(torch.tensor(-0.2))
        self.raw_alpha = nn.Parameter(torch.tensor(0.2)) if fractional else None
        self.raw_beta = nn.Parameter(torch.tensor(0.0)) if fractional else None
        self.residual = ResidualHead(24) if hybrid else None
        self.residual_scale = 0.50

    @property
    def alpha(self) -> torch.Tensor:
        if self.raw_alpha is None:
            return torch.tensor(1.0, dtype=self.raw_gain.dtype, device=self.raw_gain.device)
        return 0.35 + 0.65 * torch.sigmoid(self.raw_alpha)

    @property
    def beta(self) -> torch.Tensor:
        if self.raw_beta is None:
            return torch.tensor(2.0, dtype=self.raw_gain.dtype, device=self.raw_gain.device)
        return 0.5 + 1.5 * torch.sigmoid(self.raw_beta)

    def forward(self, time: torch.Tensor, driver: torch.Tensor, initial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quadrature = (torch.arange(12, dtype=time.dtype, device=time.device) + 0.5) / 12.0
        forcing = F.softplus(self.raw_gain) * forcing_tensor(time, driver, quadrature)
        base = self.layer(initial, time, self.alpha, forcing, quadrature, beta=self.beta)
        if self.residual is None:
            return base, base
        gate = time[:, None]
        prediction = base + self.residual_scale * gate * torch.tanh(self.residual(time, driver))
        return prediction, base


def relative_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target).clamp_min(1e-12))


def fit_model(name: str, train: dict[str, np.ndarray], test: dict[str, np.ndarray], seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    dtype = torch.float64
    train_t = torch.tensor(train["time"], dtype=dtype)
    train_driver = torch.tensor(train["driver"], dtype=dtype)
    train_y = torch.tensor(train["response"], dtype=dtype)
    test_t = torch.tensor(test["time"], dtype=dtype)
    test_driver = torch.tensor(test["driver"], dtype=dtype)
    test_y = torch.tensor(test["response"], dtype=dtype)

    if name == "Pure MLP":
        model: nn.Module = PureMLP().to(dtype=dtype)
        predict = lambda t, d, initial: (model(t, d), model(t, d))
    else:
        model = ForcedPropagation(fractional=name != "Integer propagation", hybrid=name == "DFSC + residual MLP").to(dtype=dtype)
        predict = model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.018, weight_decay=1e-7)
    started = time.perf_counter()
    for _ in range(STEPS):
        optimizer.zero_grad()
        prediction, base = predict(train_t, train_driver, train_y[0])
        loss = torch.mean((prediction - train_y) ** 2)
        if name == "DFSC + residual MLP":
            loss = loss + 2e-4 * torch.mean((prediction - base) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        train_prediction, train_base = predict(train_t, train_driver, train_y[0])
        test_prediction, test_base = predict(test_t, test_driver, test_y[0])
        residual_fraction = float(torch.linalg.vector_norm(test_prediction - test_base) / torch.linalg.vector_norm(test_prediction).clamp_min(1e-12))
    alpha = float(model.alpha.detach()) if isinstance(model, ForcedPropagation) else None
    beta = float(model.beta.detach()) if isinstance(model, ForcedPropagation) else None
    return {
        "model": name,
        "seed": seed,
        "train_relative_error": relative_error(train_prediction, train_y),
        "cycle2_relative_error": relative_error(test_prediction, test_y),
        "cycle2_channel_errors": [relative_error(test_prediction[:, i], test_y[:, i]) for i in range(3)],
        "alpha": alpha,
        "beta": beta,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": elapsed,
        "residual_fraction": residual_fraction,
    }


def main() -> None:
    torch.set_default_dtype(torch.float64)
    train, test = load_cycles()
    rows = [
        fit_model(model, train, test, seed)
        for model in ("Integer propagation", "DFSC", "Pure MLP", "DFSC + residual MLP")
        for seed in SEEDS
    ]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    summary = []
    for model, model_rows in grouped.items():
        errors = np.asarray([float(row["cycle2_relative_error"]) for row in model_rows])
        summary.append({
            "model": model,
            "runs": len(model_rows),
            "cycle2_error_mean": float(errors.mean()),
            "cycle2_error_std": float(errors.std(ddof=1)),
            "parameter_count": int(model_rows[0]["parameter_count"]),
            "alpha_mean": None if model_rows[0]["alpha"] is None else float(np.mean([float(row["alpha"]) for row in model_rows])),
            "beta_mean": None if model_rows[0]["beta"] is None else float(np.mean([float(row["beta"]) for row in model_rows])),
        })
    payload = {
        "dataset": "GeoTES pilot-scale thermocouple histories",
        "source": "https://doi.org/10.5281/zenodo.18979098",
        "license": "CC BY 4.0",
        "protocol": "first measured heating/cooling cycle for identification; second cycle for transfer; T1 driver; T2-T4 responses; 56 samples per cycle; three seeds",
        "sensor_geometry_boundary": "No sensor coordinates are assumed; the three response channels use only their documented channel ordering.",
        "raw": rows,
        "summary": summary,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "real_geotes_cross_cycle_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = RESULTS / "real_geotes_cross_cycle_summary.csv"
    try:
        handle = csv_path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        csv_path = RESULTS / f"real_geotes_cross_cycle_summary_{int(time.time())}.csv"
        handle = csv_path.open("w", newline="", encoding="utf-8")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
