"""Real spatial heat-transport OOD benchmark from Zenodo 15064388."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dfsc

DATA = ROOT / "data" / "external" / "heated_steam" / "heated_steam_profiles.csv"
RESULTS = ROOT / "revision_results"
HELD_OUT = {5, 8, 12, 16}
SEEDS = (0, 1, 2)


def load_data():
    rows = []
    with DATA.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    # Keep every fifth minute sample; no interpolation or smoothing is applied.
    rows = [row for index, row in enumerate(rows) if int(round(row["time_h"] * 60)) % 5 == 0]
    condition_keys = ("inflow_1e-5_kg_s", "inlet_temperature_k", "column_height_m", "vwc")
    conditions = torch.tensor([[row[key] for key in condition_keys] for row in rows])
    depth = torch.tensor([row["depth_m"] for row in rows])
    time = torch.tensor([row["time_h"] for row in rows])
    temperature = torch.tensor([row["temperature_k"] for row in rows])
    experiment = torch.tensor([int(row["experiment"]) for row in rows])
    return conditions, depth, time, temperature, experiment


class NeuralField(nn.Module):
    def __init__(self, inputs: int = 6, hidden: int = 64, *, zero_output: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        if zero_output:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class FractionalForcedField(nn.Module):
    def __init__(self, modes: int = 8, *, fixed_alpha: float | None = None, residual: bool = False) -> None:
        super().__init__()
        self.modes = modes
        self.fixed_alpha = fixed_alpha
        self.raw_alpha = nn.Parameter(torch.tensor(0.0)) if fixed_alpha is None else None
        self.raw_diffusivity = nn.Parameter(torch.tensor(-2.0))
        self.forcing = nn.Linear(4, modes)
        self.initial = nn.Linear(4, 1)
        self.residual = NeuralField(hidden=32, zero_output=True) if residual else None

    @property
    def alpha(self):
        return torch.as_tensor(self.fixed_alpha) if self.fixed_alpha is not None else 0.45 + 0.75 * torch.sigmoid(self.raw_alpha)

    def forward(self, condition: torch.Tensor, x: torch.Tensor, t: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        k = torch.arange(self.modes, dtype=x.dtype, device=x.device)
        basis = torch.cos(torch.pi * x[:, None] * k)
        eigenvalues = (torch.pi * k).square()
        diffusivity = 0.002 + 0.198 * torch.sigmoid(self.raw_diffusivity)
        alpha = self.alpha.to(dtype=x.dtype, device=x.device)
        ta = t.clamp_min(torch.finfo(t.dtype).tiny).pow(alpha)
        z = -diffusivity * eigenvalues * ta[:, None]
        response = ta[:, None] * dfsc.mittag_leffler_e_ab(
            alpha, alpha + 1.0, z, terms=100, method="hybrid"
        )
        base = self.initial(condition).squeeze(-1) + torch.sum(
            basis * response * self.forcing(condition), dim=-1
        )
        if self.residual is None:
            return base
        gate = 1.0 - torch.exp(-5.0 * t)
        return base + gate * self.residual(features)


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    rmse = float(torch.sqrt(torch.mean((prediction - target) ** 2)))
    mae = float(torch.mean(torch.abs(prediction - target)))
    return rmse, mae


def train_model(kind: str, seed: int, tensors):
    condition, x, t, y, features, train_mask, test_mask = tensors
    torch.manual_seed(9000 + seed)
    if kind == "integer": model = FractionalForcedField(fixed_alpha=1.0)
    elif kind == "dfsc": model = FractionalForcedField()
    elif kind == "hybrid": model = FractionalForcedField(residual=True)
    else: model = NeuralField()
    train_indices = torch.nonzero(train_mask).squeeze(-1)
    generator = torch.Generator().manual_seed(12000 + seed)

    def optimize(parameters, steps: int) -> None:
        parameters = list(parameters)
        optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-6)
        for _ in range(steps):
            selected = train_indices[torch.randint(train_indices.numel(), (1024,), generator=generator)]
            optimizer.zero_grad()
            if kind == "mlp":
                prediction = model(features[selected])
            else:
                prediction = model(condition[selected], x[selected], t[selected], features[selected])
            loss = torch.mean((prediction - y[selected]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 2.0)
            optimizer.step()

    if kind == "hybrid":
        assert model.residual is not None
        residual_parameters = list(model.residual.parameters())
        for parameter in residual_parameters:
            parameter.requires_grad_(False)
        optimize((parameter for parameter in model.parameters() if parameter.requires_grad), 700)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in residual_parameters:
            parameter.requires_grad_(True)
        optimize(residual_parameters, 1200)
    else:
        optimize(model.parameters(), 900)
    with torch.no_grad():
        prediction = model(features) if kind == "mlp" else model(condition, x, t, features)
    test_rmse, test_mae = metrics(prediction[test_mask], y[test_mask])
    train_rmse, _ = metrics(prediction[train_mask], y[train_mask])
    alpha = None if kind == "mlp" else float(model.alpha.detach())
    return {
        "seed": seed,
        "model": kind,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_rmse_k": train_rmse,
        "heldout_rmse_k": test_rmse,
        "heldout_mae_k": test_mae,
        "alpha": alpha,
    }


def main() -> None:
    torch.set_default_dtype(torch.float64)
    conditions, depth, time, temperature, experiment = load_data()
    train_mask = torch.tensor([int(value) not in HELD_OUT for value in experiment])
    test_mask = ~train_mask
    condition_mean = conditions[train_mask].mean(0)
    condition_std = conditions[train_mask].std(0).clamp_min(1e-8)
    condition = (conditions - condition_mean) / condition_std
    x = (0.30 - depth) / 0.27
    time_scale = time[train_mask].max()
    t = time / time_scale
    y = temperature - 298.15
    features = torch.cat([condition, x[:, None], t[:, None]], dim=1)
    tensors = condition, x, t, y, features, train_mask, test_mask
    rows = [train_model(kind, seed, tensors) for seed in SEEDS for kind in ("integer", "dfsc", "mlp", "hybrid")]
    summary = []
    for kind in ("integer", "dfsc", "mlp", "hybrid"):
        selected = [row for row in rows if row["model"] == kind]
        item = {"model": kind, "parameters": selected[0]["parameters"]}
        for metric in ("train_rmse_k", "heldout_rmse_k", "heldout_mae_k"):
            values = [float(row[metric]) for row in selected]
            mean = sum(values) / len(values)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        alpha_values = [row["alpha"] for row in selected if row["alpha"] is not None]
        if alpha_values: item["alpha_mean"] = sum(alpha_values) / len(alpha_values)
        summary.append(item)
    payload = {
        "dataset_doi": "10.5281/zenodo.15064388",
        "license": "CC BY 4.0",
        "held_out_experiments": sorted(HELD_OUT),
        "held_out_rule": "largest flow, inlet temperature, column height, and water content",
        "train_rows": int(train_mask.sum()), "test_rows": int(test_mask.sum()),
        "scope": "condition OOD on measured spatial temperature profiles; no interpolation or smoothing",
        "raw": rows, "summary": summary,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "real_heated_steam.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
