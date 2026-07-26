"""Sample-efficiency and structured OOD diagnostics for DFSC residual learning."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dfsc

RESULTS = ROOT / "revision_results"
SEEDS = (0, 1, 2)
SAMPLE_SIZES = (16, 32, 64, 128)


class FieldNet(nn.Module):
    def __init__(self, inputs: int, outputs: int, *, residual: bool) -> None:
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(inputs, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, outputs)
        )

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def sample(layer, x, count: int, ranges: dict[str, tuple[float, float]], seed: int):
    generator = torch.Generator().manual_seed(seed)
    coefficients = torch.randn(count, 3, generator=generator)
    modes = torch.stack([torch.sin((k + 1) * torch.pi * x) for k in range(3)])
    u0 = coefficients @ modes
    alpha = torch.empty(count).uniform_(*ranges["alpha"], generator=generator)
    time = torch.empty(count).uniform_(*ranges["time"], generator=generator)
    forcing = torch.empty(count).uniform_(*ranges["forcing"], generator=generator)
    base = torch.stack([layer(u0[i], time[i], alpha[i]) for i in range(count)])
    correction = forcing[:, None] * (1.0 - torch.exp(-4.0 * time[:, None])) * torch.sin(2 * torch.pi * x)
    correction = correction + 0.04 * time[:, None] * base.square()
    target = base + correction
    features = torch.cat([u0, time[:, None], alpha[:, None], forcing[:, None]], dim=1)
    return features, base, target


def relative_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target))


def train(model, train_data, test_sets):
    features, base, target = train_data
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0).clamp_min(1e-6)
    learned_target = target - base if model.residual else target
    target_mean = learned_target.mean(dim=0)
    target_std = learned_target.std(dim=0).clamp_min(1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-6)
    for _ in range(500):
        optimizer.zero_grad()
        normalized = model((features - feature_mean) / feature_std, base)
        loss = torch.mean((normalized - (learned_target - target_mean) / target_std) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        results = {}
        for name, (test_features, test_base, test_target) in test_sets.items():
            learned = model((test_features - feature_mean) / feature_std, test_base) * target_std + target_mean
            prediction = test_base + learned if model.residual else learned
            results[name] = relative_error(prediction, test_target)
        return results


def main() -> None:
    torch.set_default_dtype(torch.float64)
    RESULTS.mkdir(parents=True, exist_ok=True)
    x, layer = dfsc.build_dirichlet_mlsl_1d(
        num_points=24, num_modes=4, config=dfsc.MLSLConfig.stable(terms=100)
    )
    train_ranges = {"alpha": (0.75, 0.95), "time": (0.005, 0.03), "forcing": (0.0, 0.2)}
    scenarios = {
        "iid": train_ranges,
        "time_ood": {**train_ranges, "time": (0.04, 0.08)},
        "alpha_ood": {**train_ranges, "alpha": (1.05, 1.25)},
        "forcing_ood": {**train_ranges, "forcing": (0.25, 0.40)},
        "joint_ood": {"alpha": (1.05, 1.25), "time": (0.04, 0.08), "forcing": (0.25, 0.40)},
    }
    rows = []
    for seed in SEEDS:
        test_sets = {
            name: sample(layer, x, 128, ranges, 50000 + 1000 * seed + index)
            for index, (name, ranges) in enumerate(scenarios.items())
        }
        train_pool = sample(layer, x, max(SAMPLE_SIZES), train_ranges, 10000 + seed)
        for count in SAMPLE_SIZES:
            train_data = tuple(value[:count] for value in train_pool)
            for residual, model_name in ((False, "pure_mlp"), (True, "dfsc_residual")):
                torch.manual_seed(20000 + 100 * seed + count + int(residual))
                model = FieldNet(27, 24, residual=residual)
                errors = train(model, train_data, test_sets)
                rows.append({"seed": seed, "train_samples": count, "model": model_name, **errors})
    summary = []
    for count in SAMPLE_SIZES:
        for model_name in ("pure_mlp", "dfsc_residual"):
            selected = [row for row in rows if row["train_samples"] == count and row["model"] == model_name]
            item = {"train_samples": count, "model": model_name}
            for scenario in scenarios:
                values = [float(row[scenario]) for row in selected]
                mean = sum(values) / len(values)
                item[f"{scenario}_mean"] = mean
                item[f"{scenario}_std"] = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            summary.append(item)
    payload = {
        "scope": "controlled structured-propagator task with an unmodeled forcing/nonlinear correction",
        "seeds": list(SEEDS),
        "train_sample_sizes": list(SAMPLE_SIZES),
        "raw": rows,
        "summary": summary,
    }
    (RESULTS / "sample_ood_matrix.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
