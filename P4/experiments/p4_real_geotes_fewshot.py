"""Few-shot cross-cycle transfer on the public GeoTES measurements."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "P1" / "paper1_mlsl"
sys.path.insert(0, str(P1))
from experiments.exp48_real_geotes_cross_cycle import fit_model, load_cycles  # noqa: E402


OUT = ROOT / "P4" / "results" / "p4_real_geotes_fewshot.json"
SEEDS = (0, 1, 2)
FRACTIONS = (0.2, 0.4, 0.6, 1.0)


def sliced_cycle(cycle: dict[str, np.ndarray], fraction: float) -> dict[str, np.ndarray]:
    count = max(4, int(round(len(cycle["time"]) * fraction)))
    indices = np.unique(np.linspace(0, len(cycle["time"]) - 1, count).round().astype(int))
    result = {}
    for key, value in cycle.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == len(cycle["time"]):
            result[key] = value[indices].copy()
        else:
            result[key] = value
    return result


def main() -> None:
    torch.set_default_dtype(torch.float64)
    train, test = load_cycles()
    rows = []
    for fraction in FRACTIONS:
        train_subset = sliced_cycle(train, fraction)
        for seed in SEEDS:
            for model in ("Integer propagation", "DFSC", "Pure MLP", "DFSC + residual MLP"):
                started = time.perf_counter()
                row = fit_model(model, train_subset, test, seed)
                row["train_fraction"] = fraction
                row["elapsed_wall_seconds"] = time.perf_counter() - started
                rows.append(row)
    summary = []
    for fraction in FRACTIONS:
        for model in ("Integer propagation", "DFSC", "Pure MLP", "DFSC + residual MLP"):
            selected = [row for row in rows if row["train_fraction"] == fraction and row["model"] == model]
            errors = np.asarray([row["cycle2_relative_error"] for row in selected], dtype=float)
            summary.append({
                "train_fraction": fraction,
                "model": model,
                "runs": len(selected),
                "cycle2_error_mean": float(errors.mean()),
                "cycle2_error_std": float(errors.std(ddof=1)),
                "parameter_count": int(selected[0]["parameter_count"]),
            })
    payload = {
        "schema": "DFSC-P4-GeoTES-FewShot-v1",
        "dataset": "GeoTES pilot-scale thermocouple histories",
        "source": "https://doi.org/10.5281/zenodo.18979098",
        "license": "CC BY 4.0",
        "protocol": "first measured cycle truncated to the declared fraction; second measured cycle held out for transfer; T1 driver and T2-T4 responses; three seeds",
        "fractions": list(FRACTIONS),
        "raw": rows,
        "summary": summary,
        "interpretation": "Few-shot transfer is reported as a conditional structural-prior test; it is not a universal claim about the best predictor.",
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
