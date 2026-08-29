"""Calibrate near-boundary refusal over noise and observation horizon.

This experiment conditions on the rank-one candidate selected throughout the
refined boundary scan.  It calibrates the contract diagnostics, not rank selection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank, lifted_response
from probe_out_of_class_refusal import mean_lag1, oscillatory_response, prediction_from_fit


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPEATS = 4


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def generate(
    family: str,
    strength: float,
    horizon: float,
    noise_std: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    channels = 6
    times = torch.linspace(0.0, horizon, 97, dtype=DTYPE, device=DEVICE)
    scale = torch.linspace(0.72, 1.28, channels, dtype=DTYPE, device=DEVICE)
    if family == "signed_residue":
        weights = torch.stack([0.46 * scale, -strength / scale], dim=1)
        clean = lifted_response(
            times,
            weights,
            torch.tensor([0.22, 1.35], dtype=DTYPE, device=DEVICE),
        )
    elif family == "oscillation":
        clean = oscillatory_response(times, 0.38 * scale, decay=0.24, frequency=strength)
    else:
        raise ValueError(f"Unknown family: {family}")

    observations = clean + noise_std * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )
    train_pool = np.arange(1, 70)
    train_np = np.sort(rng.choice(train_pool, size=48, replace=False))
    val_np = np.arange(70, times.numel())
    return (
        times,
        observations,
        torch.tensor(train_np, dtype=torch.long, device=DEVICE),
        torch.tensor(val_np, dtype=torch.long, device=DEVICE),
    )


def evaluate(
    family: str,
    level: str,
    strength: float,
    horizon: float,
    noise_std: float,
    repeat: int,
) -> dict:
    seed = (
        17000
        + repeat
        + int(round(strength * 100000))
        + int(round(horizon * 10))
        + int(round(noise_std * 1.0e7))
        + sum(map(ord, family))
    )
    times, observations, train_idx, val_idx = generate(
        family, strength, horizon, noise_std, seed
    )
    fit = fit_rank(
        times,
        observations,
        train_idx,
        val_idx,
        rank=1,
        seed=seed * 100,
        adam_steps=190,
        lbfgs_steps=55,
    )
    residual = prediction_from_fit(times, fit)[val_idx] - observations[val_idx]
    lag1 = mean_lag1(residual)
    prediction_ok = fit.val_rmse <= max(4.0 * noise_std, 3.0e-3)
    condition_ok = fit.jacobian_condition <= 1.0e8
    residual_ok = abs(lag1) <= 0.55
    accepted = prediction_ok and condition_ok and residual_ok
    return {
        "family": family,
        "level": level,
        "strength": strength,
        "horizon": horizon,
        "noise_std": noise_std,
        "repeat": repeat,
        "decision": "ACCEPT_CONTRACT" if accepted else "REFUSE_CONTRACT",
        "validation_rmse": fit.val_rmse,
        "jacobian_condition": fit.jacobian_condition,
        "validation_residual_lag1": lag1,
        "failed_gates": [
            name for name, passed in {
                "prediction": prediction_ok,
                "condition": condition_ok,
                "residual": residual_ok,
            }.items() if not passed
        ],
    }


def summarize_cells(records: list[dict]) -> list[dict]:
    keys = sorted({
        (row["family"], row["level"], row["strength"], row["horizon"], row["noise_std"])
        for row in records
    })
    rows = []
    for family, level, strength, horizon, noise_std in keys:
        group = [
            row for row in records
            if (row["family"], row["level"], row["strength"], row["horizon"], row["noise_std"])
            == (family, level, strength, horizon, noise_std)
        ]
        refusals = sum(row["decision"] == "REFUSE_CONTRACT" for row in group)
        lower, upper = wilson_interval(refusals, len(group))
        rows.append({
            "family": family,
            "level": level,
            "strength": strength,
            "horizon": horizon,
            "noise_std": noise_std,
            "trials": len(group),
            "refusal_fraction": refusals / len(group),
            "refusal_wilson95": [lower, upper],
            "median_validation_rmse": float(np.median([row["validation_rmse"] for row in group])),
            "median_abs_residual_lag1": float(np.median([
                abs(row["validation_residual_lag1"]) for row in group
            ])),
        })
    return rows


def aggregate(records: list[dict]) -> list[dict]:
    rows = []
    for family in ("signed_residue", "oscillation"):
        for level in ("zero", "below", "above"):
            group = [row for row in records if row["family"] == family and row["level"] == level]
            refusals = sum(row["decision"] == "REFUSE_CONTRACT" for row in group)
            lower, upper = wilson_interval(refusals, len(group))
            rows.append({
                "family": family,
                "level": level,
                "trials": len(group),
                "refusal_fraction": refusals / len(group),
                "refusal_wilson95": [lower, upper],
            })
    return rows


def assess(aggregate_rows: list[dict], cells: list[dict]) -> dict:
    lookup = {(row["family"], row["level"]): row for row in aggregate_rows}
    families = {}
    for family in ("signed_residue", "oscillation"):
        zero = lookup[(family, "zero")]["refusal_fraction"]
        below = lookup[(family, "below")]["refusal_fraction"]
        above = lookup[(family, "above")]["refusal_fraction"]
        strongest_cell = next(
            row for row in cells
            if row["family"] == family
            and row["level"] == "above"
            and row["horizon"] == 20.0
            and row["noise_std"] == 3.0e-4
        )
        families[family] = {
            "zero_false_refusal_at_most_0.10": zero <= 0.10,
            "above_more_detectable_than_below": above > below,
            "low_noise_long_horizon_above_refusal_at_least_0.75": (
                strongest_cell["refusal_fraction"] >= 0.75
            ),
            "aggregate_refusal": {"zero": zero, "below": below, "above": above},
        }
    return {
        "families": families,
        "route_pass": all(all(
            value for key, value in result.items() if key != "aggregate_refusal"
        ) for result in families.values()),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "refusal_calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Refusal calibration over noise and horizon",
        "",
        "This scan conditions on a rank-one fit and calibrates the diagnostic contract.",
        "",
        "## Aggregate rates",
        "",
        "| Family | Level | Trials | Refusal fraction | Wilson 95% interval |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lower, upper = row["refusal_wilson95"]
        lines.append(
            f"| {row['family']} | {row['level']} | {row['trials']} | "
            f"{row['refusal_fraction']:.3f} | [{lower:.3f}, {upper:.3f}] |"
        )
    lines.extend([
        "",
        f"Route pass: **{payload['assessment']['route_pass']}**.",
        "",
        "Four repeats per cell are sufficient for route selection but not for final",
        "frequentist calibration or publication-level error-rate claims.",
    ])
    (RESULTS / "refusal_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    levels = {
        "signed_residue": {"zero": 0.0, "below": 0.005, "above": 0.0075},
        "oscillation": {"zero": 0.0, "below": 0.03, "above": 0.04},
    }
    horizons = (8.0, 14.0, 20.0)
    noises = (3.0e-4, 6.0e-4, 1.2e-3)
    records = []
    for family, family_levels in levels.items():
        for level, strength in family_levels.items():
            for horizon in horizons:
                for noise_std in noises:
                    for repeat in range(REPEATS):
                        row = evaluate(
                            family, level, strength, horizon, noise_std, repeat
                        )
                        records.append(row)
                        print(
                            f"family={family:14s} level={level:5s} H={horizon:4.0f} "
                            f"noise={noise_std:.1e} repeat={repeat} "
                            f"decision={row['decision']:15s} "
                            f"rmse={row['validation_rmse']:.3g} "
                            f"lag1={row['validation_residual_lag1']:.3g}",
                            flush=True,
                        )
    cells = summarize_cells(records)
    aggregate_rows = aggregate(records)
    payload = {
        "experiment": "contract_refusal_noise_horizon_calibration",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "conditional_fit_rank": 1,
        "repeats_per_cell": REPEATS,
        "levels": levels,
        "horizons": horizons,
        "noise_std": noises,
        "records": records,
        "cells": cells,
        "aggregate": aggregate_rows,
        "assessment": assess(aggregate_rows, cells),
    }
    write_outputs(payload)
    print(json.dumps(payload["assessment"], indent=2), flush=True)


if __name__ == "__main__":
    main()
