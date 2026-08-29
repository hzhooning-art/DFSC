"""Disentangle horizon and noise effects in joint rank/refusal decisions."""

from __future__ import annotations

import json

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank, lifted_response
from probe_out_of_class_refusal import mean_lag1, oscillatory_response, prediction_from_fit
from probe_refusal_calibration import RESULTS, wilson_interval


REPEATS = 6
REFERENCE_HORIZON = 14.0
REFERENCE_NOISE = 6.0e-4
REFERENCE_DT = REFERENCE_HORIZON / 96.0
HORIZONS = (10.0, 14.0, 18.0)
NOISE_LEVELS = (4.0e-4, 6.0e-4, 9.0e-4)
LEVELS = {
    "signed_residue": {"zero": 0.0, "above": 0.0075},
    "oscillation": {"zero": 0.0, "above": 0.04},
}


def operating_points() -> list[dict]:
    points = [
        {
            "axis": "horizon",
            "name": f"horizon_{horizon:g}",
            "horizon": horizon,
            "noise_std": REFERENCE_NOISE,
        }
        for horizon in HORIZONS
    ]
    points.extend({
        "axis": "noise",
        "name": f"noise_{noise_std:.1e}",
        "horizon": REFERENCE_HORIZON,
        "noise_std": noise_std,
    } for noise_std in NOISE_LEVELS if noise_std != REFERENCE_NOISE)
    return points


def clean_response(
    family: str,
    strength: float,
    times: torch.Tensor,
) -> torch.Tensor:
    channels = 6
    scale = torch.linspace(0.72, 1.28, channels, dtype=DTYPE, device=DEVICE)
    if family == "signed_residue":
        weights = torch.stack([0.46 * scale, -strength / scale], dim=1)
        return lifted_response(
            times,
            weights,
            torch.tensor([0.22, 1.35], dtype=DTYPE, device=DEVICE),
        )
    if family == "oscillation":
        return oscillatory_response(
            times, 0.38 * scale, decay=0.24, frequency=strength
        )
    raise ValueError(f"Unknown family: {family}")


def generate_fixed_cadence(
    family: str,
    strength: float,
    horizon: float,
    noise_std: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    num_points = int(round(horizon / REFERENCE_DT)) + 1
    times = torch.linspace(0.0, horizon, num_points, dtype=DTYPE, device=DEVICE)
    clean = clean_response(family, strength, times)

    observations = clean + noise_std * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )
    split = int(round(0.72 * num_points))
    train_pool = np.arange(1, split)
    train_size = min(int(round(0.50 * num_points)), len(train_pool))
    train_np = np.sort(rng.choice(train_pool, size=train_size, replace=False))
    val_np = np.arange(split, num_points)
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
    point: dict,
    repeat: int,
) -> dict:
    seed = (
        31000
        + repeat
        + int(round(strength * 100000))
        + int(round(point["horizon"] * 10))
        + int(round(point["noise_std"] * 1.0e7))
        + sum(map(ord, family))
    )
    times, observations, train_idx, val_idx = generate_fixed_cadence(
        family,
        strength,
        point["horizon"],
        point["noise_std"],
        seed,
    )
    fits = []
    for rank in (1, 2, 3):
        candidates = [
            fit_rank(
                times,
                observations,
                train_idx,
                val_idx,
                rank=rank,
                seed=seed * 100 + start,
                adam_steps=210,
                lbfgs_steps=60,
            )
            for start in range(2)
        ]
        fits.append(min(candidates, key=lambda item: item.bic))

    winner = min(fits, key=lambda item: item.bic)
    residual = prediction_from_fit(times, winner)[val_idx] - observations[val_idx]
    lag1 = mean_lag1(residual)
    rank2 = next(fit for fit in fits if fit.rank == 2)
    rank3 = next(fit for fit in fits if fit.rank == 3)
    cap_gain = rank2.bic - rank3.bic
    checks = {
        "prediction": winner.val_rmse <= max(4.0 * point["noise_std"], 3.0e-3),
        "condition": winner.jacobian_condition <= 1.0e8,
        "residual": abs(lag1) <= 0.55,
    }
    rank_cap = winner.rank == 3 and cap_gain >= 6.0
    accepted = all(checks.values()) and not rank_cap
    return {
        "axis": point["axis"],
        "point": point["name"],
        "family": family,
        "level": level,
        "strength": strength,
        "repeat": repeat,
        "horizon": point["horizon"],
        "noise_std": point["noise_std"],
        "num_time_points": int(times.numel()),
        "decision": "ACCEPT_CONTRACT" if accepted else "REFUSE_CONTRACT",
        "selected_rank": winner.rank,
        "validation_rmse": winner.val_rmse,
        "jacobian_condition": winner.jacobian_condition,
        "validation_residual_lag1": lag1,
        "rank3_vs_rank2_bic_gain": cap_gain,
        "failed_gates": [name for name, passed in checks.items() if not passed]
        + (["rank_cap"] if rank_cap else []),
        "candidate_bic": {str(fit.rank): fit.bic for fit in fits},
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    keys = sorted({(row["family"], row["level"], row["point"]) for row in records})
    for family, level, point in keys:
        group = [
            row for row in records
            if (row["family"], row["level"], row["point"])
            == (family, level, point)
        ]
        refused = sum(row["decision"] == "REFUSE_CONTRACT" for row in group)
        elevated = sum(row["selected_rank"] > 1 for row in group)
        absorbed = sum(
            row["decision"] == "ACCEPT_CONTRACT" and row["selected_rank"] > 1
            for row in group
        )
        times = torch.linspace(
            0.0,
            group[0]["horizon"],
            group[0]["num_time_points"],
            dtype=DTYPE,
            device=DEVICE,
        )
        split = int(round(0.72 * group[0]["num_time_points"]))
        mismatch = (
            clean_response(family, group[0]["strength"], times)
            - clean_response(family, 0.0, times)
        )[split:]
        mismatch_snr = float(
            torch.sqrt(torch.mean(mismatch * mismatch)).item()
            / group[0]["noise_std"]
        )
        rows.append({
            "family": family,
            "level": level,
            "axis": group[0]["axis"],
            "point": point,
            "horizon": group[0]["horizon"],
            "noise_std": group[0]["noise_std"],
            "num_time_points": group[0]["num_time_points"],
            "trials": len(group),
            "refusal_fraction": refused / len(group),
            "refusal_wilson95": wilson_interval(refused, len(group)),
            "elevated_rank_fraction": elevated / len(group),
            "absorbed_violation_fraction": absorbed / len(group),
            "validation_mismatch_rms_to_noise": mismatch_snr,
            "median_validation_rmse": float(np.median([
                row["validation_rmse"] for row in group
            ])),
            "median_abs_residual_lag1": float(np.median([
                abs(row["validation_residual_lag1"]) for row in group
            ])),
        })
    return rows


def assess(summary: list[dict]) -> dict:
    lookup = {
        (row["family"], row["level"], row["point"]): row
        for row in summary
    }
    results = {}
    horizon_points = [f"horizon_{value:g}" for value in HORIZONS]
    noise_points = [f"noise_{value:.1e}" for value in NOISE_LEVELS]
    noise_points[1] = "horizon_14"
    for family in LEVELS:
        all_rows = [row for row in summary if row["family"] == family]
        zero_rows = [row for row in all_rows if row["level"] == "zero"]
        above_rows = [row for row in all_rows if row["level"] == "above"]
        horizon_rates = [
            lookup[(family, "above", point)]["refusal_fraction"]
            for point in horizon_points
        ]
        noise_rates = [
            lookup[(family, "above", point)]["refusal_fraction"]
            for point in noise_points
        ]
        results[family] = {
            "zero_false_refusal_at_most_0.20": all(
                row["refusal_fraction"] <= 0.20 for row in zero_rows
            ),
            "zero_false_elevated_rank_at_most_0.20": all(
                row["elevated_rank_fraction"] <= 0.20 for row in zero_rows
            ),
            "horizon_effect_nondecreasing": all(
                left <= right for left, right in zip(horizon_rates, horizon_rates[1:])
            ),
            "noise_effect_nonincreasing": all(
                left >= right for left, right in zip(noise_rates, noise_rates[1:])
            ),
            "accepted_elevated_rank_absorption_at_most_0.20": all(
                row["absorbed_violation_fraction"] <= 0.20 for row in above_rows
            ),
            "rates": {
                "horizon_sweep": dict(zip(HORIZONS, horizon_rates)),
                "noise_sweep": dict(zip(NOISE_LEVELS, noise_rates)),
            },
        }
    return {
        "families": results,
        "route_pass": all(
            all(value for key, value in result.items() if key != "rates")
            for result in results.values()
        ),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "joint_horizon_noise_effects.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Joint horizon/noise effect isolation",
        "",
        "Horizon sweeps preserve the reference sampling interval; noise sweeps use H=14.",
        "",
        "| Family | Level | Point | H | Noise | N(t) | Mismatch/noise | Refusal (Wilson 95%) | Elevated |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lo, hi = row["refusal_wilson95"]
        lines.append(
            f"| {row['family']} | {row['level']} | {row['point']} | "
            f"{row['horizon']:.0f} | {row['noise_std']:.1e} | "
            f"{row['num_time_points']} | {row['validation_mismatch_rms_to_noise']:.2f} | "
            f"{row['refusal_fraction']:.2f} "
            f"[{lo:.2f}, {hi:.2f}] | {row['elevated_rank_fraction']:.2f} |"
        )
    lines.extend([
        "",
        f"Route pass: **{payload['assessment']['route_pass']}**.",
        "",
        "These local one-factor sweeps isolate directional effects around the reference",
        "operating point; they do not establish a global monotonicity theorem.",
    ])
    (RESULTS / "joint_horizon_noise_effects.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    records = []
    for point in operating_points():
        for family, levels in LEVELS.items():
            for level, strength in levels.items():
                for repeat in range(REPEATS):
                    row = evaluate(family, level, strength, point, repeat)
                    records.append(row)
                    print(
                        f"point={point['name']:16s} family={family:14s} "
                        f"level={level:5s} repeat={repeat} "
                        f"decision={row['decision']:15s} rank={row['selected_rank']} "
                        f"lag1={row['validation_residual_lag1']:.3g}",
                        flush=True,
                    )
    summary = summarize(records)
    payload = {
        "experiment": "joint_horizon_noise_effect_isolation",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "candidate_ranks": [1, 2, 3],
        "starts_per_rank": 2,
        "repeats_per_setting": REPEATS,
        "reference_sampling_interval": REFERENCE_DT,
        "operating_points": operating_points(),
        "generator_labels_used_only_for_scoring": LEVELS,
        "records": records,
        "summary": summary,
        "assessment": assess(summary),
    }
    write_outputs(payload)
    print(json.dumps(payload["assessment"], indent=2), flush=True)


if __name__ == "__main__":
    main()
