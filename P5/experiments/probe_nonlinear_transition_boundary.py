"""Map the transition from effective-rank absorption to numerical refusal."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from probe_controlled_misspecification import (
    LATE_AUDIT_RMSE_LIMIT,
    RATES,
    simulate_channel,
)
from probe_mechanism_vs_trajectory_baselines import (
    CHANNELS,
    DEVICE,
    DTYPE,
    HORIZON,
    NOISE_STD,
    NUM_POINTS,
    RESULTS,
    TRAIN_COUNT,
    TRAIN_END_FRACTION,
    fit_mechanism,
    rmse,
)


STRENGTHS = (0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200)
REPEATS = 6
Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    proportion = successes / trials
    denominator = 1.0 + Z_95**2 / trials
    centre = (proportion + Z_95**2 / (2.0 * trials)) / denominator
    radius = (
        Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + Z_95**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return centre - radius, centre + radius


def make_dataset(strength: float, repeat: int) -> dict:
    seed = 463000 + 10007 * STRENGTHS.index(strength) + 101 * repeat
    rng = np.random.default_rng(seed)
    times_np = np.linspace(0.0, HORIZON, NUM_POINTS)
    channel_scale = np.linspace(0.28, 0.82, CHANNELS)[:, None]
    pole_scale = np.linspace(0.78, 1.22, len(RATES))[None, :]
    weights = channel_scale * pole_scale / len(RATES)
    clean_np = np.column_stack(
        [
            simulate_channel(
                times_np,
                weights[channel],
                "nonlinear_feedback",
                strength,
            )
            for channel in range(CHANNELS)
        ]
    )
    observations_np = clean_np + NOISE_STD * rng.standard_normal(clean_np.shape)

    split = int(round(TRAIN_END_FRACTION * (NUM_POINTS - 1)))
    pool = np.arange(1, split + 1)
    sampled = np.sort(rng.choice(pool, size=TRAIN_COUNT - 1, replace=False))
    train_np = np.concatenate(([0], sampled))
    interpolation_np = np.setdiff1d(np.arange(split + 1), train_np)
    extrapolation_np = np.arange(split + 1, NUM_POINTS)
    return {
        "seed": seed,
        "strength": strength,
        "times": torch.tensor(times_np, dtype=DTYPE, device=DEVICE),
        "clean": torch.tensor(clean_np, dtype=DTYPE, device=DEVICE),
        "observations": torch.tensor(observations_np, dtype=DTYPE, device=DEVICE),
        "train_idx": torch.tensor(train_np, dtype=torch.long, device=DEVICE),
        "interpolation_idx": torch.tensor(
            interpolation_np, dtype=torch.long, device=DEVICE
        ),
        "extrapolation_idx": torch.tensor(
            extrapolation_np, dtype=torch.long, device=DEVICE
        ),
        "true_rank": 2,
    }


def evaluate(strength: float, repeat: int) -> dict:
    started = time.perf_counter()
    data = make_dataset(strength, repeat)
    metadata, prediction = fit_mechanism(data)
    clean = data["clean"].detach().cpu().numpy()
    extrapolation_idx = data["extrapolation_idx"].detach().cpu().numpy()
    extrapolation_rmse = rmse(
        prediction[extrapolation_idx], clean[extrapolation_idx]
    )
    late_audit_refusal = bool(
        metadata["quality_failure"]
        or extrapolation_rmse > LATE_AUDIT_RMSE_LIMIT
    )
    refusal_reasons = []
    if metadata["condition"] > 1.0e8:
        refusal_reasons.append("ill_conditioned_jacobian")
    if metadata["quality_failure"] and not refusal_reasons:
        refusal_reasons.append("training_fit_quality")
    if extrapolation_rmse > LATE_AUDIT_RMSE_LIMIT:
        refusal_reasons.append("late_audit_error")
    return {
        "strength": strength,
        "repeat": repeat,
        "seed": data["seed"],
        "selected_rank": metadata["selected_rank"],
        "condition": metadata["condition"],
        "quality_failure": metadata["quality_failure"],
        "extrapolation_rmse": extrapolation_rmse,
        "late_audit_refusal": late_audit_refusal,
        "refusal_reasons": refusal_reasons,
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for strength in STRENGTHS:
        group = [record for record in records if record["strength"] == strength]
        refusals = sum(record["late_audit_refusal"] for record in group)
        lower, upper = wilson_interval(refusals, len(group))
        rows.append(
            {
                "strength": strength,
                "trials": len(group),
                "refusals": refusals,
                "refusal_rate": refusals / len(group),
                "refusal_rate_wilson_95": [lower, upper],
                "refusal_reason_counts": {
                    reason: sum(reason in record["refusal_reasons"] for record in group)
                    for reason in (
                        "training_fit_quality",
                        "ill_conditioned_jacobian",
                        "late_audit_error",
                    )
                },
                "rank3_rate": sum(record["selected_rank"] == 3 for record in group)
                / len(group),
                "median_condition": float(
                    np.median([record["condition"] for record in group])
                ),
                "median_extrapolation_rmse": float(
                    np.median([record["extrapolation_rmse"] for record in group])
                ),
                "max_extrapolation_rmse": float(
                    np.max([record["extrapolation_rmse"] for record in group])
                ),
            }
        )
    return rows


def assess(records: list[dict], summary: list[dict]) -> dict:
    silent_failures = [
        record
        for record in records
        if record["extrapolation_rmse"] > LATE_AUDIT_RMSE_LIMIT
        and not record["late_audit_refusal"]
    ]
    low = summary[0]
    high = summary[-1]
    rates = [row["refusal_rate"] for row in summary]
    checks = {
        "lowest_strength_has_at_least_one_accepted_fit": low["refusals"] < low["trials"],
        "highest_strength_has_at_least_one_refusal": high["refusals"] > 0,
        "no_silent_late_audit_failures": not silent_failures,
        "observed_refusal_rate_is_nondecreasing": all(
            later >= earlier for earlier, later in zip(rates, rates[1:])
        ),
    }
    first_majority_refusal = next(
        (row["strength"] for row in summary if row["refusal_rate"] >= 0.5), None
    )
    last_majority_acceptance = next(
        (
            row["strength"]
            for row in reversed(summary)
            if row["refusal_rate"] <= 0.5
        ),
        None,
    )
    return {
        "checks": checks,
        "route_pass": all(checks.values()),
        "silent_failure_count": len(silent_failures),
        "first_strength_with_at_least_half_refused": first_majority_refusal,
        "last_strength_with_at_least_half_accepted": last_majority_acceptance,
        "boundary_is_descriptive_not_calibrated": True,
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "nonlinear_transition_boundary.json"
    md_path = RESULTS / "nonlinear_transition_boundary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Nonlinear misspecification transition boundary",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        f"- Repeats per strength: {REPEATS}",
        "- Wilson intervals are descriptive with six repeats per cell.",
        "- Refusal uses the frozen numerical-quality gate plus the retrospective late audit.",
        "",
        "| Strength | Refusals | Rate (95% Wilson) | Refusal causes: fit / condition / late | Rank-3 rate | Median condition | Median extrap. RMSE | Max extrap. RMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lower, upper = row["refusal_rate_wilson_95"]
        lines.append(
            f"| {row['strength']:.3f} | {row['refusals']}/{row['trials']} | "
            f"{row['refusal_rate']:.3f} [{lower:.3f}, {upper:.3f}] | "
            f"{row['refusal_reason_counts']['training_fit_quality']} / "
            f"{row['refusal_reason_counts']['ill_conditioned_jacobian']} / "
            f"{row['refusal_reason_counts']['late_audit_error']} | "
            f"{row['rank3_rate']:.3f} | {row['median_condition']:.3e} | "
            f"{row['median_extrapolation_rmse']:.3e} | "
            f"{row['max_extrapolation_rmse']:.3e} |"
        )
    lines.extend(["", "## Prespecified checks", ""])
    lines.extend(
        f"- {name}: **{value}**"
        for name, value in payload["assessment"]["checks"].items()
    )
    lines.extend(
        [
            "",
            "The reported boundary is an observed transition under this fixed design,",
            "not a population-calibrated refusal probability or an a-priori guarantee.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    records = []
    for strength in STRENGTHS:
        for repeat in range(REPEATS):
            record = evaluate(strength, repeat)
            records.append(record)
            print(
                f"strength={strength:.3f} repeat={repeat} "
                f"rank={record['selected_rank']} "
                f"condition={record['condition']:.3e} "
                f"extra={record['extrapolation_rmse']:.3e} "
                f"refuse={record['late_audit_refusal']} "
                f"elapsed={record['elapsed_seconds']:.1f}s",
                flush=True,
            )
    summary = summarize(records)
    payload = {
        "experiment": "nonlinear_transition_boundary",
        "design": {
            "strengths": STRENGTHS,
            "repeats_per_strength": REPEATS,
            "base_rates": RATES.tolist(),
            "late_audit_rmse_limit": LATE_AUDIT_RMSE_LIMIT,
            "training_support": (
                f"{TRAIN_COUNT} points in first {TRAIN_END_FRACTION:.0%} of horizon"
            ),
            "frozen_gate": "condition > 1e8 or train RMSE > max(4*noise, 3e-3)",
        },
        "records": records,
        "summary": summary,
        "assessment": assess(records, summary),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_outputs(payload)
    print(json.dumps(payload["assessment"], indent=2), flush=True)
    print(f"elapsed_seconds={payload['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
