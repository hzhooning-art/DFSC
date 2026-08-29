"""Test transfer of an RMSE/noise gate across paired noise levels."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from probe_controlled_misspecification import RATES, simulate_channel
from probe_mechanism_vs_trajectory_baselines import (
    CHANNELS,
    DEVICE,
    DTYPE,
    HORIZON,
    NUM_POINTS,
    RESULTS,
    TRAIN_COUNT,
    TRAIN_END_FRACTION,
    fit_mechanism,
    rmse,
)
from probe_nonlinear_boundary_refinement import selected_candidate
from probe_nonlinear_transition_boundary import wilson_interval


NOISE_LEVELS = (4.0e-4, 8.0e-4, 1.6e-3)
STRENGTHS = (0.075, 0.085, 0.125, 0.200)
REPEATS = 4
NORMALIZED_TRAIN_LIMIT = 4.0
FIXED_TRAIN_LIMIT = 3.2e-3
CONDITION_LIMIT = 1.0e8
NORMALIZED_EXTRAPOLATION_LIMIT = 10.0


def make_dataset(noise_std: float, strength: float, repeat: int) -> dict:
    seed = 491000 + 10007 * STRENGTHS.index(strength) + 101 * repeat
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
    standard_noise = rng.standard_normal(clean_np.shape)
    observations_np = clean_np + noise_std * standard_noise
    split = int(round(TRAIN_END_FRACTION * (NUM_POINTS - 1)))
    pool = np.arange(1, split + 1)
    sampled = np.sort(rng.choice(pool, size=TRAIN_COUNT - 1, replace=False))
    train_np = np.concatenate(([0], sampled))
    interpolation_np = np.setdiff1d(np.arange(split + 1), train_np)
    extrapolation_np = np.arange(split + 1, NUM_POINTS)
    return {
        "seed": seed,
        "noise_std": noise_std,
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


def evaluate(noise_std: float, strength: float, repeat: int) -> dict:
    started = time.perf_counter()
    data = make_dataset(noise_std, strength, repeat)
    metadata, prediction = fit_mechanism(data)
    winner = selected_candidate(metadata)
    clean = data["clean"].detach().cpu().numpy()
    extrapolation_idx = data["extrapolation_idx"].detach().cpu().numpy()
    train_ratio = winner["train_rmse"] / noise_std
    extrapolation_rmse = rmse(
        prediction[extrapolation_idx], clean[extrapolation_idx]
    )
    extrapolation_ratio = extrapolation_rmse / noise_std
    ill_conditioned = metadata["condition"] > CONDITION_LIMIT
    fixed_refusal = bool(ill_conditioned or winner["train_rmse"] > FIXED_TRAIN_LIMIT)
    normalized_refusal = bool(
        ill_conditioned or train_ratio > NORMALIZED_TRAIN_LIMIT
    )
    return {
        "noise_std": noise_std,
        "strength": strength,
        "repeat": repeat,
        "seed": data["seed"],
        "selected_rank": metadata["selected_rank"],
        "condition": metadata["condition"],
        "train_rmse": winner["train_rmse"],
        "train_rmse_over_noise": train_ratio,
        "extrapolation_rmse": extrapolation_rmse,
        "extrapolation_rmse_over_noise": extrapolation_ratio,
        "fixed_absolute_refusal": fixed_refusal,
        "noise_normalized_refusal": normalized_refusal,
        "normalized_relative_extrapolation_failure": (
            extrapolation_ratio > NORMALIZED_EXTRAPOLATION_LIMIT
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for noise_std in NOISE_LEVELS:
        for strength in STRENGTHS:
            group = [
                record
                for record in records
                if record["noise_std"] == noise_std
                and record["strength"] == strength
            ]
            fixed = sum(record["fixed_absolute_refusal"] for record in group)
            normalized = sum(record["noise_normalized_refusal"] for record in group)
            norm_lower, norm_upper = wilson_interval(normalized, len(group))
            rows.append(
                {
                    "noise_std": noise_std,
                    "strength": strength,
                    "trials": len(group),
                    "fixed_absolute_refusals": fixed,
                    "noise_normalized_refusals": normalized,
                    "noise_normalized_refusal_rate": normalized / len(group),
                    "noise_normalized_refusal_wilson_95": [norm_lower, norm_upper],
                    "median_train_rmse_over_noise": float(
                        np.median([r["train_rmse_over_noise"] for r in group])
                    ),
                    "median_extrapolation_rmse_over_noise": float(
                        np.median([r["extrapolation_rmse_over_noise"] for r in group])
                    ),
                    "accepted_normalized_extrapolation_ratios": [
                        r["extrapolation_rmse_over_noise"]
                        for r in group
                        if not r["noise_normalized_refusal"]
                    ],
                }
            )
    return rows


def gate_diagnostics(records: list[dict], decision_key: str) -> dict:
    accepted = [record for record in records if not record[decision_key]]
    refused = [record for record in records if record[decision_key]]
    silent = [
        record
        for record in accepted
        if record["normalized_relative_extrapolation_failure"]
    ]
    conservative = [
        record
        for record in refused
        if not record["normalized_relative_extrapolation_failure"]
    ]
    accepted_ratios = [record["extrapolation_rmse_over_noise"] for record in accepted]
    return {
        "accepted": len(accepted),
        "refused": len(refused),
        "silent_relative_extrapolation_failures": len(silent),
        "refusals_without_relative_extrapolation_failure": len(conservative),
        "max_accepted_extrapolation_rmse_over_noise": (
            float(max(accepted_ratios)) if accepted_ratios else None
        ),
        "median_accepted_extrapolation_rmse_over_noise": (
            float(np.median(accepted_ratios)) if accepted_ratios else None
        ),
    }


def assess(records: list[dict], summary: list[dict]) -> dict:
    normalized = gate_diagnostics(records, "noise_normalized_refusal")
    fixed = gate_diagnostics(records, "fixed_absolute_refusal")
    decision_disagreements = sum(
        record["noise_normalized_refusal"] != record["fixed_absolute_refusal"]
        for record in records
    )
    checks = {
        "paired_gate_decisions_differ_on_at_least_one_case": decision_disagreements > 0,
        "normalized_gate_has_no_silent_relative_extrapolation_failures": (
            normalized["silent_relative_extrapolation_failures"] == 0
        ),
        "normalized_gate_controls_accepted_relative_extrapolation_error": (
            normalized["max_accepted_extrapolation_rmse_over_noise"] is not None
            and normalized["max_accepted_extrapolation_rmse_over_noise"]
            <= NORMALIZED_EXTRAPOLATION_LIMIT
        ),
        "all_cells_have_expected_repeat_count": all(
            row["trials"] == REPEATS for row in summary
        ),
    }
    return {
        "checks": checks,
        "route_pass": all(checks.values()),
        "decision_disagreements": decision_disagreements,
        "noise_normalized_gate": normalized,
        "fixed_absolute_gate": fixed,
        "scope": (
            "The relative extrapolation limit is an operational audit target, not a "
            "theoretical error bound or a universal utility threshold."
        ),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "noise_normalized_gate_transfer.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Noise-normalized gate transfer",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        f"- Normalized training gate: RMSE/noise <= {NORMALIZED_TRAIN_LIMIT:.1f}",
        f"- Relative extrapolation audit target: RMSE/noise <= {NORMALIZED_EXTRAPOLATION_LIMIT:.1f}",
        "- Noise levels are paired through the same standard-normal draw and sampling indices.",
        "",
        "| Noise | Strength | Fixed refusals | Normalized refusals (95% Wilson) | Median train/noise | Median extrap./noise |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lower, upper = row["noise_normalized_refusal_wilson_95"]
        lines.append(
            f"| {row['noise_std']:.1e} | {row['strength']:.3f} | "
            f"{row['fixed_absolute_refusals']}/{row['trials']} | "
            f"{row['noise_normalized_refusals']}/{row['trials']} "
            f"[{lower:.3f}, {upper:.3f}] | "
            f"{row['median_train_rmse_over_noise']:.3f} | "
            f"{row['median_extrapolation_rmse_over_noise']:.3f} |"
        )
    lines.extend(["", "## Gate-level diagnostics", ""])
    for name in ("noise_normalized_gate", "fixed_absolute_gate"):
        lines.append(f"- {name}: {payload['assessment'][name]}")
    lines.extend(["", "## Prespecified checks", ""])
    lines.extend(
        f"- {name}: **{value}**"
        for name, value in payload["assessment"]["checks"].items()
    )
    lines.extend(["", payload["assessment"]["scope"]])
    (RESULTS / "noise_normalized_gate_transfer.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    records = []
    for noise_std in NOISE_LEVELS:
        for strength in STRENGTHS:
            for repeat in range(REPEATS):
                record = evaluate(noise_std, strength, repeat)
                records.append(record)
                print(
                    f"noise={noise_std:.1e} strength={strength:.3f} repeat={repeat} "
                    f"rank={record['selected_rank']} "
                    f"train/noise={record['train_rmse_over_noise']:.3f} "
                    f"extra/noise={record['extrapolation_rmse_over_noise']:.3f} "
                    f"fixed={record['fixed_absolute_refusal']} "
                    f"normalized={record['noise_normalized_refusal']} "
                    f"elapsed={record['elapsed_seconds']:.1f}s",
                    flush=True,
                )
    summary = summarize(records)
    payload = {
        "experiment": "noise_normalized_gate_transfer",
        "design": {
            "noise_levels": NOISE_LEVELS,
            "strengths": STRENGTHS,
            "repeats_per_cell": REPEATS,
            "paired_noise_and_sampling": True,
            "normalized_train_limit": NORMALIZED_TRAIN_LIMIT,
            "fixed_train_limit": FIXED_TRAIN_LIMIT,
            "normalized_extrapolation_limit": NORMALIZED_EXTRAPOLATION_LIMIT,
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
