"""Audit an estimated noise-normalized gate under heteroscedastic noise."""

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
from probe_noise_normalized_gate_transfer import gate_diagnostics
from probe_nonlinear_transition_boundary import wilson_interval


BASE_NOISE_LEVELS = (4.0e-4, 8.0e-4, 1.6e-3)
STRENGTHS = (0.050, 0.085, 0.200)
REPEATS = 3
NORMALIZED_TRAIN_LIMIT = 4.0
FIXED_TRAIN_LIMIT = 3.2e-3
CONDITION_LIMIT = 1.0e8
NORMALIZED_EXTRAPOLATION_LIMIT = 10.0
MAX_NOISE_SCALE_RELATIVE_ERROR = 0.20
MIN_ORACLE_DECISION_AGREEMENT = 0.90


def heteroscedastic_profile(times: np.ndarray) -> np.ndarray:
    """Return an increasing profile normalized to unit full-grid RMS."""
    profile = 0.45 + 1.20 * times / HORIZON
    return profile / np.sqrt(np.mean(profile**2))


def make_dataset(base_noise: float, strength: float, repeat: int) -> dict:
    seed = 592000 + 10007 * STRENGTHS.index(strength) + 101 * repeat
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
    point_noise = base_noise * heteroscedastic_profile(times_np)
    replicate_1 = clean_np + point_noise[:, None] * rng.standard_normal(clean_np.shape)
    replicate_2 = clean_np + point_noise[:, None] * rng.standard_normal(clean_np.shape)
    observations_np = 0.5 * (replicate_1 + replicate_2)

    split = int(round(TRAIN_END_FRACTION * (NUM_POINTS - 1)))
    pool = np.arange(1, split + 1)
    sampled = np.sort(rng.choice(pool, size=TRAIN_COUNT - 1, replace=False))
    train_np = np.concatenate(([0], sampled))
    interpolation_np = np.setdiff1d(np.arange(split + 1), train_np)
    extrapolation_np = np.arange(split + 1, NUM_POINTS)

    replicate_difference = replicate_1[train_np] - replicate_2[train_np]
    estimated_effective_train_noise = float(
        np.sqrt(np.mean(replicate_difference**2) / 4.0)
    )
    oracle_effective_train_noise = float(
        np.sqrt(np.mean(point_noise[train_np] ** 2) / 2.0)
    )
    oracle_effective_extrapolation_noise = float(
        np.sqrt(np.mean(point_noise[extrapolation_np] ** 2) / 2.0)
    )
    return {
        "seed": seed,
        "base_noise": base_noise,
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
        "estimated_effective_train_noise": estimated_effective_train_noise,
        "oracle_effective_train_noise": oracle_effective_train_noise,
        "oracle_effective_extrapolation_noise": oracle_effective_extrapolation_noise,
    }


def evaluate(base_noise: float, strength: float, repeat: int) -> dict:
    started = time.perf_counter()
    data = make_dataset(base_noise, strength, repeat)
    metadata, prediction = fit_mechanism(data)
    winner = selected_candidate(metadata)
    clean = data["clean"].detach().cpu().numpy()
    extrapolation_idx = data["extrapolation_idx"].detach().cpu().numpy()
    estimate = data["estimated_effective_train_noise"]
    oracle_train = data["oracle_effective_train_noise"]
    oracle_extrapolation = data["oracle_effective_extrapolation_noise"]
    extrapolation_rmse = rmse(
        prediction[extrapolation_idx], clean[extrapolation_idx]
    )
    oracle_train_ratio = winner["train_rmse"] / oracle_train
    estimated_train_ratio = winner["train_rmse"] / estimate
    extrapolation_ratio = extrapolation_rmse / oracle_extrapolation
    ill_conditioned = metadata["condition"] > CONDITION_LIMIT
    fixed_refusal = bool(ill_conditioned or winner["train_rmse"] > FIXED_TRAIN_LIMIT)
    oracle_refusal = bool(
        ill_conditioned or oracle_train_ratio > NORMALIZED_TRAIN_LIMIT
    )
    estimated_refusal = bool(
        ill_conditioned or estimated_train_ratio > NORMALIZED_TRAIN_LIMIT
    )
    return {
        "base_noise": base_noise,
        "strength": strength,
        "repeat": repeat,
        "seed": data["seed"],
        "selected_rank": metadata["selected_rank"],
        "condition": metadata["condition"],
        "train_rmse": winner["train_rmse"],
        "estimated_effective_train_noise": estimate,
        "oracle_effective_train_noise": oracle_train,
        "noise_scale_relative_error": abs(estimate - oracle_train) / oracle_train,
        "oracle_train_rmse_over_noise": oracle_train_ratio,
        "estimated_train_rmse_over_noise": estimated_train_ratio,
        "extrapolation_rmse": extrapolation_rmse,
        "oracle_effective_extrapolation_noise": oracle_extrapolation,
        "extrapolation_rmse_over_oracle_noise": extrapolation_ratio,
        "fixed_absolute_refusal": fixed_refusal,
        "oracle_normalized_refusal": oracle_refusal,
        "estimated_normalized_refusal": estimated_refusal,
        "normalized_relative_extrapolation_failure": (
            extrapolation_ratio > NORMALIZED_EXTRAPOLATION_LIMIT
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for base_noise in BASE_NOISE_LEVELS:
        for strength in STRENGTHS:
            group = [
                record
                for record in records
                if record["base_noise"] == base_noise
                and record["strength"] == strength
            ]
            refused = sum(record["estimated_normalized_refusal"] for record in group)
            lower, upper = wilson_interval(refused, len(group))
            rows.append(
                {
                    "base_noise": base_noise,
                    "strength": strength,
                    "trials": len(group),
                    "estimated_gate_refusals": refused,
                    "estimated_gate_refusal_wilson_95": [lower, upper],
                    "median_noise_scale_relative_error": float(
                        np.median([r["noise_scale_relative_error"] for r in group])
                    ),
                    "median_estimated_train_ratio": float(
                        np.median([r["estimated_train_rmse_over_noise"] for r in group])
                    ),
                    "median_extrapolation_ratio": float(
                        np.median(
                            [r["extrapolation_rmse_over_oracle_noise"] for r in group]
                        )
                    ),
                }
            )
    return rows


def diagnostics(records: list[dict], decision_key: str) -> dict:
    remapped = []
    for record in records:
        item = dict(record)
        item["extrapolation_rmse_over_noise"] = record[
            "extrapolation_rmse_over_oracle_noise"
        ]
        remapped.append(item)
    return gate_diagnostics(remapped, decision_key)


def assess(records: list[dict], summary: list[dict]) -> dict:
    estimated = diagnostics(records, "estimated_normalized_refusal")
    oracle = diagnostics(records, "oracle_normalized_refusal")
    fixed = diagnostics(records, "fixed_absolute_refusal")
    agreements = sum(
        record["estimated_normalized_refusal"]
        == record["oracle_normalized_refusal"]
        for record in records
    )
    agreement_rate = agreements / len(records)
    max_noise_error = max(record["noise_scale_relative_error"] for record in records)
    checks = {
        "replicate_noise_estimator_within_relative_error_limit": (
            max_noise_error <= MAX_NOISE_SCALE_RELATIVE_ERROR
        ),
        "estimated_gate_agrees_with_oracle_at_prespecified_rate": (
            agreement_rate >= MIN_ORACLE_DECISION_AGREEMENT
        ),
        "estimated_gate_has_no_silent_relative_extrapolation_failures": (
            estimated["silent_relative_extrapolation_failures"] == 0
        ),
        "estimated_gate_controls_accepted_relative_extrapolation_error": (
            estimated["max_accepted_extrapolation_rmse_over_noise"] is not None
            and estimated["max_accepted_extrapolation_rmse_over_noise"]
            <= NORMALIZED_EXTRAPOLATION_LIMIT
        ),
        "all_cells_have_expected_repeat_count": all(
            row["trials"] == REPEATS for row in summary
        ),
    }
    return {
        "checks": checks,
        "route_pass": all(checks.values()),
        "estimated_oracle_decision_agreement": agreements,
        "estimated_oracle_decision_agreement_rate": agreement_rate,
        "maximum_noise_scale_relative_error": max_noise_error,
        "estimated_normalized_gate": estimated,
        "oracle_normalized_gate": oracle,
        "fixed_absolute_gate": fixed,
        "scope": (
            "Repeated measurements identify the aggregate training noise scale in "
            "this controlled Gaussian experiment. This is not a guarantee for "
            "single-series, correlated, or non-Gaussian observations."
        ),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "replicate_noise_gate_transfer.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Replicate-estimated gate under heteroscedastic noise",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        "- Two independent replicates are averaged for fitting.",
        "- The effective training noise is estimated from their paired difference.",
        "- Noise amplitude increases with time and is normalized to the declared base RMS.",
        "",
        "| Base noise | Strength | Estimated refusals (95% Wilson) | Median noise-scale error | Median train/estimated noise | Median extrap./oracle noise |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lower, upper = row["estimated_gate_refusal_wilson_95"]
        lines.append(
            f"| {row['base_noise']:.1e} | {row['strength']:.3f} | "
            f"{row['estimated_gate_refusals']}/{row['trials']} "
            f"[{lower:.3f}, {upper:.3f}] | "
            f"{row['median_noise_scale_relative_error']:.3f} | "
            f"{row['median_estimated_train_ratio']:.3f} | "
            f"{row['median_extrapolation_ratio']:.3f} |"
        )
    lines.extend(["", "## Gate-level diagnostics", ""])
    for name in (
        "estimated_normalized_gate",
        "oracle_normalized_gate",
        "fixed_absolute_gate",
    ):
        lines.append(f"- {name}: {payload['assessment'][name]}")
    lines.extend(["", "## Prespecified checks", ""])
    lines.extend(
        f"- {name}: **{value}**"
        for name, value in payload["assessment"]["checks"].items()
    )
    lines.extend(["", payload["assessment"]["scope"]])
    (RESULTS / "replicate_noise_gate_transfer.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    records = []
    for base_noise in BASE_NOISE_LEVELS:
        for strength in STRENGTHS:
            for repeat in range(REPEATS):
                record = evaluate(base_noise, strength, repeat)
                records.append(record)
                print(
                    f"noise={base_noise:.1e} strength={strength:.3f} repeat={repeat} "
                    f"rank={record['selected_rank']} "
                    f"noise-error={record['noise_scale_relative_error']:.3f} "
                    f"train/estimated={record['estimated_train_rmse_over_noise']:.3f} "
                    f"extra/oracle={record['extrapolation_rmse_over_oracle_noise']:.3f} "
                    f"oracle={record['oracle_normalized_refusal']} "
                    f"estimated={record['estimated_normalized_refusal']} "
                    f"elapsed={record['elapsed_seconds']:.1f}s",
                    flush=True,
                )
    summary = summarize(records)
    payload = {
        "experiment": "replicate_noise_gate_transfer",
        "design": {
            "base_noise_levels": BASE_NOISE_LEVELS,
            "strengths": STRENGTHS,
            "repeats_per_cell": REPEATS,
            "measurement_replicates": 2,
            "heteroscedastic_profile": "increasing linear profile with unit full-grid RMS",
            "normalized_train_limit": NORMALIZED_TRAIN_LIMIT,
            "fixed_train_limit": FIXED_TRAIN_LIMIT,
            "normalized_extrapolation_limit": NORMALIZED_EXTRAPOLATION_LIMIT,
            "max_noise_scale_relative_error": MAX_NOISE_SCALE_RELATIVE_ERROR,
            "minimum_oracle_decision_agreement": MIN_ORACLE_DECISION_AGREEMENT,
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
