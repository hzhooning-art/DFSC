"""Test a single-series robust noise gate under sparse gross contamination."""

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
from probe_replicate_noise_gate_transfer import diagnostics
from probe_nonlinear_transition_boundary import wilson_interval


BASE_NOISE_LEVELS = (4.0e-4, 8.0e-4, 1.6e-3)
STRENGTHS = (0.050, 0.085, 0.200)
NOISE_KINDS = ("gaussian", "contaminated")
REPEATS = 2
OUTLIER_FRACTION = 0.02
OUTLIER_SCALE = 10.0
NORMALIZED_TRAIN_LIMIT = 4.0
FIXED_TRAIN_LIMIT = 3.2e-3
CONDITION_LIMIT = 1.0e8
NORMALIZED_EXTRAPOLATION_LIMIT = 10.0
MAX_MEDIAN_SCALE_ERROR = 0.30
MAX_P90_SCALE_ERROR = 0.50
MIN_ORACLE_DECISION_AGREEMENT = 0.80
NORMAL_MAD = 0.6744897501960817


def multiscale_noise_estimates(prefix: np.ndarray) -> tuple[float, float]:
    """Estimate the iid core scale without fitting the mechanism model."""
    robust_scales = []
    for lag in (1, 2):
        centers = np.arange(lag, prefix.shape[0] - lag)
        second = prefix[centers + lag] - 2.0 * prefix[centers] + prefix[centers - lag]
        centered = second - np.median(second)
        robust_scales.append(np.median(np.abs(centered)) / NORMAL_MAD)
    # Smooth-signal curvature scales as lag^2, hence its squared contribution as lag^4.
    intercept = (16.0 * robust_scales[0] ** 2 - robust_scales[1] ** 2) / 15.0
    robust = np.sqrt(max(intercept / 6.0, np.finfo(float).tiny))
    second_lag_one = prefix[2:] - 2.0 * prefix[1:-1] + prefix[:-2]
    naive = np.std(second_lag_one, ddof=1) / np.sqrt(6.0)
    return float(robust), float(naive)


def make_dataset(
    base_noise: float, strength: float, noise_kind: str, repeat: int
) -> dict:
    seed = (
        693000
        + 10007 * STRENGTHS.index(strength)
        + 1009 * NOISE_KINDS.index(noise_kind)
        + 101 * repeat
    )
    rng = np.random.default_rng(seed)
    times_np = np.linspace(0.0, HORIZON, NUM_POINTS)
    channel_scale = np.linspace(0.28, 0.82, CHANNELS)[:, None]
    pole_scale = np.linspace(0.78, 1.22, len(RATES))[None, :]
    weights = channel_scale * pole_scale / len(RATES)
    clean_np = np.column_stack(
        [
            simulate_channel(
                times_np, weights[channel], "nonlinear_feedback", strength
            )
            for channel in range(CHANNELS)
        ]
    )
    standardized_noise = rng.standard_normal(clean_np.shape)
    outlier_count = 0
    if noise_kind == "contaminated":
        mask = rng.random(clean_np.shape) < OUTLIER_FRACTION
        standardized_noise = standardized_noise + mask * rng.normal(
            0.0, OUTLIER_SCALE, size=clean_np.shape
        )
        outlier_count = int(mask.sum())
    observations_np = clean_np + base_noise * standardized_noise

    split = int(round(TRAIN_END_FRACTION * (NUM_POINTS - 1)))
    pool = np.arange(1, split + 1)
    sampled = np.sort(rng.choice(pool, size=TRAIN_COUNT - 1, replace=False))
    train_np = np.concatenate(([0], sampled))
    interpolation_np = np.setdiff1d(np.arange(split + 1), train_np)
    extrapolation_np = np.arange(split + 1, NUM_POINTS)
    robust_noise, naive_noise = multiscale_noise_estimates(
        observations_np[: split + 1]
    )
    return {
        "seed": seed,
        "base_noise": base_noise,
        "strength": strength,
        "noise_kind": noise_kind,
        "outlier_count": outlier_count,
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
        "robust_noise_estimate": robust_noise,
        "naive_noise_estimate": naive_noise,
    }


def evaluate(
    base_noise: float, strength: float, noise_kind: str, repeat: int
) -> dict:
    started = time.perf_counter()
    data = make_dataset(base_noise, strength, noise_kind, repeat)
    metadata, prediction = fit_mechanism(data)
    winner = selected_candidate(metadata)
    clean = data["clean"].detach().cpu().numpy()
    extrapolation_idx = data["extrapolation_idx"].detach().cpu().numpy()
    extrapolation_rmse = rmse(
        prediction[extrapolation_idx], clean[extrapolation_idx]
    )
    robust_noise = data["robust_noise_estimate"]
    naive_noise = data["naive_noise_estimate"]
    oracle_ratio = winner["train_rmse"] / base_noise
    robust_ratio = winner["train_rmse"] / robust_noise
    naive_ratio = winner["train_rmse"] / naive_noise
    extrapolation_ratio = extrapolation_rmse / base_noise
    ill_conditioned = metadata["condition"] > CONDITION_LIMIT
    return {
        "base_noise": base_noise,
        "strength": strength,
        "noise_kind": noise_kind,
        "repeat": repeat,
        "seed": data["seed"],
        "outlier_count": data["outlier_count"],
        "selected_rank": metadata["selected_rank"],
        "condition": metadata["condition"],
        "train_rmse": winner["train_rmse"],
        "robust_noise_estimate": robust_noise,
        "naive_noise_estimate": naive_noise,
        "robust_scale_relative_error": abs(robust_noise - base_noise) / base_noise,
        "naive_scale_relative_error": abs(naive_noise - base_noise) / base_noise,
        "oracle_train_rmse_over_noise": oracle_ratio,
        "robust_train_rmse_over_noise": robust_ratio,
        "naive_train_rmse_over_noise": naive_ratio,
        "extrapolation_rmse": extrapolation_rmse,
        "extrapolation_rmse_over_oracle_noise": extrapolation_ratio,
        "oracle_normalized_refusal": bool(
            ill_conditioned or oracle_ratio > NORMALIZED_TRAIN_LIMIT
        ),
        "robust_normalized_refusal": bool(
            ill_conditioned or robust_ratio > NORMALIZED_TRAIN_LIMIT
        ),
        "naive_normalized_refusal": bool(
            ill_conditioned or naive_ratio > NORMALIZED_TRAIN_LIMIT
        ),
        "fixed_absolute_refusal": bool(
            ill_conditioned or winner["train_rmse"] > FIXED_TRAIN_LIMIT
        ),
        "normalized_relative_extrapolation_failure": (
            extrapolation_ratio > NORMALIZED_EXTRAPOLATION_LIMIT
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for noise_kind in NOISE_KINDS:
        for base_noise in BASE_NOISE_LEVELS:
            for strength in STRENGTHS:
                group = [
                    record
                    for record in records
                    if record["noise_kind"] == noise_kind
                    and record["base_noise"] == base_noise
                    and record["strength"] == strength
                ]
                refused = sum(r["robust_normalized_refusal"] for r in group)
                lower, upper = wilson_interval(refused, len(group))
                rows.append(
                    {
                        "noise_kind": noise_kind,
                        "base_noise": base_noise,
                        "strength": strength,
                        "trials": len(group),
                        "robust_refusals": refused,
                        "robust_refusal_wilson_95": [lower, upper],
                        "median_robust_scale_ratio": float(
                            np.median(
                                [r["robust_noise_estimate"] / base_noise for r in group]
                            )
                        ),
                        "median_naive_scale_ratio": float(
                            np.median(
                                [r["naive_noise_estimate"] / base_noise for r in group]
                            )
                        ),
                        "median_extrapolation_ratio": float(
                            np.median(
                                [r["extrapolation_rmse_over_oracle_noise"] for r in group]
                            )
                        ),
                    }
                )
    return rows


def assess(records: list[dict], summary: list[dict]) -> dict:
    robust = diagnostics(records, "robust_normalized_refusal")
    naive = diagnostics(records, "naive_normalized_refusal")
    oracle = diagnostics(records, "oracle_normalized_refusal")
    fixed = diagnostics(records, "fixed_absolute_refusal")
    agreements = sum(
        r["robust_normalized_refusal"] == r["oracle_normalized_refusal"]
        for r in records
    )
    robust_errors = np.asarray([r["robust_scale_relative_error"] for r in records])
    contaminated = [r for r in records if r["noise_kind"] == "contaminated"]
    contaminated_robust_overestimate = float(
        np.median([r["robust_noise_estimate"] / r["base_noise"] for r in contaminated])
    )
    contaminated_naive_overestimate = float(
        np.median([r["naive_noise_estimate"] / r["base_noise"] for r in contaminated])
    )
    checks = {
        "median_robust_scale_error_within_limit": (
            float(np.median(robust_errors)) <= MAX_MEDIAN_SCALE_ERROR
        ),
        "p90_robust_scale_error_within_limit": (
            float(np.quantile(robust_errors, 0.90)) <= MAX_P90_SCALE_ERROR
        ),
        "robust_gate_agrees_with_oracle_at_prespecified_rate": (
            agreements / len(records) >= MIN_ORACLE_DECISION_AGREEMENT
        ),
        "robust_gate_has_no_silent_relative_extrapolation_failures": (
            robust["silent_relative_extrapolation_failures"] == 0
        ),
        "robust_gate_controls_accepted_relative_extrapolation_error": (
            robust["max_accepted_extrapolation_rmse_over_noise"] is not None
            and robust["max_accepted_extrapolation_rmse_over_noise"]
            <= NORMALIZED_EXTRAPOLATION_LIMIT
        ),
        "robust_estimator_less_inflated_than_naive_under_contamination": (
            contaminated_robust_overestimate < contaminated_naive_overestimate
        ),
        "all_cells_have_expected_repeat_count": all(
            row["trials"] == REPEATS for row in summary
        ),
    }
    return {
        "checks": checks,
        "route_pass": all(checks.values()),
        "robust_oracle_decision_agreement": agreements,
        "robust_oracle_decision_agreement_rate": agreements / len(records),
        "median_robust_scale_relative_error": float(np.median(robust_errors)),
        "p90_robust_scale_relative_error": float(np.quantile(robust_errors, 0.90)),
        "contaminated_median_robust_scale_ratio": contaminated_robust_overestimate,
        "contaminated_median_naive_scale_ratio": contaminated_naive_overestimate,
        "robust_normalized_gate": robust,
        "naive_normalized_gate": naive,
        "oracle_normalized_gate": oracle,
        "fixed_absolute_gate": fixed,
        "scope": (
            "The estimator requires one densely sampled calibration prefix, iid core "
            "noise, smooth signal curvature, and sparse gross contamination. It is not "
            "validated for temporal correlation or an arbitrarily sparse series."
        ),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "single_series_robust_noise_gate.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Single-series robust noise gate",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        "- Noise scale is estimated from a dense single-series training prefix.",
        "- The robust multiscale estimator is compared with a naive second-difference standard deviation.",
        "",
        "| Noise kind | Base noise | Strength | Robust refusals (95% Wilson) | Median robust scale/base | Median naive scale/base | Median extrap./base |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lower, upper = row["robust_refusal_wilson_95"]
        lines.append(
            f"| {row['noise_kind']} | {row['base_noise']:.1e} | {row['strength']:.3f} | "
            f"{row['robust_refusals']}/{row['trials']} [{lower:.3f}, {upper:.3f}] | "
            f"{row['median_robust_scale_ratio']:.3f} | "
            f"{row['median_naive_scale_ratio']:.3f} | "
            f"{row['median_extrapolation_ratio']:.3f} |"
        )
    lines.extend(["", "## Gate diagnostics", ""])
    for name in (
        "robust_normalized_gate",
        "naive_normalized_gate",
        "oracle_normalized_gate",
        "fixed_absolute_gate",
    ):
        lines.append(f"- {name}: {payload['assessment'][name]}")
    lines.extend(["", "## Prespecified feasibility checks", ""])
    lines.extend(
        f"- {name}: **{value}**"
        for name, value in payload["assessment"]["checks"].items()
    )
    lines.extend(["", payload["assessment"]["scope"]])
    (RESULTS / "single_series_robust_noise_gate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    records = []
    for noise_kind in NOISE_KINDS:
        for base_noise in BASE_NOISE_LEVELS:
            for strength in STRENGTHS:
                for repeat in range(REPEATS):
                    record = evaluate(base_noise, strength, noise_kind, repeat)
                    records.append(record)
                    print(
                        f"kind={noise_kind} noise={base_noise:.1e} "
                        f"strength={strength:.3f} repeat={repeat} "
                        f"rank={record['selected_rank']} "
                        f"robust/base={record['robust_noise_estimate']/base_noise:.3f} "
                        f"naive/base={record['naive_noise_estimate']/base_noise:.3f} "
                        f"train/robust={record['robust_train_rmse_over_noise']:.3f} "
                        f"extra/base={record['extrapolation_rmse_over_oracle_noise']:.3f} "
                        f"oracle={record['oracle_normalized_refusal']} "
                        f"robust={record['robust_normalized_refusal']} "
                        f"elapsed={record['elapsed_seconds']:.1f}s",
                        flush=True,
                    )
    summary = summarize(records)
    payload = {
        "experiment": "single_series_robust_noise_gate",
        "design": {
            "base_noise_levels": BASE_NOISE_LEVELS,
            "strengths": STRENGTHS,
            "noise_kinds": NOISE_KINDS,
            "repeats_per_cell": REPEATS,
            "outlier_fraction": OUTLIER_FRACTION,
            "outlier_scale": OUTLIER_SCALE,
            "dense_calibration_prefix": True,
            "normalized_train_limit": NORMALIZED_TRAIN_LIMIT,
            "normalized_extrapolation_limit": NORMALIZED_EXTRAPOLATION_LIMIT,
            "max_median_scale_error": MAX_MEDIAN_SCALE_ERROR,
            "max_p90_scale_error": MAX_P90_SCALE_ERROR,
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
