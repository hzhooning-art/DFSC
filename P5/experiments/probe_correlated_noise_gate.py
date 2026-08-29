"""Test correlation-aware residual normalization under AR(1) observation noise."""

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
from probe_replicate_noise_gate_transfer import diagnostics
from probe_single_series_robust_noise_gate import multiscale_noise_estimates


NORMALIZED_TRAIN_LIMIT = 4.0
CONDITION_LIMIT = 1.0e8
BASE_NOISE = 8.0e-4
RHOS = (0.0, 0.3, 0.6, 0.85)
STRENGTHS = (0.050, 0.085, 0.200)
REPEATS = 2
NORMALIZED_EXTRAPOLATION_LIMIT = 10.0
MIN_IDENTIFIABLE_FRACTION = 0.50
MIN_ORACLE_DECISION_AGREEMENT = 0.80
MAX_MEDIAN_SCALE_ERROR = 0.35
MAX_P90_SCALE_ERROR = 0.60


def stationary_ar1(
    rng: np.random.Generator,
    length: int,
    channels: int,
    marginal_scale: float,
    rho: float,
) -> np.ndarray:
    """Draw a stationary AR(1) process with the requested marginal scale."""
    burnin = 256
    innovations = rng.normal(
        0.0,
        marginal_scale * np.sqrt(1.0 - rho**2),
        size=(length + burnin, channels),
    )
    process = np.empty_like(innovations)
    process[0] = rng.normal(0.0, marginal_scale, size=channels)
    for index in range(1, process.shape[0]):
        process[index] = rho * process[index - 1] + innovations[index]
    return process[burnin:]


def _fit_profile_ar1(prefix: np.ndarray, degree: int) -> dict:
    """Profile a conditional AR(1) likelihood over a smooth Chebyshev mean."""
    length, channels = prefix.shape
    basis = np.polynomial.chebyshev.chebvander(
        np.linspace(-1.0, 1.0, length), degree
    )
    rho_grid = np.linspace(-0.10, 0.95, 211)
    best = None
    for rho in rho_grid:
        transformed = prefix[1:] - rho * prefix[:-1]
        design = basis[1:] - rho * basis[:-1]
        coefficients = np.linalg.lstsq(design, transformed, rcond=None)[0]
        residual = transformed - design @ coefficients
        sum_squares = float(np.sum(residual**2))
        if best is None or sum_squares < best[0]:
            best = (sum_squares, float(rho), residual)
    sum_squares, rho, residual = best
    parameter_count = (degree + 1) * channels + 1
    degrees_of_freedom = max(residual.size - parameter_count, 1)
    innovation_scale = np.sqrt(sum_squares / degrees_of_freedom)
    marginal_scale = innovation_scale / np.sqrt(max(1.0 - rho**2, 1.0e-12))
    lag_correlations = []
    for channel in range(channels):
        values = residual[:, channel]
        if np.std(values[:-1]) > 0.0 and np.std(values[1:]) > 0.0:
            lag_correlations.append(
                abs(float(np.corrcoef(values[:-1], values[1:])[0, 1]))
            )
    return {
        "marginal_scale": float(marginal_scale),
        "innovation_scale": float(innovation_scale),
        "rho": rho,
        "degree": degree,
        "innovation_lag1_abs_correlation": (
            float(np.median(lag_correlations)) if lag_correlations else float("inf")
        ),
    }


def estimate_ar1_noise(prefix: np.ndarray) -> dict:
    """Estimate AR(1) marginal scale and correlation from a dense prefix."""
    prefix_np = np.asarray(prefix, dtype=float)
    if prefix_np.ndim == 1:
        prefix_np = prefix_np[:, None]
    profile_estimates = [
        _fit_profile_ar1(prefix_np, degree) for degree in (8, 9, 10)
    ]
    estimate = dict(profile_estimates[0])
    profile_rhos = np.asarray([item["rho"] for item in profile_estimates])
    profile_scales = np.asarray(
        [item["marginal_scale"] for item in profile_estimates]
    )
    rho_half_width = 0.5 * float(np.ptp(profile_rhos))
    scale_relative_spread = float(np.ptp(profile_scales)) / max(
        float(np.median(profile_scales)), np.finfo(float).tiny
    )
    estimate["rho_half_width"] = float(rho_half_width)
    estimate["scale_relative_spread"] = float(scale_relative_spread)
    estimate["profile_degrees"] = [8, 9, 10]
    estimate["profile_rhos"] = profile_rhos.tolist()
    estimate["profile_scales"] = profile_scales.tolist()
    estimate["identifiable"] = bool(
        prefix_np.shape[0] >= 64
        and prefix_np.shape[1] >= 3
        and -0.05 <= estimate["rho"] <= 0.94
        and estimate["rho_half_width"] <= 0.15
        and estimate["scale_relative_spread"] <= 0.30
        and estimate["innovation_lag1_abs_correlation"] <= 0.20
    )
    return estimate


def correlation_aware_refusal(
    train_rmse: float,
    estimated_scale: float,
    condition: float,
    identifiable: bool,
) -> bool:
    """Refuse when the correlation model is unusable or the residual gate fails."""
    return bool(
        not identifiable
        or not np.isfinite(estimated_scale)
        or estimated_scale <= 0.0
        or condition > CONDITION_LIMIT
        or train_rmse / estimated_scale > NORMALIZED_TRAIN_LIMIT
    )


def make_dataset(rho: float, strength: float, repeat: int) -> dict:
    seed = 817000 + 10007 * RHOS.index(rho) + 1009 * STRENGTHS.index(strength) + 101 * repeat
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
    noise_np = stationary_ar1(
        rng,
        NUM_POINTS,
        CHANNELS,
        marginal_scale=BASE_NOISE,
        rho=rho,
    )
    observations_np = clean_np + noise_np
    split = int(round(TRAIN_END_FRACTION * (NUM_POINTS - 1)))
    pool = np.arange(1, split + 1)
    sampled = np.sort(rng.choice(pool, size=TRAIN_COUNT - 1, replace=False))
    train_np = np.concatenate(([0], sampled))
    interpolation_np = np.setdiff1d(np.arange(split + 1), train_np)
    extrapolation_np = np.arange(split + 1, NUM_POINTS)
    prefix = observations_np[: split + 1]
    aware = estimate_ar1_noise(prefix)
    iid_scale, _ = multiscale_noise_estimates(prefix)
    return {
        "seed": seed,
        "rho": rho,
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
        "aware_estimate": aware,
        "iid_scale_estimate": iid_scale,
    }


def evaluate(rho: float, strength: float, repeat: int) -> dict:
    started = time.perf_counter()
    data = make_dataset(rho, strength, repeat)
    metadata, prediction = fit_mechanism(data)
    winner = selected_candidate(metadata)
    clean = data["clean"].detach().cpu().numpy()
    extrapolation_idx = data["extrapolation_idx"].detach().cpu().numpy()
    extrapolation_rmse = rmse(
        prediction[extrapolation_idx], clean[extrapolation_idx]
    )
    aware = data["aware_estimate"]
    iid_scale = data["iid_scale_estimate"]
    condition = metadata["condition"]
    oracle_ratio = winner["train_rmse"] / BASE_NOISE
    iid_ratio = winner["train_rmse"] / iid_scale
    aware_ratio = winner["train_rmse"] / aware["marginal_scale"]
    extrapolation_ratio = extrapolation_rmse / BASE_NOISE
    ill_conditioned = condition > CONDITION_LIMIT
    return {
        "rho": rho,
        "strength": strength,
        "repeat": repeat,
        "seed": data["seed"],
        "selected_rank": metadata["selected_rank"],
        "condition": condition,
        "train_rmse": winner["train_rmse"],
        "estimated_rho": aware["rho"],
        "rho_absolute_error": abs(aware["rho"] - rho),
        "rho_half_width": aware["rho_half_width"],
        "scale_relative_spread": aware["scale_relative_spread"],
        "innovation_lag1_abs_correlation": aware[
            "innovation_lag1_abs_correlation"
        ],
        "correlation_identifiable": aware["identifiable"],
        "aware_marginal_scale_estimate": aware["marginal_scale"],
        "aware_innovation_scale_estimate": aware["innovation_scale"],
        "iid_scale_estimate": iid_scale,
        "aware_scale_relative_error": abs(aware["marginal_scale"] - BASE_NOISE)
        / BASE_NOISE,
        "iid_scale_relative_error": abs(iid_scale - BASE_NOISE) / BASE_NOISE,
        "oracle_train_rmse_over_noise": oracle_ratio,
        "aware_train_rmse_over_noise": aware_ratio,
        "iid_train_rmse_over_noise": iid_ratio,
        "extrapolation_rmse": extrapolation_rmse,
        "extrapolation_rmse_over_oracle_noise": extrapolation_ratio,
        "oracle_normalized_refusal": bool(
            ill_conditioned or oracle_ratio > NORMALIZED_TRAIN_LIMIT
        ),
        "aware_normalized_refusal": correlation_aware_refusal(
            winner["train_rmse"],
            aware["marginal_scale"],
            condition,
            aware["identifiable"],
        ),
        "iid_normalized_refusal": bool(
            ill_conditioned or iid_ratio > NORMALIZED_TRAIN_LIMIT
        ),
        "normalized_relative_extrapolation_failure": bool(
            extrapolation_ratio > NORMALIZED_EXTRAPOLATION_LIMIT
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for rho in RHOS:
        for strength in STRENGTHS:
            group = [
                record
                for record in records
                if record["rho"] == rho and record["strength"] == strength
            ]
            refused = sum(record["aware_normalized_refusal"] for record in group)
            lower, upper = wilson_interval(refused, len(group))
            rows.append(
                {
                    "rho": rho,
                    "strength": strength,
                    "trials": len(group),
                    "identifiable": sum(
                        record["correlation_identifiable"] for record in group
                    ),
                    "aware_refusals": refused,
                    "aware_refusal_wilson_95": [lower, upper],
                    "median_estimated_rho": float(
                        np.median([record["estimated_rho"] for record in group])
                    ),
                    "median_aware_scale_ratio": float(
                        np.median(
                            [
                                record["aware_marginal_scale_estimate"] / BASE_NOISE
                                for record in group
                            ]
                        )
                    ),
                    "median_iid_scale_ratio": float(
                        np.median(
                            [
                                record["iid_scale_estimate"] / BASE_NOISE
                                for record in group
                            ]
                        )
                    ),
                    "median_extrapolation_ratio": float(
                        np.median(
                            [
                                record["extrapolation_rmse_over_oracle_noise"]
                                for record in group
                            ]
                        )
                    ),
                }
            )
    return rows


def assess(records: list[dict], summary: list[dict]) -> dict:
    aware = diagnostics(records, "aware_normalized_refusal")
    iid = diagnostics(records, "iid_normalized_refusal")
    oracle = diagnostics(records, "oracle_normalized_refusal")
    identifiable = [record for record in records if record["correlation_identifiable"]]
    unidentifiable = [
        record for record in records if not record["correlation_identifiable"]
    ]
    agreements = sum(
        record["aware_normalized_refusal"]
        == record["oracle_normalized_refusal"]
        for record in identifiable
    )
    scale_errors = np.asarray(
        [record["aware_scale_relative_error"] for record in identifiable]
    )
    rho_errors = np.asarray([record["rho_absolute_error"] for record in identifiable])
    agreement_rate = agreements / len(identifiable) if identifiable else 0.0
    max_accepted = aware["max_accepted_extrapolation_rmse_over_noise"]
    checks = {
        "at_least_half_of_cases_are_identifiable": (
            len(identifiable) / len(records) >= MIN_IDENTIFIABLE_FRACTION
        ),
        "median_aware_scale_error_within_limit": (
            bool(scale_errors.size)
            and float(np.median(scale_errors)) <= MAX_MEDIAN_SCALE_ERROR
        ),
        "p90_aware_scale_error_within_limit": (
            bool(scale_errors.size)
            and float(np.quantile(scale_errors, 0.90)) <= MAX_P90_SCALE_ERROR
        ),
        "aware_gate_agrees_with_oracle_at_prespecified_rate": (
            agreement_rate >= MIN_ORACLE_DECISION_AGREEMENT
        ),
        "aware_gate_has_no_silent_relative_extrapolation_failures": (
            aware["silent_relative_extrapolation_failures"] == 0
        ),
        "aware_gate_controls_accepted_relative_extrapolation_error": (
            max_accepted is not None
            and max_accepted <= NORMALIZED_EXTRAPOLATION_LIMIT
        ),
        "all_unidentifiable_cases_are_refused": all(
            record["aware_normalized_refusal"] for record in unidentifiable
        ),
        "aware_gate_not_less_safe_than_iid_gate": (
            aware["silent_relative_extrapolation_failures"]
            <= iid["silent_relative_extrapolation_failures"]
        ),
        "all_cells_have_expected_repeat_count": all(
            row["trials"] == REPEATS for row in summary
        ),
    }
    return {
        "checks": checks,
        "route_pass": all(checks.values()),
        "identifiable": len(identifiable),
        "unidentifiable": len(unidentifiable),
        "identifiable_fraction": len(identifiable) / len(records),
        "aware_oracle_decision_agreement": agreements,
        "aware_oracle_decision_agreement_rate_identifiable": agreement_rate,
        "median_aware_scale_relative_error_identifiable": (
            float(np.median(scale_errors)) if scale_errors.size else None
        ),
        "p90_aware_scale_relative_error_identifiable": (
            float(np.quantile(scale_errors, 0.90)) if scale_errors.size else None
        ),
        "median_rho_absolute_error_identifiable": (
            float(np.median(rho_errors)) if rho_errors.size else None
        ),
        "aware_normalized_gate": aware,
        "iid_normalized_gate": iid,
        "oracle_normalized_gate": oracle,
        "scope": (
            "The correlation model is stationary AR(1), shared only at the model-class "
            "level across channels, and estimated from one dense calibration prefix. "
            "Long-memory, nonstationary, and irregularly sampled noise remain untested."
        ),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "correlated_noise_gate.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Correlation-aware noise gate",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        "- Observation noise follows a stationary AR(1) process.",
        "- Unidentifiable correlation forces refusal.",
        "",
        "| rho | Strength | Identifiable | Aware refusals (95% Wilson) | Median estimated rho | Median aware scale/base | Median iid scale/base | Median extrap./base |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lower, upper = row["aware_refusal_wilson_95"]
        lines.append(
            f"| {row['rho']:.2f} | {row['strength']:.3f} | "
            f"{row['identifiable']}/{row['trials']} | "
            f"{row['aware_refusals']}/{row['trials']} [{lower:.3f}, {upper:.3f}] | "
            f"{row['median_estimated_rho']:.3f} | "
            f"{row['median_aware_scale_ratio']:.3f} | "
            f"{row['median_iid_scale_ratio']:.3f} | "
            f"{row['median_extrapolation_ratio']:.3f} |"
        )
    lines.extend(["", "## Gate diagnostics", ""])
    for name in (
        "aware_normalized_gate",
        "iid_normalized_gate",
        "oracle_normalized_gate",
    ):
        lines.append(f"- {name}: {payload['assessment'][name]}")
    lines.extend(["", "## Prespecified feasibility checks", ""])
    lines.extend(
        f"- {name}: **{value}**"
        for name, value in payload["assessment"]["checks"].items()
    )
    lines.extend(["", payload["assessment"]["scope"]])
    (RESULTS / "correlated_noise_gate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    records = []
    for rho in RHOS:
        for strength in STRENGTHS:
            for repeat in range(REPEATS):
                record = evaluate(rho, strength, repeat)
                records.append(record)
                print(
                    f"rho={rho:.2f} strength={strength:.3f} repeat={repeat} "
                    f"rank={record['selected_rank']} "
                    f"rho_hat={record['estimated_rho']:.3f} "
                    f"half_width={record['rho_half_width']:.3f} "
                    f"identifiable={record['correlation_identifiable']} "
                    f"aware/base={record['aware_marginal_scale_estimate']/BASE_NOISE:.3f} "
                    f"iid/base={record['iid_scale_estimate']/BASE_NOISE:.3f} "
                    f"extra/base={record['extrapolation_rmse_over_oracle_noise']:.3f} "
                    f"aware_refuse={record['aware_normalized_refusal']} "
                    f"elapsed={record['elapsed_seconds']:.1f}s",
                    flush=True,
                )
    summary = summarize(records)
    payload = {
        "experiment": "correlated_noise_gate",
        "design": {
            "base_noise": BASE_NOISE,
            "rhos": RHOS,
            "strengths": STRENGTHS,
            "repeats_per_cell": REPEATS,
            "normalized_train_limit": NORMALIZED_TRAIN_LIMIT,
            "normalized_extrapolation_limit": NORMALIZED_EXTRAPOLATION_LIMIT,
            "minimum_identifiable_fraction": MIN_IDENTIFIABLE_FRACTION,
            "minimum_oracle_decision_agreement": MIN_ORACLE_DECISION_AGREEMENT,
            "max_median_scale_error": MAX_MEDIAN_SCALE_ERROR,
            "max_p90_scale_error": MAX_P90_SCALE_ERROR,
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
