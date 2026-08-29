"""Test active refusal when AR(1) normalization faces long-memory noise."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from scipy.signal import fftconvolve
from scipy.stats import chi2

from probe_controlled_misspecification import RATES, simulate_channel
from probe_correlated_noise_gate import (
    BASE_NOISE,
    CONDITION_LIMIT,
    NORMALIZED_EXTRAPOLATION_LIMIT,
    NORMALIZED_TRAIN_LIMIT,
    STRENGTHS,
    correlation_aware_refusal,
    estimate_ar1_noise,
)
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


AR1_ADEQUACY_ALPHA = 0.01
MEMORY_ORDERS = (0.0, 0.15, 0.30, 0.45)
REPEATS = 2
MIN_CONTROL_ADEQUACY_RATE = 0.75
MIN_STRONG_MEMORY_DETECTION_RATE = 0.75


def fractional_noise(
    rng: np.random.Generator,
    length: int,
    channels: int,
    marginal_scale: float,
    d: float,
) -> np.ndarray:
    """Draw finite-burn-in ARFIMA(0,d,0) noise with requested marginal scale."""
    if not 0.0 <= d < 0.5:
        raise ValueError("d must satisfy 0 <= d < 0.5")
    burnin = max(4096, length // 2)
    total = length + burnin
    weights = np.empty(total, dtype=float)
    weights[0] = 1.0
    for index in range(1, total):
        weights[index] = weights[index - 1] * (index - 1 + d) / index
    innovation_scale = marginal_scale / np.sqrt(np.sum(weights**2))
    innovations = rng.normal(0.0, innovation_scale, size=(total, channels))
    process = np.column_stack(
        [fftconvolve(innovations[:, channel], weights, mode="full")[:total]
         for channel in range(channels)]
    )
    return process[burnin:]


def ar1_whiteness_diagnostic(
    prefix: np.ndarray,
    rho: float,
    degree: int = 8,
    max_lag: int = 8,
) -> dict:
    """Apply a pooled Ljung-Box check to conditional AR(1) innovations."""
    values = np.asarray(prefix, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    length, channels = values.shape
    if length <= max_lag + degree + 2:
        return {
            "adequate": False,
            "p_value": 0.0,
            "statistic": float("inf"),
            "degrees_of_freedom": channels * max_lag,
        }
    basis = np.polynomial.chebyshev.chebvander(
        np.linspace(-1.0, 1.0, length), degree
    )
    transformed = values[1:] - rho * values[:-1]
    design = basis[1:] - rho * basis[:-1]
    coefficients = np.linalg.lstsq(design, transformed, rcond=None)[0]
    residual = transformed - design @ coefficients
    residual_length = residual.shape[0]
    statistic = 0.0
    for channel in range(channels):
        centered = residual[:, channel] - np.mean(residual[:, channel])
        denominator = float(np.dot(centered, centered))
        if denominator <= np.finfo(float).tiny:
            return {
                "adequate": False,
                "p_value": 0.0,
                "statistic": float("inf"),
                "degrees_of_freedom": channels * max_lag,
            }
        for lag in range(1, max_lag + 1):
            correlation = float(np.dot(centered[lag:], centered[:-lag]) / denominator)
            statistic += (
                residual_length
                * (residual_length + 2.0)
                * correlation**2
                / (residual_length - lag)
            )
    degrees_of_freedom = channels * max_lag
    p_value = float(chi2.sf(statistic, degrees_of_freedom))
    return {
        "adequate": bool(p_value >= AR1_ADEQUACY_ALPHA),
        "p_value": p_value,
        "statistic": float(statistic),
        "degrees_of_freedom": degrees_of_freedom,
    }


def long_memory_aware_refusal(
    train_rmse: float,
    estimated_scale: float,
    condition: float,
    identifiable: bool,
    ar1_adequate: bool,
) -> bool:
    """Refuse if either AR(1) estimation or its whiteness audit is invalid."""
    return bool(
        not ar1_adequate
        or correlation_aware_refusal(
            train_rmse,
            estimated_scale,
            condition,
            identifiable,
        )
    )


def make_dataset(d: float, strength: float, repeat: int) -> dict:
    """Create one deterministic mechanism-fit dataset with fractional noise."""
    seed = (
        927000
        + 10007 * MEMORY_ORDERS.index(d)
        + 1009 * STRENGTHS.index(strength)
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
    noise_np = fractional_noise(
        rng,
        NUM_POINTS,
        CHANNELS,
        marginal_scale=BASE_NOISE,
        d=d,
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
    whiteness = ar1_whiteness_diagnostic(
        prefix, rho=aware["rho"], degree=8, max_lag=8
    )
    iid_scale, _ = multiscale_noise_estimates(prefix)
    return {
        "seed": seed,
        "d": d,
        "strength": strength,
        "times": torch.tensor(times_np, dtype=DTYPE, device=DEVICE),
        "clean": torch.tensor(clean_np, dtype=DTYPE, device=DEVICE),
        "observations": torch.tensor(
            observations_np, dtype=DTYPE, device=DEVICE
        ),
        "train_idx": torch.tensor(train_np, dtype=torch.long, device=DEVICE),
        "interpolation_idx": torch.tensor(
            interpolation_np, dtype=torch.long, device=DEVICE
        ),
        "extrapolation_idx": torch.tensor(
            extrapolation_np, dtype=torch.long, device=DEVICE
        ),
        "true_rank": 2,
        "aware_estimate": aware,
        "ar1_whiteness": whiteness,
        "iid_scale_estimate": iid_scale,
    }


def evaluate(d: float, strength: float, repeat: int) -> dict:
    started = time.perf_counter()
    data = make_dataset(d, strength, repeat)
    metadata, prediction = fit_mechanism(data)
    winner = selected_candidate(metadata)
    clean = data["clean"].detach().cpu().numpy()
    extrapolation_idx = data["extrapolation_idx"].detach().cpu().numpy()
    extrapolation_rmse = rmse(
        prediction[extrapolation_idx], clean[extrapolation_idx]
    )
    aware = data["aware_estimate"]
    whiteness = data["ar1_whiteness"]
    iid_scale = data["iid_scale_estimate"]
    condition = metadata["condition"]
    oracle_ratio = winner["train_rmse"] / BASE_NOISE
    aware_ratio = winner["train_rmse"] / aware["marginal_scale"]
    iid_ratio = winner["train_rmse"] / iid_scale
    extrapolation_ratio = extrapolation_rmse / BASE_NOISE
    ill_conditioned = condition > CONDITION_LIMIT
    return {
        "d": d,
        "strength": strength,
        "repeat": repeat,
        "seed": data["seed"],
        "selected_rank": metadata["selected_rank"],
        "condition": condition,
        "train_rmse": winner["train_rmse"],
        "estimated_ar1_rho": aware["rho"],
        "rho_half_width": aware["rho_half_width"],
        "ar1_estimate_identifiable": aware["identifiable"],
        "ar1_whiteness_p_value": whiteness["p_value"],
        "ar1_whiteness_statistic": whiteness["statistic"],
        "ar1_model_adequate": whiteness["adequate"],
        "long_memory_mismatch_detected": not whiteness["adequate"],
        "aware_marginal_scale_estimate": aware["marginal_scale"],
        "aware_scale_relative_error": abs(
            aware["marginal_scale"] - BASE_NOISE
        )
        / BASE_NOISE,
        "iid_scale_estimate": iid_scale,
        "oracle_train_rmse_over_noise": oracle_ratio,
        "aware_train_rmse_over_noise": aware_ratio,
        "iid_train_rmse_over_noise": iid_ratio,
        "extrapolation_rmse": extrapolation_rmse,
        "extrapolation_rmse_over_oracle_noise": extrapolation_ratio,
        "oracle_normalized_refusal": bool(
            ill_conditioned or oracle_ratio > NORMALIZED_TRAIN_LIMIT
        ),
        "ar1_only_refusal": correlation_aware_refusal(
            winner["train_rmse"],
            aware["marginal_scale"],
            condition,
            aware["identifiable"],
        ),
        "mismatch_aware_refusal": long_memory_aware_refusal(
            winner["train_rmse"],
            aware["marginal_scale"],
            condition,
            aware["identifiable"],
            whiteness["adequate"],
        ),
        "iid_normalized_refusal": bool(
            ill_conditioned or iid_ratio > NORMALIZED_TRAIN_LIMIT
        ),
        "normalized_relative_extrapolation_failure": bool(
            extrapolation_ratio > NORMALIZED_EXTRAPOLATION_LIMIT
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def assess_prescreen(records: list[dict], expected_count: int) -> dict:
    """Apply frozen go/no-go checks before expensive mechanism fitting."""
    controls = [record for record in records if record["d"] == 0.0]
    strong_memory = [record for record in records if record["d"] >= 0.30]
    control_adequacy_rate = (
        sum(record["ar1_model_adequate"] for record in controls) / len(controls)
        if controls
        else 0.0
    )
    strong_detection_rate = (
        sum(
            record["long_memory_mismatch_detected"] for record in strong_memory
        )
        / len(strong_memory)
        if strong_memory
        else 0.0
    )
    checks = {
        "all_prescreen_cells_complete": len(records) == expected_count,
        "control_ar1_adequacy_rate_within_limit": (
            control_adequacy_rate >= MIN_CONTROL_ADEQUACY_RATE
        ),
        "strong_long_memory_detection_rate_within_limit": (
            strong_detection_rate >= MIN_STRONG_MEMORY_DETECTION_RATE
        ),
    }
    return {
        "checks": checks,
        "route_pass": all(checks.values()),
        "control_adequacy_rate": control_adequacy_rate,
        "strong_long_memory_detection_rate": strong_detection_rate,
        "expected_count": expected_count,
        "observed_count": len(records),
    }


def run_prescreen() -> dict:
    records = []
    for d in MEMORY_ORDERS:
        for strength in STRENGTHS:
            for repeat in range(REPEATS):
                data = make_dataset(d, strength, repeat)
                whiteness = data["ar1_whiteness"]
                records.append(
                    {
                        "d": d,
                        "strength": strength,
                        "repeat": repeat,
                        "seed": data["seed"],
                        "estimated_ar1_rho": data["aware_estimate"]["rho"],
                        "ar1_estimate_identifiable": data["aware_estimate"][
                            "identifiable"
                        ],
                        "ar1_whiteness_p_value": whiteness["p_value"],
                        "ar1_whiteness_statistic": whiteness["statistic"],
                        "ar1_model_adequate": whiteness["adequate"],
                        "long_memory_mismatch_detected": not whiteness[
                            "adequate"
                        ],
                    }
                )
    expected_count = len(MEMORY_ORDERS) * len(STRENGTHS) * REPEATS
    return {
        "experiment": "long_memory_mismatch_prescreen",
        "design": {
            "memory_orders": MEMORY_ORDERS,
            "strengths": STRENGTHS,
            "repeats_per_cell": REPEATS,
            "ar1_adequacy_alpha": AR1_ADEQUACY_ALPHA,
            "minimum_control_adequacy_rate": MIN_CONTROL_ADEQUACY_RATE,
            "minimum_strong_memory_detection_rate": MIN_STRONG_MEMORY_DETECTION_RATE,
        },
        "records": records,
        "assessment": assess_prescreen(records, expected_count),
    }


def write_prescreen_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "long_memory_mismatch_prescreen.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    assessment = payload["assessment"]
    lines = [
        "# Long-memory mismatch prescreen",
        "",
        f"- Route pass: **{assessment['route_pass']}**",
        f"- Control AR(1) adequacy rate: {assessment['control_adequacy_rate']:.3f}",
        f"- Strong-memory detection rate: {assessment['strong_long_memory_detection_rate']:.3f}",
        "- The expensive mechanism-fit matrix is skipped when this prescreen fails.",
        "",
        "| d | Strength | Repeat | Estimated AR1 rho | Whiteness p | Adequate |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for record in payload["records"]:
        lines.append(
            f"| {record['d']:.2f} | {record['strength']:.3f} | "
            f"{record['repeat']} | {record['estimated_ar1_rho']:.3f} | "
            f"{record['ar1_whiteness_p_value']:.3g} | "
            f"{record['ar1_model_adequate']} |"
        )
    lines.extend(["", "## Frozen checks", ""])
    for name, passed in assessment["checks"].items():
        lines.append(f"- {name}: **{passed}**")
    lines.extend(
        [
            "",
            "The pooled Ljung-Box diagnostic does not separate the iid control "
            "from strong finite-sample ARFIMA noise reliably enough under the "
            "frozen thresholds. The route is rejected without tuning the test "
            "after observing these outcomes.",
        ]
    )
    (RESULTS / "long_memory_mismatch_prescreen.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for d in MEMORY_ORDERS:
        for strength in STRENGTHS:
            group = [
                record
                for record in records
                if record["d"] == d and record["strength"] == strength
            ]
            detected = sum(
                record["long_memory_mismatch_detected"] for record in group
            )
            refused = sum(record["mismatch_aware_refusal"] for record in group)
            lower, upper = wilson_interval(refused, len(group))
            rows.append(
                {
                    "d": d,
                    "strength": strength,
                    "trials": len(group),
                    "mismatch_detected": detected,
                    "mismatch_aware_refusals": refused,
                    "refusal_wilson_95": [lower, upper],
                    "median_ar1_rho": float(
                        np.median([record["estimated_ar1_rho"] for record in group])
                    ),
                    "median_whiteness_p_value": float(
                        np.median(
                            [record["ar1_whiteness_p_value"] for record in group]
                        )
                    ),
                    "median_scale_ratio": float(
                        np.median(
                            [
                                record["aware_marginal_scale_estimate"]
                                / BASE_NOISE
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
    mismatch = diagnostics(records, "mismatch_aware_refusal")
    ar1_only = diagnostics(records, "ar1_only_refusal")
    oracle = diagnostics(records, "oracle_normalized_refusal")
    controls = [record for record in records if record["d"] == 0.0]
    strong_memory = [record for record in records if record["d"] >= 0.30]
    control_adequacy_rate = sum(
        record["ar1_model_adequate"] for record in controls
    ) / len(controls)
    strong_detection_rate = sum(
        record["long_memory_mismatch_detected"] for record in strong_memory
    ) / len(strong_memory)
    detected = [
        record for record in records if record["long_memory_mismatch_detected"]
    ]
    max_accepted = mismatch["max_accepted_extrapolation_rmse_over_noise"]
    checks = {
        "control_ar1_adequacy_rate_within_limit": (
            control_adequacy_rate >= MIN_CONTROL_ADEQUACY_RATE
        ),
        "strong_long_memory_detection_rate_within_limit": (
            strong_detection_rate >= MIN_STRONG_MEMORY_DETECTION_RATE
        ),
        "all_detected_mismatches_are_refused": all(
            record["mismatch_aware_refusal"] for record in detected
        ),
        "mismatch_gate_has_no_silent_relative_extrapolation_failures": (
            mismatch["silent_relative_extrapolation_failures"] == 0
        ),
        "mismatch_gate_not_less_safe_than_ar1_only_gate": (
            mismatch["silent_relative_extrapolation_failures"]
            <= ar1_only["silent_relative_extrapolation_failures"]
        ),
        "mismatch_gate_controls_accepted_relative_extrapolation_error": (
            max_accepted is not None
            and max_accepted <= NORMALIZED_EXTRAPOLATION_LIMIT
        ),
        "control_group_has_at_least_one_acceptance": any(
            not record["mismatch_aware_refusal"] for record in controls
        ),
        "all_cells_have_expected_repeat_count": all(
            row["trials"] == REPEATS for row in summary
        ),
    }
    return {
        "checks": checks,
        "route_pass": all(checks.values()),
        "control_ar1_adequacy_rate": control_adequacy_rate,
        "strong_long_memory_detection_rate": strong_detection_rate,
        "detected_mismatches": len(detected),
        "mismatch_aware_gate": mismatch,
        "ar1_only_gate": ar1_only,
        "oracle_normalized_gate": oracle,
        "scope": (
            "The mismatch audit covers finite-burn-in ARFIMA(0,d,0) Gaussian "
            "noise on one dense, regularly sampled calibration prefix. It does "
            "not validate general fractional Gaussian noise, nonstationarity, "
            "irregular sampling, or cross-channel dependence."
        ),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "long_memory_mismatch_gate.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Long-memory mismatch gate",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        "- Noise model: finite-burn-in ARFIMA(0,d,0).",
        "- AR(1) inadequacy forces refusal.",
        "",
        "| d | Strength | Detected | Refusals (95% Wilson) | Median AR1 rho | Median p | Median scale/base | Median extrap./base |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lower, upper = row["refusal_wilson_95"]
        lines.append(
            f"| {row['d']:.2f} | {row['strength']:.3f} | "
            f"{row['mismatch_detected']}/{row['trials']} | "
            f"{row['mismatch_aware_refusals']}/{row['trials']} "
            f"[{lower:.3f}, {upper:.3f}] | "
            f"{row['median_ar1_rho']:.3f} | "
            f"{row['median_whiteness_p_value']:.3g} | "
            f"{row['median_scale_ratio']:.3f} | "
            f"{row['median_extrapolation_ratio']:.3f} |"
        )
    lines.extend(["", "## Gate diagnostics", ""])
    for name in ("mismatch_aware_gate", "ar1_only_gate", "oracle_normalized_gate"):
        lines.append(f"- {name}: {payload['assessment'][name]}")
    lines.extend(["", "## Prespecified feasibility checks", ""])
    for name, passed in payload["assessment"]["checks"].items():
        lines.append(f"- {name}: **{passed}**")
    lines.extend(["", payload["assessment"]["scope"]])
    (RESULTS / "long_memory_mismatch_gate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    prescreen = run_prescreen()
    write_prescreen_outputs(prescreen)
    print(json.dumps(prescreen["assessment"], indent=2), flush=True)
    if not prescreen["assessment"]["route_pass"]:
        print("prescreen_failed: skipping mechanism-fit matrix", flush=True)
        return
    records = []
    for d in MEMORY_ORDERS:
        for strength in STRENGTHS:
            for repeat in range(REPEATS):
                record = evaluate(d, strength, repeat)
                records.append(record)
                print(
                    f"d={d:.2f} strength={strength:.3f} repeat={repeat} "
                    f"rank={record['selected_rank']} "
                    f"rho_hat={record['estimated_ar1_rho']:.3f} "
                    f"p={record['ar1_whiteness_p_value']:.3g} "
                    f"adequate={record['ar1_model_adequate']} "
                    f"extra/base={record['extrapolation_rmse_over_oracle_noise']:.3f} "
                    f"refuse={record['mismatch_aware_refusal']} "
                    f"elapsed={record['elapsed_seconds']:.1f}s",
                    flush=True,
                )
    summary = summarize(records)
    assessment = assess(records, summary)
    payload = {
        "experiment": "long_memory_mismatch_gate",
        "design": {
            "memory_orders": MEMORY_ORDERS,
            "strengths": STRENGTHS,
            "repeats_per_cell": REPEATS,
            "base_marginal_noise": BASE_NOISE,
            "ar1_adequacy_alpha": AR1_ADEQUACY_ALPHA,
            "minimum_control_adequacy_rate": MIN_CONTROL_ADEQUACY_RATE,
            "minimum_strong_memory_detection_rate": MIN_STRONG_MEMORY_DETECTION_RATE,
            "normalized_training_limit": NORMALIZED_TRAIN_LIMIT,
            "normalized_extrapolation_limit": NORMALIZED_EXTRAPOLATION_LIMIT,
        },
        "records": records,
        "summary": summary,
        "assessment": assessment,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_outputs(payload)
    print(json.dumps(assessment, indent=2), flush=True)
    print(f"elapsed_seconds={payload['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
