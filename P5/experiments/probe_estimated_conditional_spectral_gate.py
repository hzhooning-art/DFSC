"""Estimated-mean conditional null for spectral long-memory detection."""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.interpolate import UnivariateSpline

from probe_correlated_noise_gate import BASE_NOISE
from probe_mechanism_vs_trajectory_baselines import RESULTS
from probe_spectral_long_memory_gate import (
    CALIBRATION_DRAWS,
    MEMORY_ORDERS,
    NULL_QUANTILE,
    REPEATS,
    STRENGTHS,
    assess_length,
    make_prefix_case,
    spectral_memory_statistic,
)


ELIGIBLE_PREFIX_LENGTHS = (256, 512)


def estimate_smooth_mean(
    observations: np.ndarray, noise_scale: float = BASE_NOISE
) -> np.ndarray:
    """Estimate a multichannel smooth mean under a declared iid noise scale."""
    values = np.asarray(observations, dtype=float)
    if values.ndim != 2:
        raise ValueError("observations must have shape (time, channels)")
    positions = np.linspace(0.0, 1.0, values.shape[0])
    smoothing = values.shape[0] * noise_scale**2
    estimate = np.empty_like(values)
    for channel in range(values.shape[1]):
        spline = UnivariateSpline(
            positions,
            values[:, channel],
            k=3,
            s=smoothing,
        )
        estimate[:, channel] = spline(positions)
    return estimate


def make_independent_control(length: int, strength: float) -> dict:
    """Generate a calibration control disjoint from project evaluation seeds."""
    base = make_prefix_case(length, d=0.0, strength=strength, repeat=0)
    seed = (
        2100000
        + 100003 * ELIGIBLE_PREFIX_LENGTHS.index(length)
        + 1009 * STRENGTHS.index(strength)
    )
    rng = np.random.default_rng(seed)
    observations = base["clean"] + rng.normal(
        0.0, BASE_NOISE, size=base["clean"].shape
    )
    return {
        "seed": seed,
        "clean": base["clean"],
        "observations": observations,
    }


def estimated_conditional_threshold(
    calibration_observations: np.ndarray,
    draws: int,
    seed: int,
    quantile: float,
) -> float:
    """Calibrate a conditional null while refitting its estimated mean."""
    calibration = np.asarray(calibration_observations, dtype=float)
    fitted_mean = estimate_smooth_mean(calibration)
    rng = np.random.default_rng(seed)
    statistics = []
    for _ in range(draws):
        pseudo_calibration = fitted_mean + rng.normal(
            0.0, BASE_NOISE, size=fitted_mean.shape
        )
        refitted_mean = estimate_smooth_mean(pseudo_calibration)
        pseudo_evaluation = refitted_mean + rng.normal(
            0.0, BASE_NOISE, size=fitted_mean.shape
        )
        statistics.append(spectral_memory_statistic(pseudo_evaluation))
    return float(np.quantile(statistics, quantile, method="higher"))


def run_estimated_conditional_map(
    prefix_lengths: tuple[int, ...] = ELIGIBLE_PREFIX_LENGTHS,
    memory_orders: tuple[float, ...] = MEMORY_ORDERS,
    strengths: tuple[float, ...] = STRENGTHS,
    repeats: int = REPEATS,
    calibration_draws: int = CALIBRATION_DRAWS,
) -> dict:
    """Run the frozen independent-control conditional feasibility matrix."""
    started = time.perf_counter()
    records = []
    assessments = {}
    expected_per_length = len(memory_orders) * len(strengths) * repeats
    for length in prefix_lengths:
        length_records = []
        thresholds = {}
        for strength in strengths:
            control = make_independent_control(length, strength)
            threshold_seed = (
                2300000
                + 100003 * ELIGIBLE_PREFIX_LENGTHS.index(length)
                + 1009 * STRENGTHS.index(strength)
            )
            threshold = estimated_conditional_threshold(
                control["observations"],
                draws=calibration_draws,
                seed=threshold_seed,
                quantile=NULL_QUANTILE,
            )
            fitted_mean = estimate_smooth_mean(control["observations"])
            thresholds[str(strength)] = {
                "threshold": threshold,
                "seed": threshold_seed,
                "control_seed": control["seed"],
                "mean_estimation_rmse": float(
                    np.sqrt(np.mean((fitted_mean - control["clean"]) ** 2))
                ),
            }
            for d in memory_orders:
                for repeat in range(repeats):
                    case = make_prefix_case(length, d, strength, repeat)
                    statistic = spectral_memory_statistic(case["observations"])
                    record = {
                        "length": length,
                        "d": d,
                        "strength": strength,
                        "repeat": repeat,
                        "seed": case["seed"],
                        "statistic": statistic,
                        "threshold": threshold,
                        "threshold_seed": threshold_seed,
                        "control_seed": control["seed"],
                        "margin": statistic - threshold,
                        "mismatch_detected": statistic > threshold,
                    }
                    records.append(record)
                    length_records.append(record)
        assessment = assess_length(length_records, expected_per_length)
        assessment["conditional_thresholds"] = thresholds
        assessments[str(length)] = assessment

    passing_lengths = [
        length
        for length in prefix_lengths
        if assessments[str(length)]["route_pass"]
    ]
    return {
        "experiment": "estimated_conditional_spectral_long_memory_feasibility",
        "protocol": {
            "prefix_lengths": list(prefix_lengths),
            "memory_orders": list(memory_orders),
            "strengths": list(strengths),
            "repeats": repeats,
            "calibration_draws": calibration_draws,
            "null_quantile": NULL_QUANTILE,
            "independent_control_trajectory": True,
            "mean_refit_in_bootstrap": True,
            "known_noise_scale": BASE_NOISE,
            "decision_frozen_before_project_run": True,
        },
        "records": records,
        "assessment_by_length": assessments,
        "minimal_passing_length": min(passing_lengths)
        if passing_lengths
        else None,
        "route_pass": bool(passing_lengths),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _markdown_report(result: dict) -> str:
    lines = [
        "# Estimated-mean conditional spectral feasibility map",
        "",
        "Each null uses an independent iid calibration trajectory and refits its smooth mean inside every bootstrap draw.",
        "",
        "| Prefix length | Control adequacy | Strong-memory detection | Route pass |",
        "|---:|---:|---:|:---:|",
    ]
    for length in result["protocol"]["prefix_lengths"]:
        assessment = result["assessment_by_length"][str(length)]
        lines.append(
            "| {length} | {control:.3f} | {strong:.3f} | {passed} |".format(
                length=length,
                control=assessment["control_adequacy_rate"],
                strong=assessment["strong_memory_detection_rate"],
                passed="PASS" if assessment["route_pass"] else "FAIL",
            )
        )
    minimum = result["minimal_passing_length"]
    lines.extend(
        [
            "",
            f"- Overall route: {'PASS' if result['route_pass'] else 'FAIL'}",
            f"- Minimal passing prefix length: {minimum if minimum is not None else 'none'}",
            "- The experiment estimates the mean but still assumes a known iid noise scale.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = run_estimated_conditional_map()
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "estimated_conditional_spectral_feasibility.json"
    markdown_path = RESULTS / "estimated_conditional_spectral_feasibility.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "route_pass": result["route_pass"],
                "minimal_passing_length": result["minimal_passing_length"],
                "elapsed_seconds": result["elapsed_seconds"],
                "json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
