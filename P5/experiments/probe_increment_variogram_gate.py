"""Increment-domain feasibility map for long-memory mismatch detection."""

from __future__ import annotations

import json
import time

import numpy as np

from probe_correlated_noise_gate import BASE_NOISE
from probe_estimated_conditional_spectral_gate import (
    ELIGIBLE_PREFIX_LENGTHS,
    estimate_smooth_mean,
    make_independent_control,
)
from probe_mechanism_vs_trajectory_baselines import RESULTS
from probe_spectral_long_memory_gate import (
    CALIBRATION_DRAWS,
    MEMORY_ORDERS,
    NULL_QUANTILE,
    REPEATS,
    STRENGTHS,
    assess_length,
    make_prefix_case,
)


INCREMENT_LAGS = (4, 8, 16)


def _second_difference_variance(values: np.ndarray, lag: int) -> np.ndarray:
    sample = np.asarray(values, dtype=float)
    if sample.ndim == 1:
        sample = sample[:, None]
    if lag < 1 or 2 * lag >= sample.shape[0]:
        raise ValueError("lag must leave at least one second difference")
    differences = sample[2 * lag :] - 2.0 * sample[lag:-lag] + sample[: -2 * lag]
    return np.var(differences, axis=0, ddof=1)


def increment_memory_statistic(values: np.ndarray) -> float:
    """Return a robust multiscale second-difference variance slope."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim == 1:
        sample = sample[:, None]
    baseline = _second_difference_variance(sample, 1)
    floor = np.finfo(float).tiny
    slopes = []
    for lag in INCREMENT_LAGS:
        variance = _second_difference_variance(sample, lag)
        slopes.extend(
            (np.log(np.maximum(variance, floor) / np.maximum(baseline, floor))
             / np.log(float(lag))).tolist()
        )
    return float(np.median(slopes))


def increment_null_threshold(
    calibration_observations: np.ndarray,
    draws: int,
    seed: int,
    quantile: float = NULL_QUANTILE,
) -> float:
    """Calibrate the increment statistic while propagating mean uncertainty."""
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
        statistics.append(increment_memory_statistic(pseudo_evaluation))
    return float(np.quantile(statistics, quantile, method="higher"))


def run_increment_feasibility_map(
    prefix_lengths: tuple[int, ...] = ELIGIBLE_PREFIX_LENGTHS,
    memory_orders: tuple[float, ...] = MEMORY_ORDERS,
    strengths: tuple[float, ...] = STRENGTHS,
    repeats: int = REPEATS,
    calibration_draws: int = CALIBRATION_DRAWS,
) -> dict:
    """Run the frozen increment-domain feasibility matrix."""
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
                2700000
                + 100003 * ELIGIBLE_PREFIX_LENGTHS.index(length)
                + 1009 * STRENGTHS.index(strength)
            )
            threshold = increment_null_threshold(
                control["observations"],
                draws=calibration_draws,
                seed=threshold_seed,
                quantile=NULL_QUANTILE,
            )
            thresholds[str(strength)] = {
                "threshold": threshold,
                "seed": threshold_seed,
                "control_seed": control["seed"],
            }
            for d in memory_orders:
                for repeat in range(repeats):
                    case = make_prefix_case(length, d, strength, repeat)
                    statistic = increment_memory_statistic(case["observations"])
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
        length for length in prefix_lengths if assessments[str(length)]["route_pass"]
    ]
    return {
        "experiment": "increment_variogram_long_memory_feasibility",
        "protocol": {
            "prefix_lengths": list(prefix_lengths),
            "memory_orders": list(memory_orders),
            "strengths": list(strengths),
            "repeats": repeats,
            "increment_lags": list(INCREMENT_LAGS),
            "calibration_draws": calibration_draws,
            "null_quantile": NULL_QUANTILE,
            "linear_trend_annihilation": True,
            "independent_control_trajectory": True,
            "mean_refit_in_bootstrap": True,
            "strength_conditional_threshold": True,
            "known_noise_scale": BASE_NOISE,
            "decision_frozen_before_project_run": True,
        },
        "records": records,
        "assessment_by_length": assessments,
        "minimal_passing_length": min(passing_lengths) if passing_lengths else None,
        "route_pass": bool(passing_lengths),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _markdown_report(result: dict) -> str:
    lines = [
        "# Increment-domain long-memory feasibility map",
        "",
        "The statistic uses frozen multiscale second-difference variance slopes and an independent conditional null.",
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
            "- The feasibility probe still assumes a known iid noise scale and a strength-matched independent calibration trajectory.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = run_increment_feasibility_map()
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "increment_variogram_feasibility.json"
    markdown_path = RESULTS / "increment_variogram_feasibility.md"
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
