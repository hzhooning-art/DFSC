"""Frequency-domain feasibility map for long-memory mismatch detection."""

from __future__ import annotations

import json
import time

import numpy as np

from probe_controlled_misspecification import RATES, simulate_channel
from probe_correlated_noise_gate import BASE_NOISE, STRENGTHS
from probe_long_memory_mismatch_gate import fractional_noise
from probe_mechanism_vs_trajectory_baselines import (
    CHANNELS,
    HORIZON,
    RESULTS,
    TRAIN_END_FRACTION,
)


NULL_QUANTILE = 0.99
CALIBRATION_DRAWS = 128
PREFIX_LENGTHS = (78, 256, 512)
MEMORY_ORDERS = (0.0, 0.15, 0.30, 0.45)
REPEATS = 2
MIN_CONTROL_ADEQUACY_RATE = 0.75
MIN_STRONG_MEMORY_DETECTION_RATE = 0.75


def local_whittle_d(values: np.ndarray, bandwidth: int) -> float:
    """Estimate an ARFIMA memory order from the lowest Fourier frequencies."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim == 1:
        sample = sample[:, None]
    length = sample.shape[0]
    if not 4 <= bandwidth < length // 2:
        raise ValueError("bandwidth must use at least four non-Nyquist frequencies")
    centered = sample - np.mean(sample, axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    periodogram = np.mean(np.abs(spectrum[1 : bandwidth + 1]) ** 2, axis=1)
    periodogram = np.maximum(periodogram, np.finfo(float).tiny)
    frequencies = 2.0 * np.pi * np.arange(1, bandwidth + 1) / length
    log_frequencies = np.log(frequencies)
    grid = np.linspace(0.0, 0.49, 491)
    objectives = np.empty_like(grid)
    for index, d in enumerate(grid):
        scaled_mean = np.mean(frequencies ** (2.0 * d) * periodogram)
        objectives[index] = np.log(scaled_mean) - 2.0 * d * np.mean(
            log_frequencies
        )
    return float(grid[int(np.argmin(objectives))])


def _detrend(values: np.ndarray, degree: int) -> np.ndarray:
    sample = np.asarray(values, dtype=float)
    if sample.ndim == 1:
        sample = sample[:, None]
    basis = np.polynomial.chebyshev.chebvander(
        np.linspace(-1.0, 1.0, sample.shape[0]), degree
    )
    coefficients = np.linalg.lstsq(basis, sample, rcond=None)[0]
    return sample - basis @ coefficients


def _bandwidths(length: int) -> tuple[int, ...]:
    candidates = [
        int(np.floor(length**0.55)),
        int(np.floor(length**0.65)),
        int(np.floor(length**0.75)),
    ]
    upper = max(4, length // 4)
    return tuple(sorted({min(max(value, 4), upper) for value in candidates}))


def spectral_memory_statistic(values: np.ndarray) -> float:
    """Return a sensitivity-median local-Whittle memory estimate."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim == 1:
        sample = sample[:, None]
    estimates = []
    for degree in (8, 9, 10):
        residual = _detrend(sample, degree)
        for bandwidth in _bandwidths(sample.shape[0]):
            estimates.append(local_whittle_d(residual, bandwidth))
    return float(np.median(estimates))


def calibrated_null_threshold(
    length: int,
    channels: int,
    draws: int,
    seed: int,
    quantile: float = NULL_QUANTILE,
) -> float:
    """Calibrate the complete spectral statistic under an iid Gaussian null."""
    rng = np.random.default_rng(seed)
    statistics = [
        spectral_memory_statistic(rng.normal(size=(length, channels)))
        for _ in range(draws)
    ]
    return float(np.quantile(statistics, quantile, method="higher"))


def make_prefix_case(
    length: int, d: float, strength: float, repeat: int
) -> dict:
    """Create a deterministic observed prefix with controlled memory order."""
    if length not in PREFIX_LENGTHS:
        raise ValueError(f"unsupported prefix length: {length}")
    seed = (
        1137000
        + 100003 * PREFIX_LENGTHS.index(length)
        + 10007 * MEMORY_ORDERS.index(d)
        + 1009 * STRENGTHS.index(strength)
        + 101 * repeat
    )
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, HORIZON * TRAIN_END_FRACTION, length)
    channel_scale = np.linspace(0.28, 0.82, CHANNELS)[:, None]
    pole_scale = np.linspace(0.78, 1.22, len(RATES))[None, :]
    weights = channel_scale * pole_scale / len(RATES)
    clean = np.column_stack(
        [
            simulate_channel(
                times, weights[channel], "nonlinear_feedback", strength
            )
            for channel in range(CHANNELS)
        ]
    )
    observations = clean + fractional_noise(
        rng, length, CHANNELS, marginal_scale=BASE_NOISE, d=d
    )
    return {
        "seed": seed,
        "times": times,
        "clean": clean,
        "observations": observations,
    }


def assess_length(records: list[dict], expected_count: int) -> dict:
    """Apply the frozen control and strong-memory acceptance criteria."""
    controls = [record for record in records if record["d"] == 0.0]
    strong = [record for record in records if record["d"] >= 0.30]
    complete = len(records) == expected_count
    control_adequacy = (
        sum(not record["mismatch_detected"] for record in controls)
        / len(controls)
        if controls
        else 0.0
    )
    strong_detection = (
        sum(record["mismatch_detected"] for record in strong) / len(strong)
        if strong
        else 0.0
    )
    checks = {
        "matrix_complete": complete,
        "control_adequacy": control_adequacy
        >= MIN_CONTROL_ADEQUACY_RATE,
        "strong_memory_detection": strong_detection
        >= MIN_STRONG_MEMORY_DETECTION_RATE,
    }
    return {
        "record_count": len(records),
        "expected_count": expected_count,
        "control_count": len(controls),
        "strong_memory_count": len(strong),
        "control_adequacy_rate": control_adequacy,
        "strong_memory_detection_rate": strong_detection,
        "checks": checks,
        "route_pass": all(checks.values()),
    }


def run_feasibility_map() -> dict:
    """Run the frozen 78/256/512 frequency-domain feasibility matrix."""
    started = time.perf_counter()
    records = []
    by_length = {}
    expected_per_length = (
        len(MEMORY_ORDERS) * len(STRENGTHS) * REPEATS
    )
    for length in PREFIX_LENGTHS:
        threshold_seed = 1600000 + length
        threshold = calibrated_null_threshold(
            length,
            CHANNELS,
            CALIBRATION_DRAWS,
            threshold_seed,
            NULL_QUANTILE,
        )
        length_records = []
        for d in MEMORY_ORDERS:
            for strength in STRENGTHS:
                for repeat in range(REPEATS):
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
                        "margin": statistic - threshold,
                        "mismatch_detected": statistic > threshold,
                    }
                    records.append(record)
                    length_records.append(record)
        by_length[str(length)] = assess_length(
            length_records, expected_per_length
        )
        by_length[str(length)]["threshold"] = threshold

    passing_lengths = [
        length
        for length in PREFIX_LENGTHS
        if by_length[str(length)]["route_pass"]
    ]
    return {
        "experiment": "spectral_long_memory_feasibility",
        "protocol": {
            "prefix_lengths": list(PREFIX_LENGTHS),
            "memory_orders": list(MEMORY_ORDERS),
            "strengths": list(STRENGTHS),
            "repeats": REPEATS,
            "calibration_draws": CALIBRATION_DRAWS,
            "null_quantile": NULL_QUANTILE,
            "minimum_control_adequacy_rate": MIN_CONTROL_ADEQUACY_RATE,
            "minimum_strong_memory_detection_rate": MIN_STRONG_MEMORY_DETECTION_RATE,
            "decision_frozen_before_project_run": True,
        },
        "records": records,
        "assessment_by_length": by_length,
        "minimal_passing_length": min(passing_lengths)
        if passing_lengths
        else None,
        "route_pass": bool(passing_lengths),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _markdown_report(result: dict) -> str:
    lines = [
        "# Spectral long-memory feasibility map",
        "",
        "The acceptance thresholds and prefix-length matrix were frozen before the project-level run.",
        "",
        "| Prefix length | Null threshold | Control adequacy | Strong-memory detection | Route pass |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for length in PREFIX_LENGTHS:
        assessment = result["assessment_by_length"][str(length)]
        lines.append(
            "| {length} | {threshold:.4f} | {control:.3f} | {strong:.3f} | {passed} |".format(
                length=length,
                threshold=assessment["threshold"],
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
            "- A failed length is not eligible for a later mechanism-fit experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = run_feasibility_map()
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "spectral_long_memory_feasibility.json"
    markdown_path = RESULTS / "spectral_long_memory_feasibility.md"
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
