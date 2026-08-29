"""Oracle conditional-null probe for spectral long-memory detection."""

from __future__ import annotations

import json
import time

import numpy as np

from probe_correlated_noise_gate import BASE_NOISE
from probe_mechanism_vs_trajectory_baselines import RESULTS
from probe_spectral_long_memory_gate import (
    CALIBRATION_DRAWS,
    MEMORY_ORDERS,
    NULL_QUANTILE,
    PREFIX_LENGTHS,
    REPEATS,
    STRENGTHS,
    assess_length,
    make_prefix_case,
    spectral_memory_statistic,
)


def conditional_null_threshold(
    clean: np.ndarray,
    draws: int,
    seed: int,
    quantile: float,
) -> float:
    """Calibrate the statistic around a declared deterministic trajectory."""
    mean = np.asarray(clean, dtype=float)
    if mean.ndim != 2:
        raise ValueError("clean trajectory must have shape (time, channels)")
    rng = np.random.default_rng(seed)
    statistics = [
        spectral_memory_statistic(
            mean + rng.normal(0.0, BASE_NOISE, size=mean.shape)
        )
        for _ in range(draws)
    ]
    return float(np.quantile(statistics, quantile, method="higher"))


def run_oracle_conditional_map(
    prefix_lengths: tuple[int, ...] = PREFIX_LENGTHS,
    memory_orders: tuple[float, ...] = MEMORY_ORDERS,
    strengths: tuple[float, ...] = STRENGTHS,
    repeats: int = REPEATS,
    calibration_draws: int = CALIBRATION_DRAWS,
) -> dict:
    """Evaluate the frozen matrix using trajectory-conditional nulls."""
    started = time.perf_counter()
    records = []
    assessments = {}
    expected_per_length = len(memory_orders) * len(strengths) * repeats
    for length in prefix_lengths:
        length_records = []
        thresholds = {}
        for strength in strengths:
            clean = make_prefix_case(
                length, d=0.0, strength=strength, repeat=0
            )["clean"]
            threshold_seed = (
                1800000
                + 100003 * PREFIX_LENGTHS.index(length)
                + 1009 * STRENGTHS.index(strength)
            )
            threshold = conditional_null_threshold(
                clean,
                draws=calibration_draws,
                seed=threshold_seed,
                quantile=NULL_QUANTILE,
            )
            thresholds[str(strength)] = {
                "threshold": threshold,
                "seed": threshold_seed,
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
        "experiment": "oracle_conditional_spectral_long_memory_feasibility",
        "protocol": {
            "prefix_lengths": list(prefix_lengths),
            "memory_orders": list(memory_orders),
            "strengths": list(strengths),
            "repeats": repeats,
            "calibration_draws": calibration_draws,
            "null_quantile": NULL_QUANTILE,
            "oracle_clean_trajectory": True,
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
        "# Oracle conditional spectral feasibility map",
        "",
        "This is an upper-bound experiment: each null is conditioned on the known clean deterministic trajectory.",
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
            "- Passing this oracle experiment is necessary but not sufficient for an operational unknown-trend gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = run_oracle_conditional_map()
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "oracle_conditional_spectral_feasibility.json"
    markdown_path = RESULTS / "oracle_conditional_spectral_feasibility.md"
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
