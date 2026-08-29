"""Leave-one-strength-out transfer test for the increment-domain gate."""

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
from probe_increment_variogram_gate import increment_memory_statistic
from probe_mechanism_vs_trajectory_baselines import RESULTS
from probe_spectral_long_memory_gate import (
    CALIBRATION_DRAWS,
    MEMORY_ORDERS,
    NULL_QUANTILE,
    STRENGTHS,
    assess_length,
    make_prefix_case,
)


TRANSFER_REPEATS = 4
MIN_TRANSFER_RATE = 0.75


def _conditional_statistics(
    calibration_observations: np.ndarray,
    draws: int,
    seed: int,
) -> list[float]:
    fitted_mean = estimate_smooth_mean(calibration_observations)
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
    return statistics


def leave_one_strength_out_threshold(
    length: int,
    held_out_strength: float,
    draws_per_donor: int,
    seed: int,
    strengths: tuple[float, ...] = STRENGTHS,
    quantile: float = NULL_QUANTILE,
) -> dict:
    """Pool null draws from every declared strength except the evaluated one."""
    donor_strengths = [value for value in strengths if value != held_out_strength]
    if len(donor_strengths) < 1:
        raise ValueError("leave-one-strength-out calibration needs at least one donor")
    pooled_statistics = []
    donor_seeds = []
    for donor_index, donor_strength in enumerate(donor_strengths):
        control = make_independent_control(length, donor_strength)
        donor_seed = seed + 1009 * donor_index
        donor_seeds.append(donor_seed)
        pooled_statistics.extend(
            _conditional_statistics(
                control["observations"], draws=draws_per_donor, seed=donor_seed
            )
        )
    return {
        "threshold": float(
            np.quantile(pooled_statistics, quantile, method="higher")
        ),
        "held_out_strength": held_out_strength,
        "donor_strengths": donor_strengths,
        "donor_seeds": donor_seeds,
        "null_sample_count": len(pooled_statistics),
    }


def _assess_strength(records: list[dict], expected_count: int) -> dict:
    controls = [record for record in records if record["d"] == 0.0]
    strong = [record for record in records if record["d"] >= 0.30]
    control_rate = (
        sum(not record["mismatch_detected"] for record in controls) / len(controls)
        if controls
        else 0.0
    )
    strong_rate = (
        sum(record["mismatch_detected"] for record in strong) / len(strong)
        if strong
        else 0.0
    )
    checks = {
        "matrix_complete": len(records) == expected_count,
        "control_adequacy": control_rate >= MIN_TRANSFER_RATE,
        "strong_memory_detection": strong_rate >= MIN_TRANSFER_RATE,
    }
    return {
        "record_count": len(records),
        "expected_count": expected_count,
        "control_count": len(controls),
        "strong_memory_count": len(strong),
        "control_adequacy_rate": control_rate,
        "strong_memory_detection_rate": strong_rate,
        "checks": checks,
        "route_pass": all(checks.values()),
    }


def run_increment_transfer_map(
    prefix_lengths: tuple[int, ...] = ELIGIBLE_PREFIX_LENGTHS,
    memory_orders: tuple[float, ...] = MEMORY_ORDERS,
    strengths: tuple[float, ...] = STRENGTHS,
    repeats: int = TRANSFER_REPEATS,
    calibration_draws: int = CALIBRATION_DRAWS,
) -> dict:
    """Run the frozen leave-one-strength-out transfer matrix."""
    started = time.perf_counter()
    records = []
    assessments = {}
    expected_per_strength = len(memory_orders) * repeats
    expected_per_length = expected_per_strength * len(strengths)
    for length in prefix_lengths:
        length_records = []
        by_strength = {}
        thresholds = {}
        for held_out_index, held_out_strength in enumerate(strengths):
            threshold_seed = (
                3100000
                + 100003 * ELIGIBLE_PREFIX_LENGTHS.index(length)
                + 1009 * held_out_index
            )
            threshold_info = leave_one_strength_out_threshold(
                length,
                held_out_strength,
                draws_per_donor=calibration_draws,
                seed=threshold_seed,
                strengths=strengths,
            )
            thresholds[str(held_out_strength)] = threshold_info
            strength_records = []
            for d in memory_orders:
                for repeat in range(repeats):
                    case = make_prefix_case(length, d, held_out_strength, repeat)
                    statistic = increment_memory_statistic(case["observations"])
                    record = {
                        "length": length,
                        "d": d,
                        "strength": held_out_strength,
                        "repeat": repeat,
                        "seed": case["seed"],
                        "statistic": statistic,
                        "threshold": threshold_info["threshold"],
                        "threshold_seed": threshold_seed,
                        "donor_strengths": threshold_info["donor_strengths"],
                        "margin": statistic - threshold_info["threshold"],
                        "mismatch_detected": statistic > threshold_info["threshold"],
                    }
                    records.append(record)
                    length_records.append(record)
                    strength_records.append(record)
            by_strength[str(held_out_strength)] = _assess_strength(
                strength_records, expected_per_strength
            )
        aggregate = assess_length(length_records, expected_per_length)
        aggregate["by_strength"] = by_strength
        aggregate["leave_one_out_thresholds"] = thresholds
        aggregate["all_strengths_pass"] = all(
            item["route_pass"] for item in by_strength.values()
        )
        aggregate["route_pass"] = bool(
            aggregate["route_pass"] and aggregate["all_strengths_pass"]
        )
        assessments[str(length)] = aggregate

    passing_lengths = [
        length for length in prefix_lengths if assessments[str(length)]["route_pass"]
    ]
    return {
        "experiment": "increment_variogram_leave_one_strength_out_transfer",
        "protocol": {
            "prefix_lengths": list(prefix_lengths),
            "memory_orders": list(memory_orders),
            "strengths": list(strengths),
            "repeats": repeats,
            "calibration_draws_per_donor": calibration_draws,
            "null_quantile": NULL_QUANTILE,
            "leave_one_strength_out": True,
            "strength_matched_calibration": False,
            "minimum_per_strength_control_adequacy_rate": MIN_TRANSFER_RATE,
            "minimum_per_strength_strong_detection_rate": MIN_TRANSFER_RATE,
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
        "# Increment-domain leave-one-strength-out transfer map",
        "",
        "Each evaluated strength is calibrated only from independent controls at the other declared strengths.",
        "",
        "| Length | Held-out strength | Control adequacy | Strong detection | Cell pass |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for length in result["protocol"]["prefix_lengths"]:
        assessment = result["assessment_by_length"][str(length)]
        for strength, cell in assessment["by_strength"].items():
            lines.append(
                "| {length} | {strength} | {control:.3f} | {strong:.3f} | {passed} |".format(
                    length=length,
                    strength=strength,
                    control=cell["control_adequacy_rate"],
                    strong=cell["strong_memory_detection_rate"],
                    passed="PASS" if cell["route_pass"] else "FAIL",
                )
            )
    minimum = result["minimal_passing_length"]
    lines.extend(
        [
            "",
            f"- Overall route: {'PASS' if result['route_pass'] else 'FAIL'}",
            f"- Minimal passing prefix length: {minimum if minimum is not None else 'none'}",
            "- Every held-out strength must pass both frozen 75% criteria.",
            "- The experiment still assumes the declared strength set and known iid noise scale.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = run_increment_transfer_map()
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "increment_transfer_feasibility.json"
    markdown_path = RESULTS / "increment_transfer_feasibility.md"
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
