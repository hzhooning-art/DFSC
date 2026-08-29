"""Observable-curvature normalization for the increment-domain memory gate."""

from __future__ import annotations

import json
import time

import numpy as np

from probe_controlled_misspecification import RATES, simulate_channel
from probe_correlated_noise_gate import BASE_NOISE
from probe_estimated_conditional_spectral_gate import (
    ELIGIBLE_PREFIX_LENGTHS,
    estimate_smooth_mean,
)
from probe_increment_transfer_gate import (
    MIN_TRANSFER_RATE,
    TRANSFER_REPEATS,
    _assess_strength,
)
from probe_increment_variogram_gate import (
    _second_difference_variance,
    increment_memory_statistic,
    increment_null_threshold,
)
from probe_mechanism_vs_trajectory_baselines import (
    CHANNELS,
    HORIZON,
    RESULTS,
    TRAIN_END_FRACTION,
)
from probe_spectral_long_memory_gate import (
    CALIBRATION_DRAWS,
    MEMORY_ORDERS,
    NULL_QUANTILE,
    STRENGTHS,
    assess_length,
    make_prefix_case,
)


CALIBRATION_STRENGTHS = (0.035, 0.065, 0.11, 0.16, 0.24)
CURVATURE_LAG = 16


def curvature_proxy(values: np.ndarray) -> float:
    """Estimate dimensionless smooth-curvature leakage from observed values."""
    fitted_mean = estimate_smooth_mean(values)
    curvature_variance = _second_difference_variance(fitted_mean, CURVATURE_LAG)
    ratio = float(np.median(curvature_variance) / (6.0 * BASE_NOISE**2))
    return float(np.log1p(max(ratio, 0.0)))


def _make_calibration_control(length: int, strength: float, index: int) -> dict:
    seed = 3500000 + 100003 * ELIGIBLE_PREFIX_LENGTHS.index(length) + 1009 * index
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, HORIZON * TRAIN_END_FRACTION, length)
    channel_scale = np.linspace(0.28, 0.82, CHANNELS)[:, None]
    pole_scale = np.linspace(0.78, 1.22, len(RATES))[None, :]
    weights = channel_scale * pole_scale / len(RATES)
    clean = np.column_stack(
        [
            simulate_channel(times, weights[channel], "nonlinear_feedback", strength)
            for channel in range(CHANNELS)
        ]
    )
    observations = clean + rng.normal(0.0, BASE_NOISE, size=clean.shape)
    return {"seed": seed, "clean": clean, "observations": observations}


def build_curvature_calibration(
    length: int,
    draws: int = CALIBRATION_DRAWS,
    calibration_strengths: tuple[float, ...] = CALIBRATION_STRENGTHS,
) -> dict:
    """Fit a threshold map using only observable proxies from a disjoint bank."""
    bank = []
    for index, strength in enumerate(calibration_strengths):
        control = _make_calibration_control(length, strength, index)
        threshold_seed = 3700000 + 100003 * ELIGIBLE_PREFIX_LENGTHS.index(length) + 1009 * index
        proxy = curvature_proxy(control["observations"])
        threshold = increment_null_threshold(
            control["observations"],
            draws=draws,
            seed=threshold_seed,
            quantile=NULL_QUANTILE,
        )
        bank.append(
            {
                "calibration_strength": strength,
                "control_seed": control["seed"],
                "threshold_seed": threshold_seed,
                "curvature_proxy": proxy,
                "threshold": threshold,
            }
        )
    proxies = np.asarray([item["curvature_proxy"] for item in bank])
    thresholds = np.asarray([item["threshold"] for item in bank])
    slope, intercept = np.polyfit(proxies, thresholds, deg=1)
    predictions = intercept + slope * proxies
    total = float(np.sum((thresholds - np.mean(thresholds)) ** 2))
    residual = float(np.sum((thresholds - predictions) ** 2))
    return {
        "bank": bank,
        "model": {
            "input": "curvature_proxy",
            "form": "affine",
            "slope": float(slope),
            "intercept": float(intercept),
            "proxy_min": float(np.min(proxies)),
            "proxy_max": float(np.max(proxies)),
            "r_squared": float(1.0 - residual / total) if total > 0.0 else 1.0,
        },
    }


def _threshold_from_proxy(proxy: float, model: dict) -> tuple[float, bool]:
    in_scope = bool(model["proxy_min"] <= proxy <= model["proxy_max"])
    threshold = model["intercept"] + model["slope"] * proxy
    return float(threshold), in_scope


def run_curvature_normalized_map(
    prefix_lengths: tuple[int, ...] = ELIGIBLE_PREFIX_LENGTHS,
    memory_orders: tuple[float, ...] = MEMORY_ORDERS,
    strengths: tuple[float, ...] = STRENGTHS,
    repeats: int = TRANSFER_REPEATS,
    calibration_draws: int = CALIBRATION_DRAWS,
) -> dict:
    """Run the frozen observable-curvature normalized transfer matrix."""
    started = time.perf_counter()
    records = []
    assessments = {}
    calibration_by_length = {}
    expected_per_strength = len(memory_orders) * repeats
    expected_per_length = expected_per_strength * len(strengths)
    for length in prefix_lengths:
        calibration = build_curvature_calibration(length, draws=calibration_draws)
        calibration_by_length[str(length)] = calibration
        model = calibration["model"]
        length_records = []
        by_strength = {}
        for strength in strengths:
            strength_records = []
            for d in memory_orders:
                for repeat in range(repeats):
                    case = make_prefix_case(length, d, strength, repeat)
                    proxy = curvature_proxy(case["observations"])
                    threshold, proxy_in_scope = _threshold_from_proxy(proxy, model)
                    statistic = increment_memory_statistic(case["observations"])
                    mismatch = bool(not proxy_in_scope or statistic > threshold)
                    record = {
                        "length": length,
                        "d": d,
                        "strength": strength,
                        "repeat": repeat,
                        "seed": case["seed"],
                        "curvature_proxy": proxy,
                        "proxy_in_scope": proxy_in_scope,
                        "statistic": statistic,
                        "threshold": threshold,
                        "margin": statistic - threshold,
                        "mismatch_detected": mismatch,
                    }
                    records.append(record)
                    length_records.append(record)
                    strength_records.append(record)
            by_strength[str(strength)] = _assess_strength(
                strength_records, expected_per_strength
            )
        aggregate = assess_length(length_records, expected_per_length)
        aggregate["by_strength"] = by_strength
        aggregate["all_strengths_pass"] = all(
            item["route_pass"] for item in by_strength.values()
        )
        aggregate["proxy_out_of_scope_count"] = sum(
            not item["proxy_in_scope"] for item in length_records
        )
        aggregate["route_pass"] = bool(
            aggregate["route_pass"] and aggregate["all_strengths_pass"]
        )
        assessments[str(length)] = aggregate

    passing_lengths = [
        length for length in prefix_lengths if assessments[str(length)]["route_pass"]
    ]
    return {
        "experiment": "curvature_normalized_increment_memory_feasibility",
        "protocol": {
            "prefix_lengths": list(prefix_lengths),
            "memory_orders": list(memory_orders),
            "project_strengths": list(strengths),
            "calibration_strengths": list(CALIBRATION_STRENGTHS),
            "repeats": repeats,
            "calibration_draws": calibration_draws,
            "null_quantile": NULL_QUANTILE,
            "curvature_lag": CURVATURE_LAG,
            "observable_curvature_normalization": True,
            "project_strength_used_by_gate": False,
            "out_of_proxy_scope_forces_refusal": True,
            "minimum_per_strength_control_adequacy_rate": MIN_TRANSFER_RATE,
            "minimum_per_strength_strong_detection_rate": MIN_TRANSFER_RATE,
            "known_noise_scale": BASE_NOISE,
            "decision_frozen_before_project_run": True,
        },
        "calibration_by_length": calibration_by_length,
        "records": records,
        "assessment_by_length": assessments,
        "minimal_passing_length": min(passing_lengths) if passing_lengths else None,
        "route_pass": bool(passing_lengths),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _markdown_report(result: dict) -> str:
    lines = [
        "# Observable-curvature normalized increment feasibility map",
        "",
        "A disjoint calibration bank maps an observed curvature proxy to the increment-statistic null threshold.",
        "",
        "| Length | Strength | Control adequacy | Strong detection | Cell pass |",
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
            "- Project strength labels are not used by the gate.",
            "- The calibration bank and known iid noise scale remain declared dependencies.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = run_curvature_normalized_map()
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "curvature_normalized_increment_feasibility.json"
    markdown_path = RESULTS / "curvature_normalized_increment_feasibility.md"
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
