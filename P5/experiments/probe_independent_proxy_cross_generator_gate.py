"""Independent-proxy cross-generator transfer for the increment memory gate."""

from __future__ import annotations

import json
import time

import numpy as np

from probe_controlled_misspecification import RATES, simulate_channel
from probe_correlated_noise_gate import BASE_NOISE
from probe_curvature_normalized_increment_gate import (
    build_curvature_calibration,
    curvature_proxy,
    _threshold_from_proxy,
)
from probe_estimated_conditional_spectral_gate import ELIGIBLE_PREFIX_LENGTHS
from probe_increment_transfer_gate import (
    MIN_TRANSFER_RATE,
    TRANSFER_REPEATS,
    _assess_strength,
)
from probe_increment_variogram_gate import increment_memory_statistic
from probe_long_memory_mismatch_gate import fractional_noise
from probe_mechanism_vs_trajectory_baselines import (
    CHANNELS,
    HORIZON,
    RESULTS,
    TRAIN_END_FRACTION,
)
from probe_spectral_long_memory_gate import (
    CALIBRATION_DRAWS,
    MEMORY_ORDERS,
    assess_length,
)


EXTERNAL_MECHANISM_STRENGTHS = {
    "rate_drift": (1.5, 3.0, 8.0),
    "stretched_exponential": (0.69, 0.70, 0.71),
}


def _external_clean(
    length: int,
    mechanism: str,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0.0, HORIZON * TRAIN_END_FRACTION, length)
    if mechanism == "rate_drift":
        channel_scale = np.linspace(0.28, 0.82, CHANNELS)[:, None]
        pole_scale = np.linspace(0.78, 1.22, len(RATES))[None, :]
        weights = channel_scale * pole_scale / len(RATES)
        clean = np.column_stack(
            [
                simulate_channel(times, weights[channel], mechanism, strength)
                for channel in range(CHANNELS)
            ]
        )
    elif mechanism == "stretched_exponential":
        scales = np.linspace(0.22, 0.65, CHANNELS)
        clean = np.column_stack(
            [np.exp(-(scales[channel] * times) ** strength) for channel in range(CHANNELS)]
        )
    else:
        raise ValueError(f"unsupported external mechanism: {mechanism}")
    return times, clean


def make_external_case(
    length: int,
    mechanism: str,
    strength: float,
    d: float,
    repeat: int,
    role: str,
    mechanism_strengths: dict[str, tuple[float, ...]] = EXTERNAL_MECHANISM_STRENGTHS,
) -> dict:
    """Create an external-family trace with role-separated deterministic seeds."""
    if length not in ELIGIBLE_PREFIX_LENGTHS:
        raise ValueError(f"unsupported prefix length: {length}")
    if mechanism not in mechanism_strengths:
        raise ValueError(f"unsupported external mechanism: {mechanism}")
    if strength not in mechanism_strengths[mechanism]:
        raise ValueError(f"unsupported strength {strength} for {mechanism}")
    if d not in MEMORY_ORDERS:
        raise ValueError(f"unsupported memory order: {d}")
    if role not in {"proxy", "project"}:
        raise ValueError("role must be 'proxy' or 'project'")
    if role == "proxy" and d != 0.0:
        raise ValueError("the independent proxy trace must be an iid control")

    length_index = ELIGIBLE_PREFIX_LENGTHS.index(length)
    mechanism_index = list(mechanism_strengths).index(mechanism)
    strength_index = mechanism_strengths[mechanism].index(strength)
    role_offset = 0 if role == "proxy" else 700000
    seed = (
        4100000
        + role_offset
        + 100003 * length_index
        + 10007 * mechanism_index
        + 1009 * strength_index
        + 101 * repeat
        + 17 * MEMORY_ORDERS.index(d)
    )
    rng = np.random.default_rng(seed)
    times, clean = _external_clean(length, mechanism, strength)
    observations = clean + fractional_noise(
        rng,
        length,
        CHANNELS,
        marginal_scale=BASE_NOISE,
        d=d,
    )
    return {
        "seed": seed,
        "role": role,
        "mechanism": mechanism,
        "strength": strength,
        "d": d,
        "times": times,
        "clean": clean,
        "observations": observations,
    }


def run_external_transfer_map(
    prefix_lengths: tuple[int, ...] = ELIGIBLE_PREFIX_LENGTHS,
    memory_orders: tuple[float, ...] = MEMORY_ORDERS,
    mechanism_strengths: dict[str, tuple[float, ...]] = EXTERNAL_MECHANISM_STRENGTHS,
    repeats: int = TRANSFER_REPEATS,
    calibration_draws: int = CALIBRATION_DRAWS,
) -> dict:
    """Run the frozen independent-proxy cross-generator transfer matrix."""
    started = time.perf_counter()
    records = []
    assessments = {}
    calibration_by_length = {}
    expected_per_cell = len(memory_orders) * repeats
    cells_per_length = sum(len(values) for values in mechanism_strengths.values())
    expected_per_length = expected_per_cell * cells_per_length

    for length in prefix_lengths:
        calibration = build_curvature_calibration(length, draws=calibration_draws)
        calibration_by_length[str(length)] = calibration
        model = calibration["model"]
        length_records = []
        by_mechanism = {}
        for mechanism, strengths in mechanism_strengths.items():
            by_strength = {}
            for strength in strengths:
                cell_records = []
                proxy_by_repeat = {}
                for repeat in range(repeats):
                    proxy_case = make_external_case(
                        length,
                        mechanism,
                        strength,
                        0.0,
                        repeat,
                        role="proxy",
                        mechanism_strengths=mechanism_strengths,
                    )
                    proxy = curvature_proxy(proxy_case["observations"])
                    threshold, proxy_in_scope = _threshold_from_proxy(proxy, model)
                    proxy_by_repeat[repeat] = {
                        "proxy_seed": proxy_case["seed"],
                        "curvature_proxy": proxy,
                        "threshold": threshold,
                        "proxy_in_scope": proxy_in_scope,
                    }
                for d in memory_orders:
                    for repeat in range(repeats):
                        project = make_external_case(
                            length,
                            mechanism,
                            strength,
                            d,
                            repeat,
                            role="project",
                            mechanism_strengths=mechanism_strengths,
                        )
                        proxy_info = proxy_by_repeat[repeat]
                        deterministic_trend_statistic = increment_memory_statistic(
                            project["clean"]
                        )
                        statistic = increment_memory_statistic(project["observations"])
                        mismatch = bool(
                            not proxy_info["proxy_in_scope"]
                            or statistic > proxy_info["threshold"]
                        )
                        record = {
                            "length": length,
                            "mechanism": mechanism,
                            "strength": strength,
                            "d": d,
                            "repeat": repeat,
                            "project_seed": project["seed"],
                            "proxy_seed": proxy_info["proxy_seed"],
                            "curvature_proxy": proxy_info["curvature_proxy"],
                            "proxy_in_scope": proxy_info["proxy_in_scope"],
                            "statistic": statistic,
                            "deterministic_trend_statistic": deterministic_trend_statistic,
                            "deterministic_trend_margin": (
                                deterministic_trend_statistic - proxy_info["threshold"]
                            ),
                            "threshold": proxy_info["threshold"],
                            "margin": statistic - proxy_info["threshold"],
                            "mismatch_detected": mismatch,
                        }
                        records.append(record)
                        length_records.append(record)
                        cell_records.append(record)
                cell = _assess_strength(cell_records, expected_per_cell)
                cell["proxy_out_of_scope_count"] = sum(
                    not item["proxy_in_scope"] for item in cell_records
                )
                cell["all_proxy_repeats_in_scope"] = all(
                    item["proxy_in_scope"] for item in proxy_by_repeat.values()
                )
                cell["route_pass"] = bool(
                    cell["route_pass"] and cell["all_proxy_repeats_in_scope"]
                )
                by_strength[str(strength)] = cell
            by_mechanism[mechanism] = {
                "by_strength": by_strength,
                "all_strengths_pass": all(
                    item["route_pass"] for item in by_strength.values()
                ),
            }

        aggregate = assess_length(length_records, expected_per_length)
        aggregate["by_mechanism"] = by_mechanism
        aggregate["all_external_cells_pass"] = all(
            item["all_strengths_pass"] for item in by_mechanism.values()
        )
        aggregate["proxy_out_of_scope_count"] = sum(
            not item["proxy_in_scope"] for item in length_records
        )
        aggregate["route_pass"] = bool(
            aggregate["route_pass"] and aggregate["all_external_cells_pass"]
        )
        assessments[str(length)] = aggregate

    passing_lengths = [
        length for length in prefix_lengths if assessments[str(length)]["route_pass"]
    ]
    return {
        "experiment": "independent_proxy_cross_generator_increment_transfer",
        "protocol": {
            "prefix_lengths": list(prefix_lengths),
            "memory_orders": list(memory_orders),
            "external_mechanism_strengths": {
                key: list(values) for key, values in mechanism_strengths.items()
            },
            "repeats": repeats,
            "calibration_draws": calibration_draws,
            "independent_proxy_trace": True,
            "proxy_trace_memory_order": 0.0,
            "project_observation_used_for_proxy": False,
            "deterministic_trend_statistic_is_diagnostic_only": True,
            "cross_generator_transfer": True,
            "calibration_generator": "nonlinear_feedback",
            "project_generators": list(mechanism_strengths),
            "project_strength_used_by_gate": False,
            "out_of_proxy_scope_forces_refusal": True,
            "minimum_per_cell_control_adequacy_rate": MIN_TRANSFER_RATE,
            "minimum_per_cell_strong_detection_rate": MIN_TRANSFER_RATE,
            "known_noise_scale": BASE_NOISE,
            "external_strengths_selected_from_d0_proxy_scope_only": True,
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
        "# Independent-proxy cross-generator transfer map",
        "",
        "The threshold uses a role-separated iid proxy trace and a nonlinear-feedback calibration bank.",
        "",
        "| Length | External mechanism | Strength | Control adequacy | Strong detection | Proxy in scope | Cell pass |",
        "|---:|---|---:|---:|---:|:---:|:---:|",
    ]
    for length in result["protocol"]["prefix_lengths"]:
        assessment = result["assessment_by_length"][str(length)]
        for mechanism, mechanism_result in assessment["by_mechanism"].items():
            for strength, cell in mechanism_result["by_strength"].items():
                lines.append(
                    "| {length} | {mechanism} | {strength} | {control:.3f} | {strong:.3f} | {scope} | {passed} |".format(
                        length=length,
                        mechanism=mechanism,
                        strength=strength,
                        control=cell["control_adequacy_rate"],
                        strong=cell["strong_memory_detection_rate"],
                        scope="yes" if cell["all_proxy_repeats_in_scope"] else "no",
                        passed="PASS" if cell["route_pass"] else "FAIL",
                    )
                )
    minimum = result["minimal_passing_length"]
    lines.extend(
        [
            "",
            f"- Overall route: {'PASS' if result['route_pass'] else 'FAIL'}",
            f"- Minimal passing prefix length: {minimum if minimum is not None else 'none'}",
            "- The proxy trace is independent of every evaluated project trace.",
            "- Calibration and project trends come from different generator families.",
            "- The known iid noise scale remains a declared dependency.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = run_external_transfer_map()
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "independent_proxy_cross_generator_feasibility.json"
    markdown_path = RESULTS / "independent_proxy_cross_generator_feasibility.md"
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
