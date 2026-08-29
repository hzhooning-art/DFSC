"""Test a frozen noise-scale correction and optimizer-adequacy gate.

Stage 45 coefficients remain the lower bound of the validation contract.  A
disjoint exact-sharing bank calibrates an observable noise-scale correction and
a four-start fitting-adequacy threshold.  Fresh transfer records are never used
for calibration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from probe_decomposed_tolerance_transfer import (
    CHANNEL_COUNTS,
    HETEROGENEITY_CONSTRUCTIONS,
    LOG_SPECTRAL_DRIFTS,
    MODEL_ALLOWANCE,
    NOISE_CORRELATION,
    NOISE_ENVELOPE,
    PROXY_SCOPE,
    build_transfer_observation,
    frozen_total_tolerance,
)
from probe_high_dimensional_shared_spectrum import DEVICE, DTYPE, fit_candidate
from probe_nested_group_sharing_gate import fit_grouped_candidate
from probe_noise_aware_sharing_gate import (
    classify_with_limit,
    second_difference_correlation_proxy,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CALIBRATION_CHANNELS = 64
CALIBRATION_NOISE_STDS = (3.0e-4, 7.0e-4, 1.2e-3, 1.8e-3)
CALIBRATION_REPEATS = 4
PROJECT_NOISE_STDS = (4.0e-4, 1.6e-3)
PROJECT_REPEATS = 3
SHARED_STARTS = 4
GROUPED_STARTS = 2
CALIBRATION_COVERAGE = 0.90
CALIBRATION_SEED_BASE = 71000
PROJECT_SEED_BASE = 81000


def mixed_difference_noise_scale(observations: torch.Tensor) -> float:
    """Robustly estimate idiosyncratic noise after suppressing smooth signal.

    A temporal second difference followed by a channel second difference has
    standard deviation 6 sigma for iid noise.  Channel differencing also removes
    a common-mode noise component.  The median absolute deviation limits the
    influence of block boundaries in heterogeneous constructions.
    """
    temporal = observations[2:] - 2.0 * observations[1:-1] + observations[:-2]
    mixed = temporal[:, 2:] - 2.0 * temporal[:, 1:-1] + temporal[:, :-2]
    centre = torch.median(mixed)
    mad = torch.median(torch.abs(mixed - centre))
    return float((mad / (0.6744897501960817 * 6.0)).detach().cpu())


def shared_candidates(times, observations, train_idx, val_idx, seed: int):
    return [
        fit_candidate(
            times,
            observations,
            train_idx,
            val_idx,
            2,
            True,
            seed * 10 + start,
        )
        for start in range(SHARED_STARTS)
    ]


def fit_calibration(records: list[dict]) -> dict:
    """Freeze a one-sided scale correction and a two-start adequacy threshold."""
    x = np.asarray([record["noise_scale_proxy"] for record in records], dtype=float)
    excess = np.asarray(
        [
            record["consensus_val_rmse"] - record["stage45_total_tolerance"]
            for record in records
        ],
        dtype=float,
    )
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, excess, rcond=None)
    slope = max(0.0, float(coefficients[1]))
    intercept = float(np.mean(excess) - slope * np.mean(x))
    residuals = excess - (intercept + slope * x)
    order = min(
        len(residuals) - 1,
        math.ceil((len(residuals) + 1) * CALIBRATION_COVERAGE) - 1,
    )
    residual_allowance = float(np.sort(residuals)[order])
    second_start_ratios = np.asarray(
        [record["second_best_train_to_noise_ratio"] for record in records], dtype=float
    )
    adequacy_order = min(
        len(second_start_ratios) - 1,
        math.ceil((len(second_start_ratios) + 1) * CALIBRATION_COVERAGE) - 1,
    )
    grouped = {}
    for record in records:
        grouped.setdefault(record["noise_std_diagnostic"], []).append(
            record["noise_scale_proxy"]
        )
    padding = max(
        (
            float(np.std(values, ddof=1))
            for values in grouped.values()
            if len(values) > 1
        ),
        default=0.0,
    )
    return {
        "scale_correction_intercept": intercept,
        "scale_correction_slope": slope,
        "scale_correction_one_sided_residual": residual_allowance,
        "optimizer_train_to_noise_threshold": float(
            np.sort(second_start_ratios)[adequacy_order]
        ),
        "coverage_target": CALIBRATION_COVERAGE,
        "noise_proxy_min": max(0.0, float(x.min()) - padding),
        "noise_proxy_max": float(x.max()) + padding,
        "noise_proxy_padding": padding,
    }


def scale_correction(noise_scale_proxy: float, calibration: dict) -> float:
    return max(
        0.0,
        calibration["scale_correction_intercept"]
        + calibration["scale_correction_slope"] * noise_scale_proxy
        + calibration["scale_correction_one_sided_residual"],
    )


def select_consensus_candidate(candidates, noise_scale: float, threshold: float):
    ratios = np.asarray(
        [candidate.train_rmse / max(noise_scale, 1.0e-12) for candidate in candidates]
    )
    adequate = [
        candidate
        for candidate, ratio in zip(candidates, ratios)
        if ratio <= threshold
    ]
    if len(adequate) < 2:
        return None, ratios, len(adequate)
    log_rates = np.asarray([np.log(candidate.rates) for candidate in adequate])
    centre = np.median(log_rates, axis=0)
    distances = np.linalg.norm(log_rates - centre[None, :], axis=1)
    return adequate[int(np.argmin(distances))], ratios, len(adequate)


def calibration_record(noise_std: float, repeat: int) -> dict:
    construction = HETEROGENEITY_CONSTRUCTIONS[repeat % len(HETEROGENEITY_CONSTRUCTIONS)]
    seed = CALIBRATION_SEED_BASE + int(noise_std * 1.0e6) + repeat
    times, observations, train_idx, val_idx, _, _ = build_transfer_observation(
        CALIBRATION_CHANNELS, noise_std, construction, 0.0, seed
    )
    candidates = shared_candidates(times, observations, train_idx, val_idx, seed)
    noise_proxy = mixed_difference_noise_scale(observations)
    ordered = sorted(candidates, key=lambda candidate: candidate.train_rmse)
    log_rates = np.asarray([np.log(candidate.rates) for candidate in candidates])
    centre = np.median(log_rates, axis=0)
    consensus = candidates[int(np.argmin(np.linalg.norm(log_rates - centre[None, :], axis=1)))]
    correlation_proxy = second_difference_correlation_proxy(observations)
    return {
        "channels": CALIBRATION_CHANNELS,
        "noise_std_diagnostic": noise_std,
        "heterogeneity_construction_diagnostic": construction,
        "repeat": repeat,
        "seed": seed,
        "noise_scale_proxy": noise_proxy,
        "correlation_proxy": correlation_proxy,
        "stage45_total_tolerance": frozen_total_tolerance(correlation_proxy),
        "consensus_train_rmse": consensus.train_rmse,
        "consensus_val_rmse": consensus.val_rmse,
        "second_best_train_to_noise_ratio": ordered[1].train_rmse
        / max(noise_proxy, 1.0e-12),
        "all_start_train_rmse": [candidate.train_rmse for candidate in candidates],
        "all_start_val_rmse": [candidate.val_rmse for candidate in candidates],
        "all_start_rates": [candidate.rates for candidate in candidates],
    }


def project_record(
    channels: int,
    noise_std: float,
    construction: str,
    drift: float,
    repeat: int,
    calibration: dict,
) -> dict:
    construction_code = HETEROGENEITY_CONSTRUCTIONS.index(construction)
    seed = (
        PROJECT_SEED_BASE
        + channels * 10
        + int(noise_std * 1.0e6)
        + construction_code * 1000
        + int(drift * 1000)
        + repeat
    )
    times, observations, train_idx, val_idx, _, block_labels = build_transfer_observation(
        channels, noise_std, construction, drift, seed
    )
    candidates = shared_candidates(times, observations, train_idx, val_idx, seed)
    noise_proxy = mixed_difference_noise_scale(observations)
    shared, start_ratios, adequate_starts = select_consensus_candidate(
        candidates,
        noise_proxy,
        calibration["optimizer_train_to_noise_threshold"],
    )
    correlation_proxy = second_difference_correlation_proxy(observations)
    base_tolerance = frozen_total_tolerance(correlation_proxy)
    correction = scale_correction(noise_proxy, calibration)
    total_tolerance = base_tolerance + correction
    in_scope = (
        calibration["noise_proxy_min"]
        <= noise_proxy
        <= calibration["noise_proxy_max"]
        and PROXY_SCOPE[0] <= correlation_proxy <= PROXY_SCOPE[1]
    )

    grouped = min(
        [
            fit_grouped_candidate(
                times,
                observations,
                train_idx,
                val_idx,
                torch.tensor(block_labels, dtype=torch.long, device=DEVICE),
                seed * 10 + 20 + start,
            )
            for start in range(GROUPED_STARTS)
        ],
        key=lambda candidate: candidate.bic,
    )
    if shared is None:
        decision = "INDETERMINATE_OPTIMIZATION"
        shared_bic = None
        shared_train_rmse = None
        shared_val_rmse = None
        support = None
    else:
        support = shared.bic - grouped.bic
        decision = classify_with_limit(
            support, shared.val_rmse, grouped.val_rmse, total_tolerance
        )
        shared_bic = shared.bic
        shared_train_rmse = shared.train_rmse
        shared_val_rmse = shared.val_rmse
    return {
        "channels": channels,
        "noise_std_diagnostic": noise_std,
        "heterogeneity_construction": construction,
        "log_spectral_drift": drift,
        "noise_correlation_diagnostic": NOISE_CORRELATION,
        "repeat": repeat,
        "seed": seed,
        "noise_scale_proxy": noise_proxy,
        "correlation_proxy": correlation_proxy,
        "diagnostics_in_calibration_scope": in_scope,
        "adequate_shared_starts": adequate_starts,
        "shared_start_train_to_noise_ratios": start_ratios.tolist(),
        "stage45_total_tolerance": base_tolerance,
        "noise_scale_correction": correction,
        "augmented_total_tolerance": total_tolerance,
        "decision": decision,
        "group_bic_support": support,
        "shared_bic": shared_bic,
        "shared_train_rmse": shared_train_rmse,
        "shared_val_rmse": shared_val_rmse,
        "grouped_val_rmse": grouped.val_rmse,
    }


def summarize(records: list[dict]) -> dict:
    rows = []
    for channels in CHANNEL_COUNTS:
        for noise_std in PROJECT_NOISE_STDS:
            for construction in HETEROGENEITY_CONSTRUCTIONS:
                for drift in LOG_SPECTRAL_DRIFTS:
                    group = [
                        record
                        for record in records
                        if record["channels"] == channels
                        and record["noise_std_diagnostic"] == noise_std
                        and record["heterogeneity_construction"] == construction
                        and record["log_spectral_drift"] == drift
                    ]
                    refused = np.mean(
                        [r["decision"] == "REFUSE_SHARED_MECHANISM" for r in group]
                    )
                    indeterminate = np.mean(
                        [r["decision"] == "INDETERMINATE_OPTIMIZATION" for r in group]
                    )
                    rows.append(
                        {
                            "channels": channels,
                            "noise_std": noise_std,
                            "heterogeneity_construction": construction,
                            "log_spectral_drift": drift,
                            "trials": len(group),
                            "refuse_fraction": float(refused),
                            "indeterminate_fraction": float(indeterminate),
                            "adverse_fraction": float(refused + indeterminate),
                            "median_noise_scale_proxy": float(
                                np.median([r["noise_scale_proxy"] for r in group])
                            ),
                            "median_augmented_tolerance": float(
                                np.median([r["augmented_total_tolerance"] for r in group])
                            ),
                            "median_adequate_shared_starts": float(
                                np.median([r["adequate_shared_starts"] for r in group])
                            ),
                            "in_scope_fraction": float(
                                np.mean([r["diagnostics_in_calibration_scope"] for r in group])
                            ),
                        }
                    )
    exact_mild_retained = all(
        row["adverse_fraction"] <= 1.0 / 3.0
        for row in rows
        if row["log_spectral_drift"] in (0.0, 0.05)
    )
    severe_refused = all(
        row["refuse_fraction"] >= 2.0 / 3.0
        for row in rows
        if row["log_spectral_drift"] == 0.15
    )
    expected = (
        len(CHANNEL_COUNTS)
        * len(PROJECT_NOISE_STDS)
        * len(HETEROGENEITY_CONSTRUCTIONS)
        * len(LOG_SPECTRAL_DRIFTS)
        * PROJECT_REPEATS
    )
    checks = {
        "complete_project_matrix": len(records) == expected,
        "exact_and_mild_adverse_fraction_at_most_one_third": exact_mild_retained,
        "severe_refuse_fraction_at_least_two_thirds": severe_refused,
        "all_diagnostics_in_calibration_scope": all(
            record["diagnostics_in_calibration_scope"] for record in records
        ),
    }
    return {
        "rows": rows,
        "checks": checks,
        "route_pass": bool(all(checks.values())),
        "frozen_rule": {
            "minimum_adequate_shared_starts": 2,
            "mild_adverse_fraction_max": 1.0 / 3.0,
            "severe_refuse_fraction_min": 2.0 / 3.0,
            "indeterminate_counts_as_adverse_for_exact_and_mild": True,
            "indeterminate_does_not_count_as_refusal_for_severe": True,
            "out_of_diagnostic_scope_action": "route failure",
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "noise_scale_optimizer_transfer.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Noise-scale and optimizer-adequacy transfer audit",
        "",
        f"Device: `{payload['device']}`; route pass: **{payload['summary']['route_pass']}**.",
        "",
        "| Channels | Noise std | Construction | Drift | Refused | Indeterminate | Adverse | Noise proxy | Limit | Adequate starts | In scope |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["rows"]:
        lines.append(
            f"| {row['channels']} | {row['noise_std']:.1e} | {row['heterogeneity_construction']} | "
            f"{row['log_spectral_drift']:.2f} | {row['refuse_fraction']:.2f} | "
            f"{row['indeterminate_fraction']:.2f} | {row['adverse_fraction']:.2f} | "
            f"{row['median_noise_scale_proxy']:.3g} | {row['median_augmented_tolerance']:.4g} | "
            f"{row['median_adequate_shared_starts']:.1f} | {row['in_scope_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Stage 45 coefficients are retained as a floor. Calibration and project seeds are disjoint; no project record changes a coefficient or threshold.",
        ]
    )
    (RESULTS / "noise_scale_optimizer_transfer.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    calibration_records = []
    for noise_std in CALIBRATION_NOISE_STDS:
        for repeat in range(CALIBRATION_REPEATS):
            record = calibration_record(noise_std, repeat)
            calibration_records.append(record)
            print(
                f"calibration noise={noise_std:.1e} repeat={repeat} "
                f"proxy={record['noise_scale_proxy']:.4g} "
                f"val={record['consensus_val_rmse']:.4g}",
                flush=True,
            )
    calibration = fit_calibration(calibration_records)
    print(f"frozen calibration={calibration}", flush=True)

    project_records = []
    for channels in CHANNEL_COUNTS:
        for noise_std in PROJECT_NOISE_STDS:
            for construction in HETEROGENEITY_CONSTRUCTIONS:
                for drift in LOG_SPECTRAL_DRIFTS:
                    for repeat in range(PROJECT_REPEATS):
                        record = project_record(
                            channels,
                            noise_std,
                            construction,
                            drift,
                            repeat,
                            calibration,
                        )
                        project_records.append(record)
                        print(
                            f"project channels={channels} noise={noise_std:.1e} "
                            f"construction={construction} drift={drift:.2f} repeat={repeat} "
                            f"decision={record['decision']} adequate={record['adequate_shared_starts']} "
                            f"scale={record['noise_scale_proxy']:.4g}",
                            flush=True,
                        )
    summary = summarize(project_records)
    payload = {
        "experiment": "noise_scale_optimizer_transfer",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "protocol": {
            "calibration_channels": CALIBRATION_CHANNELS,
            "calibration_noise_stds": list(CALIBRATION_NOISE_STDS),
            "calibration_repeats": CALIBRATION_REPEATS,
            "project_channel_counts": list(CHANNEL_COUNTS),
            "project_noise_stds": list(PROJECT_NOISE_STDS),
            "heterogeneity_constructions": list(HETEROGENEITY_CONSTRUCTIONS),
            "log_spectral_drifts": list(LOG_SPECTRAL_DRIFTS),
            "project_repeats": PROJECT_REPEATS,
            "shared_starts": SHARED_STARTS,
            "grouped_starts": GROUPED_STARTS,
            "calibration_seed_base": CALIBRATION_SEED_BASE,
            "project_seed_base": PROJECT_SEED_BASE,
        },
        "retained_stage45_budget": {
            "noise_envelope": NOISE_ENVELOPE,
            "model_allowance": MODEL_ALLOWANCE,
            "correlation_proxy_scope": list(PROXY_SCOPE),
        },
        "calibration_records": calibration_records,
        "frozen_stage47_calibration": calibration,
        "project_records": project_records,
        "summary": summary,
    }
    write_outputs(payload)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
