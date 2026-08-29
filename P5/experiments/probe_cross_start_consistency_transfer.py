"""Audit optimizer adequacy through cross-start functional consistency.

Stage 47 showed that an absolute train-residual threshold confounds optimizer
failure with genuine model mismatch.  This experiment keeps the independently
calibrated observable noise correction, but declares a fit adequate only when a
second initialization agrees with the best initialization in both objective
value and predicted function.  Neither diagnostic uses the residual magnitude.
"""

from __future__ import annotations

import json
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
from probe_high_dimensional_shared_spectrum import DEVICE, DTYPE
from probe_memory_rank import lifted_response
from probe_nested_group_sharing_gate import fit_grouped_candidate
from probe_noise_aware_sharing_gate import (
    classify_with_limit,
    second_difference_correlation_proxy,
)
from probe_noise_scale_optimizer_transfer import (
    CALIBRATION_CHANNELS,
    CALIBRATION_NOISE_STDS,
    CALIBRATION_REPEATS,
    GROUPED_STARTS,
    PROJECT_NOISE_STDS,
    PROJECT_REPEATS,
    SHARED_STARTS,
    fit_calibration,
    mixed_difference_noise_scale,
    scale_correction,
    shared_candidates,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CALIBRATION_SEED_BASE = 73000
PROJECT_SEED_BASE = 91000


def candidate_prediction(times: torch.Tensor, candidate) -> torch.Tensor:
    rates = torch.as_tensor(candidate.rates, dtype=DTYPE, device=DEVICE)
    weights = torch.as_tensor(candidate.weights, dtype=DTYPE, device=DEVICE)
    return lifted_response(times, weights, rates)


def cross_start_diagnostics(candidates, times: torch.Tensor, observations: torch.Tensor):
    """Compare the best objective with every independent initialization."""
    ordered = sorted(candidates, key=lambda candidate: candidate.train_rmse)
    best = ordered[0]
    best_prediction = candidate_prediction(times, best)
    signal_scale = float(torch.sqrt(observations.square().mean()).detach().cpu())
    signal_scale = max(signal_scale, 1.0e-12)
    diagnostics = []
    for candidate in ordered:
        prediction = candidate_prediction(times, candidate)
        prediction_gap = float(
            torch.sqrt((prediction - best_prediction).square().mean()).detach().cpu()
        ) / signal_scale
        diagnostics.append(
            {
                "train_objective_gap": abs(candidate.train_rmse - best.train_rmse)
                / signal_scale,
                "prediction_gap": prediction_gap,
                "train_rmse": candidate.train_rmse,
                "val_rmse": candidate.val_rmse,
                "rates": candidate.rates,
            }
        )
    return best, diagnostics, signal_scale


def fit_consistency_calibration(records: list[dict]) -> dict:
    """Use the maximum exact-control second-start gap as a frozen threshold."""
    objective = np.asarray(
        [record["second_start_objective_gap"] for record in records], dtype=float
    )
    prediction = np.asarray(
        [record["second_start_prediction_gap"] for record in records], dtype=float
    )
    return {
        "objective_gap_threshold": float(objective.max()),
        "prediction_gap_threshold": float(prediction.max()),
        "calibration_rule": "maximum second-best gap over disjoint exact controls",
        "minimum_consistent_starts": 2,
        "calibration_records": len(records),
    }


def select_consistent_candidate(candidates, times, observations, calibration: dict):
    best, diagnostics, signal_scale = cross_start_diagnostics(
        candidates, times, observations
    )
    consistent = [
        item
        for item in diagnostics
        if item["train_objective_gap"] <= calibration["objective_gap_threshold"]
        and item["prediction_gap"] <= calibration["prediction_gap_threshold"]
    ]
    if len(consistent) < calibration["minimum_consistent_starts"]:
        return None, diagnostics, len(consistent), signal_scale
    return best, diagnostics, len(consistent), signal_scale


def calibration_record(noise_std: float, repeat: int) -> tuple[dict, dict]:
    construction = HETEROGENEITY_CONSTRUCTIONS[
        repeat % len(HETEROGENEITY_CONSTRUCTIONS)
    ]
    seed = CALIBRATION_SEED_BASE + int(noise_std * 1.0e6) + repeat
    times, observations, train_idx, val_idx, _, _ = build_transfer_observation(
        CALIBRATION_CHANNELS, noise_std, construction, 0.0, seed
    )
    candidates = shared_candidates(times, observations, train_idx, val_idx, seed)
    best, diagnostics, signal_scale = cross_start_diagnostics(
        candidates, times, observations
    )
    correlation_proxy = second_difference_correlation_proxy(observations)
    noise_proxy = mixed_difference_noise_scale(observations)
    record = {
        "channels": CALIBRATION_CHANNELS,
        "noise_std_diagnostic": noise_std,
        "heterogeneity_construction_diagnostic": construction,
        "repeat": repeat,
        "seed": seed,
        "noise_scale_proxy": noise_proxy,
        "correlation_proxy": correlation_proxy,
        "signal_scale": signal_scale,
        "stage45_total_tolerance": frozen_total_tolerance(correlation_proxy),
        "consensus_val_rmse": best.val_rmse,
        "second_best_train_to_noise_ratio": 0.0,
        "second_start_objective_gap": diagnostics[1]["train_objective_gap"],
        "second_start_prediction_gap": diagnostics[1]["prediction_gap"],
        "cross_start_diagnostics": diagnostics,
    }
    return record, {"best_train_rmse": best.train_rmse}


def project_record(
    channels: int,
    noise_std: float,
    construction: str,
    drift: float,
    repeat: int,
    noise_calibration: dict,
    consistency_calibration: dict,
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
    times, observations, train_idx, val_idx, _, block_labels = (
        build_transfer_observation(channels, noise_std, construction, drift, seed)
    )
    candidates = shared_candidates(times, observations, train_idx, val_idx, seed)
    shared, diagnostics, consistent_starts, signal_scale = select_consistent_candidate(
        candidates, times, observations, consistency_calibration
    )
    noise_proxy = mixed_difference_noise_scale(observations)
    correlation_proxy = second_difference_correlation_proxy(observations)
    base_tolerance = frozen_total_tolerance(correlation_proxy)
    correction = scale_correction(noise_proxy, noise_calibration)
    total_tolerance = base_tolerance + correction
    in_scope = (
        noise_calibration["noise_proxy_min"]
        <= noise_proxy
        <= noise_calibration["noise_proxy_max"]
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
        support = None
        shared_bic = None
        shared_train_rmse = None
        shared_val_rmse = None
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
        "signal_scale": signal_scale,
        "diagnostics_in_calibration_scope": in_scope,
        "adequate_shared_starts": consistent_starts,
        "cross_start_diagnostics": diagnostics,
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


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cross_start_consistency_transfer.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Cross-start consistency transfer audit",
        "",
        f"Device: `{payload['device']}`; route pass: **{payload['summary']['route_pass']}**.",
        "",
        "| Channels | Noise std | Construction | Drift | Refused | Indeterminate | Adverse | Consistent starts | In scope |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["rows"]:
        lines.append(
            f"| {row['channels']} | {row['noise_std']:.1e} | "
            f"{row['heterogeneity_construction']} | {row['log_spectral_drift']:.2f} | "
            f"{row['refuse_fraction']:.2f} | {row['indeterminate_fraction']:.2f} | "
            f"{row['adverse_fraction']:.2f} | "
            f"{row['median_adequate_shared_starts']:.1f} | "
            f"{row['in_scope_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Calibration and project seeds are disjoint. Cross-start thresholds are the maximum exact-control second-start gaps and are never updated from project records.",
        ]
    )
    (RESULTS / "cross_start_consistency_transfer.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    calibration_records = []
    for noise_std in CALIBRATION_NOISE_STDS:
        for repeat in range(CALIBRATION_REPEATS):
            record, _ = calibration_record(noise_std, repeat)
            calibration_records.append(record)
            print(
                f"calibration noise={noise_std:.1e} repeat={repeat} "
                f"objective_gap={record['second_start_objective_gap']:.4g} "
                f"prediction_gap={record['second_start_prediction_gap']:.4g}",
                flush=True,
            )
    noise_calibration = fit_calibration(calibration_records)
    consistency_calibration = fit_consistency_calibration(calibration_records)
    print(f"frozen noise calibration={noise_calibration}", flush=True)
    print(f"frozen consistency calibration={consistency_calibration}", flush=True)

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
                            noise_calibration,
                            consistency_calibration,
                        )
                        project_records.append(record)
                        print(
                            f"project channels={channels} noise={noise_std:.1e} "
                            f"construction={construction} drift={drift:.2f} "
                            f"repeat={repeat} decision={record['decision']} "
                            f"consistent={record['adequate_shared_starts']}",
                            flush=True,
                        )
    summary = summarize(project_records)
    summary["frozen_rule"].update(
        {
            "optimizer_adequacy": "cross-start objective and prediction agreement",
            "objective_gap_threshold": consistency_calibration[
                "objective_gap_threshold"
            ],
            "prediction_gap_threshold": consistency_calibration[
                "prediction_gap_threshold"
            ],
            "uses_absolute_residual_level": False,
        }
    )
    payload = {
        "experiment": "cross_start_consistency_transfer",
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
        "frozen_noise_calibration": noise_calibration,
        "frozen_consistency_calibration": consistency_calibration,
        "project_records": project_records,
        "summary": summary,
    }
    write_outputs(payload)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
