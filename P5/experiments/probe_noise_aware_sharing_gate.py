"""Calibrate a non-oracle, correlation-aware sharing refusal gate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from probe_approximate_sharing_refusal_boundary import (
    BLOCKS,
    build_block_heterogeneous_observation,
)
from probe_high_dimensional_shared_spectrum import DTYPE, DEVICE, fit_candidate
from probe_memory_rank import lifted_response
from probe_nested_group_sharing_gate import (
    BIC_EVIDENCE_LIMIT,
    RELATIVE_DEGRADATION_LIMIT,
    VALIDATION_RMSE_LIMIT,
    fit_grouped_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CALIBRATION_CORRELATIONS = (0.0, 0.20, 0.40, 0.60, 0.80)
CALIBRATION_REPEATS = 4
CALIBRATION_REPEAT_OFFSET = 200
PROJECT_DRIFTS = (0.0, 0.05, 0.075, 0.15)
PROJECT_CORRELATIONS = (0.0, 0.60)
PROJECT_REPEATS = 4
PROJECT_REPEAT_OFFSET = 300
CALIBRATION_COVERAGE = 0.90


def second_difference_correlation_proxy(observations: torch.Tensor) -> float:
    """Estimate common-channel noise correlation after suppressing smooth signal."""
    differences = observations[2:] - 2.0 * observations[1:-1] + observations[:-2]
    differences = differences - differences.mean(dim=0, keepdim=True)
    covariance = differences.T @ differences / max(differences.shape[0] - 1, 1)
    diagonal = torch.diagonal(covariance)
    total_variance = diagonal.mean().clamp_min(1.0e-30)
    off_diagonal_sum = covariance.sum() - diagonal.sum()
    channels = observations.shape[1]
    off_diagonal_mean = off_diagonal_sum / max(channels * (channels - 1), 1)
    estimate = torch.clamp(off_diagonal_mean / total_variance, 0.0, 0.95)
    return float(estimate.detach().cpu())


def best_shared_fit(
    times: torch.Tensor,
    observations: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    seed: int,
):
    candidates = [
        fit_candidate(times, observations, train_idx, val_idx, 2, True, seed * 10 + start)
        for start in range(2)
    ]
    return min(candidates, key=lambda item: item.bic)


def fit_calibration_envelope(records: list[dict]) -> dict:
    """Fit a monotone affine centre plus a one-sided conformal residual."""
    x = np.asarray([record["correlation_proxy"] for record in records], dtype=float)
    y = np.asarray([record["shared_val_rmse"] for record in records], dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope = max(0.0, float(coefficients[1]))
    intercept = float(np.mean(y) - slope * np.mean(x))
    residuals = y - (intercept + slope * x)
    order = min(len(residuals) - 1, math.ceil((len(residuals) + 1) * CALIBRATION_COVERAGE) - 1)
    one_sided_residual = float(np.sort(residuals)[order])
    return {
        "intercept": intercept,
        "slope": slope,
        "one_sided_residual": one_sided_residual,
        "coverage_target": CALIBRATION_COVERAGE,
        "proxy_min": float(np.min(x)),
        "proxy_max": float(np.max(x)),
    }


def calibrated_limit(proxy: float, envelope: dict) -> float:
    return max(
        1.0e-6,
        envelope["intercept"] + envelope["slope"] * proxy + envelope["one_sided_residual"],
    )


def classify_with_limit(
    group_support: float,
    shared_val_rmse: float,
    grouped_val_rmse: float,
    validation_limit: float,
) -> str:
    degradation = shared_val_rmse / max(grouped_val_rmse, 1.0e-15)
    heterogeneity_detected = group_support >= BIC_EVIDENCE_LIMIT
    materially_inadequate = (
        shared_val_rmse > validation_limit
        or (heterogeneity_detected and degradation > RELATIVE_DEGRADATION_LIMIT)
    )
    if materially_inadequate:
        return "REFUSE_SHARED_MECHANISM"
    if heterogeneity_detected:
        return "ACCEPT_WITH_SCOPE_LIMITS"
    return "ACCEPT_SHARED_MECHANISM"


def calibration_record(noise_correlation: float, local_repeat: int) -> dict:
    repeat = CALIBRATION_REPEAT_OFFSET + local_repeat
    seed = 41000 + int(100 * noise_correlation) + repeat
    times, observations, train_idx, val_idx, _, _ = build_block_heterogeneous_observation(
        0.0, noise_correlation, seed
    )
    shared = best_shared_fit(times, observations, train_idx, val_idx, seed)
    return {
        "noise_correlation_diagnostic": noise_correlation,
        "local_repeat": local_repeat,
        "seed": seed,
        "correlation_proxy": second_difference_correlation_proxy(observations),
        "shared_val_rmse": shared.val_rmse,
    }


def project_record(
    log_spectral_drift: float,
    noise_correlation: float,
    local_repeat: int,
    envelope: dict,
) -> dict:
    repeat = PROJECT_REPEAT_OFFSET + local_repeat
    seed = 41000 + int(1000 * log_spectral_drift) + int(100 * noise_correlation) + repeat
    times, observations, train_idx, val_idx, _, block_labels = build_block_heterogeneous_observation(
        log_spectral_drift, noise_correlation, seed
    )
    shared = best_shared_fit(times, observations, train_idx, val_idx, seed)
    labels = torch.tensor(block_labels, dtype=torch.long, device=DEVICE)
    grouped_candidates = [
        fit_grouped_candidate(
            times, observations, train_idx, val_idx, labels, seed * 10 + 5 + start
        )
        for start in range(2)
    ]
    grouped = min(grouped_candidates, key=lambda item: item.bic)
    support = shared.bic - grouped.bic
    proxy = second_difference_correlation_proxy(observations)
    limit = calibrated_limit(proxy, envelope)
    old_decision = classify_with_limit(
        support, shared.val_rmse, grouped.val_rmse, VALIDATION_RMSE_LIMIT
    )
    new_decision = classify_with_limit(support, shared.val_rmse, grouped.val_rmse, limit)
    return {
        "log_spectral_drift": log_spectral_drift,
        "noise_correlation_diagnostic": noise_correlation,
        "local_repeat": local_repeat,
        "seed": seed,
        "correlation_proxy": proxy,
        "calibrated_validation_limit": limit,
        "proxy_in_calibration_scope": envelope["proxy_min"] <= proxy <= envelope["proxy_max"],
        "old_decision": old_decision,
        "noise_aware_decision": new_decision,
        "group_bic_support": support,
        "shared_val_rmse": shared.val_rmse,
        "grouped_val_rmse": grouped.val_rmse,
        "shared_to_grouped_val_ratio": shared.val_rmse / max(grouped.val_rmse, 1.0e-15),
    }


def summarize(records: list[dict]) -> dict:
    rows = []
    for drift in PROJECT_DRIFTS:
        for rho in PROJECT_CORRELATIONS:
            group = [
                record
                for record in records
                if record["log_spectral_drift"] == drift
                and record["noise_correlation_diagnostic"] == rho
            ]
            rows.append(
                {
                    "log_spectral_drift": drift,
                    "noise_correlation_diagnostic": rho,
                    "trials": len(group),
                    "old_refuse_fraction": float(np.mean([r["old_decision"] == "REFUSE_SHARED_MECHANISM" for r in group])),
                    "noise_aware_refuse_fraction": float(np.mean([r["noise_aware_decision"] == "REFUSE_SHARED_MECHANISM" for r in group])),
                    "median_correlation_proxy": float(np.median([r["correlation_proxy"] for r in group])),
                    "median_calibrated_limit": float(np.median([r["calibrated_validation_limit"] for r in group])),
                    "median_shared_val_rmse": float(np.median([r["shared_val_rmse"] for r in group])),
                    "in_scope_fraction": float(np.mean([r["proxy_in_calibration_scope"] for r in group])),
                }
            )
    cells = {(row["log_spectral_drift"], row["noise_correlation_diagnostic"]): row for row in rows}
    mild_retained = all(
        cells[(drift, rho)]["noise_aware_refuse_fraction"] <= 0.25
        for drift in (0.0, 0.05)
        for rho in PROJECT_CORRELATIONS
    )
    severe_refused = all(
        cells[(0.15, rho)]["noise_aware_refuse_fraction"] >= 0.75
        for rho in PROJECT_CORRELATIONS
    )
    old_boundary_gap = abs(cells[(0.075, 0.0)]["old_refuse_fraction"] - cells[(0.075, 0.60)]["old_refuse_fraction"])
    new_boundary_gap = abs(cells[(0.075, 0.0)]["noise_aware_refuse_fraction"] - cells[(0.075, 0.60)]["noise_aware_refuse_fraction"])
    checks = {
        "complete_project_matrix": len(records) == len(PROJECT_DRIFTS) * len(PROJECT_CORRELATIONS) * PROJECT_REPEATS,
        "mild_sharing_retained": mild_retained,
        "severe_drift_refused": severe_refused,
        "boundary_noise_gap_not_worse": new_boundary_gap <= old_boundary_gap,
        "all_project_proxies_in_scope": all(record["proxy_in_calibration_scope"] for record in records),
    }
    return {
        "rows": rows,
        "old_boundary_noise_gap": old_boundary_gap,
        "noise_aware_boundary_noise_gap": new_boundary_gap,
        "checks": checks,
        "route_pass": bool(all(checks.values())),
        "frozen_rule": {
            "mild_refuse_fraction_max": 0.25,
            "severe_refuse_fraction_min": 0.75,
            "boundary_drift": 0.075,
            "noise_gap_must_not_increase": True,
            "out_of_proxy_scope_action": "route failure",
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "noise_aware_sharing_gate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Observable correlation-aware sharing gate",
        "",
        f"Device: `{payload['device']}`; route pass: **{payload['summary']['route_pass']}**.",
        "",
        "| Drift | True rho (diagnostic) | Old refused | Noise-aware refused | Proxy | Calibrated limit | Shared RMSE | In scope |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["rows"]:
        lines.append(
            f"| {row['log_spectral_drift']:.3f} | {row['noise_correlation_diagnostic']:.2f} | "
            f"{row['old_refuse_fraction']:.2f} | {row['noise_aware_refuse_fraction']:.2f} | "
            f"{row['median_correlation_proxy']:.3f} | {row['median_calibrated_limit']:.4g} | "
            f"{row['median_shared_val_rmse']:.4g} | {row['in_scope_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The true noise correlation is retained only as an evaluation diagnostic; the gate uses the observed second-difference proxy.",
        ]
    )
    (RESULTS / "noise_aware_sharing_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    calibration = []
    for rho in CALIBRATION_CORRELATIONS:
        for repeat in range(CALIBRATION_REPEATS):
            record = calibration_record(rho, repeat)
            calibration.append(record)
            print(
                f"calibration rho={rho:.2f} repeat={repeat} proxy={record['correlation_proxy']:.3f} "
                f"rmse={record['shared_val_rmse']:.4g}",
                flush=True,
            )
    envelope = fit_calibration_envelope(calibration)
    print(f"envelope={envelope}", flush=True)

    project = []
    for drift in PROJECT_DRIFTS:
        for rho in PROJECT_CORRELATIONS:
            for repeat in range(PROJECT_REPEATS):
                record = project_record(drift, rho, repeat, envelope)
                project.append(record)
                print(
                    f"project drift={drift:.3f} rho={rho:.2f} repeat={repeat} "
                    f"old={record['old_decision']} new={record['noise_aware_decision']} "
                    f"proxy={record['correlation_proxy']:.3f} limit={record['calibrated_validation_limit']:.4g}",
                    flush=True,
                )
    summary = summarize(project)
    payload = {
        "experiment": "noise_aware_sharing_gate",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "protocol": {
            "calibration_correlations": list(CALIBRATION_CORRELATIONS),
            "calibration_repeats": CALIBRATION_REPEATS,
            "project_drifts": list(PROJECT_DRIFTS),
            "project_correlations": list(PROJECT_CORRELATIONS),
            "project_repeats": PROJECT_REPEATS,
        },
        "calibration_envelope": envelope,
        "calibration_records": calibration,
        "project_records": project,
        "summary": summary,
    }
    write_outputs(payload)
    print(json.dumps({"route_pass": summary["route_pass"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
