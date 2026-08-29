"""Test separate model-discrepancy and observable-noise tolerance budgets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from probe_approximate_sharing_refusal_boundary import build_block_heterogeneous_observation
from probe_high_dimensional_shared_spectrum import DTYPE, DEVICE
from probe_nested_group_sharing_gate import fit_grouped_candidate
from probe_noise_aware_sharing_gate import (
    BLOCKS,
    PROJECT_CORRELATIONS,
    PROJECT_DRIFTS,
    best_shared_fit,
    calibrated_limit,
    classify_with_limit,
    fit_calibration_envelope,
    second_difference_correlation_proxy,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NOISE_CALIBRATION_CORRELATIONS = (0.0, 0.20, 0.40, 0.60, 0.80)
NOISE_CALIBRATION_REPEATS = 5
NOISE_CALIBRATION_OFFSET = 400
ALLOWED_MODEL_DRIFT = 0.05
MODEL_CALIBRATION_CORRELATIONS = (0.0, 0.60)
MODEL_CALIBRATION_REPEATS = 5
MODEL_CALIBRATION_OFFSET = 500
PROJECT_REPEATS = 4
PROJECT_REPEAT_OFFSET = 600
MODEL_COVERAGE = 0.90


def fit_proxy_scope(records: list[dict]) -> dict:
    """Add a calibration-only replicate guard band around the observed proxy range."""
    grouped = {}
    for record in records:
        grouped.setdefault(record["noise_correlation_diagnostic"], []).append(
            record["correlation_proxy"]
        )
    replicate_spreads = [
        float(np.std(values, ddof=1)) for values in grouped.values() if len(values) > 1
    ]
    padding = max(replicate_spreads, default=0.0)
    proxies = np.asarray([record["correlation_proxy"] for record in records])
    return {
        "padding": padding,
        "proxy_min": max(0.0, float(proxies.min()) - padding),
        "proxy_max": min(0.95, float(proxies.max()) + padding),
    }


def fit_model_allowance(records: list[dict], noise_envelope: dict) -> dict:
    """Estimate the one-sided allowance at the declared approximation boundary."""
    excesses = np.asarray(
        [
            record["shared_val_rmse"]
            - (
                noise_envelope["intercept"]
                + noise_envelope["slope"] * record["correlation_proxy"]
            )
            for record in records
        ],
        dtype=float,
    )
    order = min(
        len(excesses) - 1,
        math.ceil((len(excesses) + 1) * MODEL_COVERAGE) - 1,
    )
    return {
        "allowance": max(0.0, float(np.sort(excesses)[order])),
        "coverage_target": MODEL_COVERAGE,
        "calibrated_max_log_spectral_drift": ALLOWED_MODEL_DRIFT,
        "excess_min": float(excesses.min()),
        "excess_max": float(excesses.max()),
    }


def decomposed_limit(proxy: float, noise_envelope: dict, model_budget: dict) -> float:
    return calibrated_limit(proxy, noise_envelope) + model_budget["allowance"]


def make_observation(drift: float, rho: float, repeat: int, offset: int):
    seed = 51000 + int(1000 * drift) + int(100 * rho) + offset + repeat
    values = build_block_heterogeneous_observation(drift, rho, seed)
    return seed, values


def calibration_record(drift: float, rho: float, repeat: int, offset: int) -> dict:
    seed, (times, observations, train_idx, val_idx, _, _) = make_observation(
        drift, rho, repeat, offset
    )
    shared = best_shared_fit(times, observations, train_idx, val_idx, seed)
    return {
        "log_spectral_drift": drift,
        "noise_correlation_diagnostic": rho,
        "local_repeat": repeat,
        "seed": seed,
        "correlation_proxy": second_difference_correlation_proxy(observations),
        "shared_val_rmse": shared.val_rmse,
    }


def project_record(
    drift: float,
    rho: float,
    repeat: int,
    noise_envelope: dict,
    model_budget: dict,
    proxy_scope: dict,
) -> dict:
    seed, (times, observations, train_idx, val_idx, _, block_labels) = make_observation(
        drift, rho, repeat, PROJECT_REPEAT_OFFSET
    )
    shared = best_shared_fit(times, observations, train_idx, val_idx, seed)
    labels = torch.tensor(block_labels, dtype=torch.long, device=DEVICE)
    grouped = min(
        [
            fit_grouped_candidate(
                times, observations, train_idx, val_idx, labels, seed * 10 + 5 + start
            )
            for start in range(2)
        ],
        key=lambda item: item.bic,
    )
    support = shared.bic - grouped.bic
    proxy = second_difference_correlation_proxy(observations)
    noise_only_limit = calibrated_limit(proxy, noise_envelope)
    total_limit = decomposed_limit(proxy, noise_envelope, model_budget)
    return {
        "log_spectral_drift": drift,
        "noise_correlation_diagnostic": rho,
        "local_repeat": repeat,
        "seed": seed,
        "correlation_proxy": proxy,
        "proxy_in_calibration_scope": proxy_scope["proxy_min"] <= proxy <= proxy_scope["proxy_max"],
        "noise_tolerance": noise_only_limit,
        "model_tolerance": model_budget["allowance"],
        "total_tolerance": total_limit,
        "noise_only_decision": classify_with_limit(
            support, shared.val_rmse, grouped.val_rmse, noise_only_limit
        ),
        "decomposed_decision": classify_with_limit(
            support, shared.val_rmse, grouped.val_rmse, total_limit
        ),
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
                    "noise_only_refuse_fraction": float(
                        np.mean(
                            [r["noise_only_decision"] == "REFUSE_SHARED_MECHANISM" for r in group]
                        )
                    ),
                    "decomposed_refuse_fraction": float(
                        np.mean(
                            [r["decomposed_decision"] == "REFUSE_SHARED_MECHANISM" for r in group]
                        )
                    ),
                    "median_proxy": float(np.median([r["correlation_proxy"] for r in group])),
                    "median_total_tolerance": float(
                        np.median([r["total_tolerance"] for r in group])
                    ),
                    "median_shared_val_rmse": float(
                        np.median([r["shared_val_rmse"] for r in group])
                    ),
                    "in_scope_fraction": float(
                        np.mean([r["proxy_in_calibration_scope"] for r in group])
                    ),
                }
            )
    cells = {
        (row["log_spectral_drift"], row["noise_correlation_diagnostic"]): row
        for row in rows
    }
    mild_retained = all(
        cells[(drift, rho)]["decomposed_refuse_fraction"] <= 0.25
        for drift in (0.0, 0.05)
        for rho in PROJECT_CORRELATIONS
    )
    severe_refused = all(
        cells[(0.15, rho)]["decomposed_refuse_fraction"] >= 0.75
        for rho in PROJECT_CORRELATIONS
    )
    boundary_gap = abs(
        cells[(0.075, 0.0)]["decomposed_refuse_fraction"]
        - cells[(0.075, 0.60)]["decomposed_refuse_fraction"]
    )
    checks = {
        "complete_project_matrix": len(records)
        == len(PROJECT_DRIFTS) * len(PROJECT_CORRELATIONS) * PROJECT_REPEATS,
        "exact_and_mild_sharing_retained": mild_retained,
        "severe_drift_refused": severe_refused,
        "boundary_noise_gap_at_most_0_25": boundary_gap <= 0.25,
        "all_project_proxies_in_scope": all(
            record["proxy_in_calibration_scope"] for record in records
        ),
    }
    return {
        "rows": rows,
        "boundary_noise_gap": boundary_gap,
        "checks": checks,
        "route_pass": bool(all(checks.values())),
        "frozen_rule": {
            "allowed_model_drift": ALLOWED_MODEL_DRIFT,
            "mild_refuse_fraction_max": 0.25,
            "severe_refuse_fraction_min": 0.75,
            "boundary_drift": 0.075,
            "boundary_noise_gap_max": 0.25,
            "out_of_proxy_scope_action": "route failure",
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "decomposed_tolerance_sharing_gate.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Decomposed model-and-noise tolerance sharing gate",
        "",
        f"Device: `{payload['device']}`; route pass: **{payload['summary']['route_pass']}**.",
        "",
        f"Model allowance: `{payload['model_budget']['allowance']:.6g}`.",
        "",
        "| Drift | True rho (diagnostic) | Noise-only refused | Decomposed refused | Proxy | Total limit | Shared RMSE | In scope |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["rows"]:
        lines.append(
            f"| {row['log_spectral_drift']:.3f} | {row['noise_correlation_diagnostic']:.2f} | "
            f"{row['noise_only_refuse_fraction']:.2f} | {row['decomposed_refuse_fraction']:.2f} | "
            f"{row['median_proxy']:.3f} | {row['median_total_tolerance']:.4g} | "
            f"{row['median_shared_val_rmse']:.4g} | {row['in_scope_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The model allowance is calibrated on an independent bank at the predeclared approximate-sharing boundary; project decisions use only the observed proxy.",
        ]
    )
    (RESULTS / "decomposed_tolerance_sharing_gate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    noise_calibration = [
        calibration_record(0.0, rho, repeat, NOISE_CALIBRATION_OFFSET)
        for rho in NOISE_CALIBRATION_CORRELATIONS
        for repeat in range(NOISE_CALIBRATION_REPEATS)
    ]
    noise_envelope = fit_calibration_envelope(noise_calibration)
    proxy_scope = fit_proxy_scope(noise_calibration)
    model_calibration = [
        calibration_record(
            ALLOWED_MODEL_DRIFT, rho, repeat, MODEL_CALIBRATION_OFFSET
        )
        for rho in MODEL_CALIBRATION_CORRELATIONS
        for repeat in range(MODEL_CALIBRATION_REPEATS)
    ]
    model_budget = fit_model_allowance(model_calibration, noise_envelope)
    print(
        f"noise_envelope={noise_envelope} proxy_scope={proxy_scope} "
        f"model_budget={model_budget}",
        flush=True,
    )

    project = []
    for drift in PROJECT_DRIFTS:
        for rho in PROJECT_CORRELATIONS:
            for repeat in range(PROJECT_REPEATS):
                record = project_record(
                    drift,
                    rho,
                    repeat,
                    noise_envelope,
                    model_budget,
                    proxy_scope,
                )
                project.append(record)
                print(
                    f"project drift={drift:.3f} rho={rho:.2f} repeat={repeat} "
                    f"decision={record['decomposed_decision']} proxy={record['correlation_proxy']:.3f} "
                    f"limit={record['total_tolerance']:.4g} rmse={record['shared_val_rmse']:.4g}",
                    flush=True,
                )

    summary = summarize(project)
    payload = {
        "experiment": "decomposed_tolerance_sharing_gate",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "protocol": {
            "noise_calibration_correlations": list(NOISE_CALIBRATION_CORRELATIONS),
            "noise_calibration_repeats": NOISE_CALIBRATION_REPEATS,
            "allowed_model_drift": ALLOWED_MODEL_DRIFT,
            "model_calibration_correlations": list(MODEL_CALIBRATION_CORRELATIONS),
            "model_calibration_repeats": MODEL_CALIBRATION_REPEATS,
            "project_drifts": list(PROJECT_DRIFTS),
            "project_correlations": list(PROJECT_CORRELATIONS),
            "project_repeats": PROJECT_REPEATS,
        },
        "noise_envelope": noise_envelope,
        "proxy_scope": proxy_scope,
        "model_budget": model_budget,
        "noise_calibration_records": noise_calibration,
        "model_calibration_records": model_calibration,
        "project_records": project,
        "summary": summary,
    }
    write_outputs(payload)
    print(json.dumps({"route_pass": summary["route_pass"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
