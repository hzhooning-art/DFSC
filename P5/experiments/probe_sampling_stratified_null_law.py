"""Replicate null laws and validate sampling-stratified refusal thresholds."""

from __future__ import annotations

import json

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank
from probe_out_of_class_refusal import mean_lag1, prediction_from_fit
from probe_refusal_calibration import RESULTS, wilson_interval
from probe_multiwindow_external_calibration import NOISE_STD, WINDOW_FRACTIONS, response
from probe_sampling_process_stress import (
    FIT_CONDITION_LIMIT,
    FIT_RMSE_LIMIT,
    SAMPLING_PROCESSES,
    make_sampling_times,
    split_indices,
)


SOURCE_ARTIFACT = RESULTS / "sampling_aware_residual_calibration.json"
CALIBRATION_REPEATS_PER_FAMILY = 30
EVALUATION_REPEATS_PER_FAMILY = 20
CALIBRATION_OFFSET = 161000
EVALUATION_OFFSET = 181000
FAMILYWISE_ALPHA = 0.05


def seed_for(case: str, repeat: int, process: str, split: str) -> int:
    offset = CALIBRATION_OFFSET if split == "calibration" else EVALUATION_OFFSET
    return offset + 2003 * SAMPLING_PROCESSES.index(process) + 101 * repeat + sum(map(ord, case))


def fit_null(case: str, repeat: int, process: str, split: str) -> dict:
    seed = seed_for(case, repeat, process, split)
    times = make_sampling_times(seed, process)
    clean = response(case, times)
    rng = np.random.default_rng(seed + 17)
    observations = clean + NOISE_STD * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )
    train_np, diagnostic_np, windows_np = split_indices(times, seed + 29)
    train_idx = torch.tensor(train_np, dtype=torch.long, device=DEVICE)
    diagnostic_idx = torch.tensor(diagnostic_np, dtype=torch.long, device=DEVICE)
    candidates = [
        fit_rank(
            times,
            observations,
            train_idx,
            diagnostic_idx,
            rank=1,
            seed=seed * 100 + start,
            adam_steps=165,
            lbfgs_steps=48,
        )
        for start in range(2)
    ]
    winner = min(candidates, key=lambda item: item.bic)
    prediction = prediction_from_fit(times, winner)
    statistics = {}
    for name, index_np in windows_np.items():
        index = torch.tensor(index_np, dtype=torch.long, device=DEVICE)
        statistics[name] = mean_lag1(prediction[index] - observations[index])
    quality_pass = (
        np.isfinite(winner.val_rmse)
        and np.isfinite(winner.jacobian_condition)
        and winner.val_rmse <= FIT_RMSE_LIMIT
        and winner.jacobian_condition <= FIT_CONDITION_LIMIT
    )
    return {
        "split": split,
        "case": case,
        "repeat": repeat,
        "seed": seed,
        "sampling_process": process,
        "observation_count": int(len(times)),
        "training_count": int(len(train_np)),
        "validation_rmse": winner.val_rmse,
        "jacobian_condition": winner.jacobian_condition,
        "fit_quality_pass": bool(quality_pass),
        "statistics": statistics,
        "max_abs_statistic": max(abs(value) for value in statistics.values()),
        "strongest_window": max(statistics, key=lambda name: abs(statistics[name])),
    }


def calibrate(records: list[dict], process: str) -> dict:
    local = [
        record for record in records
        if record["sampling_process"] == process and record["fit_quality_pass"]
    ]
    maxima = np.asarray([record["max_abs_statistic"] for record in local])
    quantile = 1.0 - FAMILYWISE_ALPHA
    threshold = float(np.quantile(maxima, quantile, method="higher"))
    rng = np.random.default_rng(91031 + SAMPLING_PROCESSES.index(process))
    bootstrap = []
    for _ in range(2000):
        sample = rng.choice(maxima, size=maxima.size, replace=True)
        bootstrap.append(float(np.quantile(sample, quantile, method="higher")))
    lo = float(np.quantile(bootstrap, 0.025))
    hi = float(np.quantile(bootstrap, 0.975))
    return {
        "sampling_process": process,
        "method": "95th empirical quantile of independent per-fit three-window maxima",
        "familywise_alpha": FAMILYWISE_ALPHA,
        "independent_calibration_fits": sum(
            record["sampling_process"] == process for record in records
        ),
        "valid_calibration_fits": len(local),
        "invalid_calibration_fits": sum(
            record["sampling_process"] == process and not record["fit_quality_pass"]
            for record in records
        ),
        "threshold": threshold,
        "bootstrap_interval95": [lo, hi],
        "bootstrap_width": hi - lo,
        "calibration_max": float(np.max(maxima)),
    }


def apply_decisions(records: list[dict], frozen: float, strata: dict[str, dict]) -> None:
    for record in records:
        process_threshold = strata[record["sampling_process"]]["threshold"]
        record["decisions"] = {
            "frozen_jitter_threshold": (
                "ACCEPT_CONTRACT"
                if record["fit_quality_pass"] and record["max_abs_statistic"] <= frozen
                else "REFUSE_CONTRACT"
            ),
            "sampling_stratified_threshold": (
                "ACCEPT_CONTRACT"
                if record["fit_quality_pass"] and record["max_abs_statistic"] <= process_threshold
                else "REFUSE_CONTRACT"
            ),
        }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for process in SAMPLING_PROCESSES:
        process_group = [record for record in records if record["sampling_process"] == process]
        for case in ("all_zero_controls", "signed_zero", "oscillation_zero"):
            group = process_group if case == "all_zero_controls" else [
                record for record in process_group if record["case"] == case
            ]
            methods = {}
            for method in ("frozen_jitter_threshold", "sampling_stratified_threshold"):
                refused = sum(record["decisions"][method] == "REFUSE_CONTRACT" for record in group)
                methods[method] = {
                    "refusals": refused,
                    "false_refusal_fraction": refused / len(group),
                    "wilson95": wilson_interval(refused, len(group)),
                }
            rows.append({
                "sampling_process": process,
                "case": case,
                "trials": len(group),
                "invalid_fit_count": sum(not record["fit_quality_pass"] for record in group),
                "median_max_abs_statistic": float(np.median([
                    record["max_abs_statistic"] for record in group
                ])),
                "methods": methods,
            })
    return rows


def assess(calibration: dict[str, dict], summary: list[dict]) -> dict:
    lookup = {(row["sampling_process"], row["case"]): row for row in summary}
    checks = {}
    process_pass = {}
    for process in SAMPLING_PROCESSES:
        result = lookup[(process, "all_zero_controls")]["methods"]["sampling_stratified_threshold"]
        local = {
            "at_least_59_of_60_valid_calibration_fits": (
                calibration[process]["valid_calibration_fits"] >= 59
            ),
            "bootstrap_width_at_most_0.20": calibration[process]["bootstrap_width"] <= 0.20,
            "threshold_at_most_0.40": calibration[process]["threshold"] <= 0.40,
            "heldout_refusals_at_most_3_of_40": result["refusals"] <= 3,
            "heldout_wilson_upper_at_most_0.20": result["wilson95"][1] <= 0.20,
        }
        checks[process] = local
        process_pass[process] = all(local.values())
    return {
        "checks": checks,
        "process_pass": process_pass,
        "route_pass": all(process_pass.values()),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sampling_stratified_null_law.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Sampling-stratified null-law replication",
        "",
        f"- Frozen jitter threshold: {payload['design']['frozen_jitter_threshold']:.6f}",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        "",
        "| Sampling | Stratum threshold | Bootstrap 95% | Frozen false refusal | Stratified false refusal |",
        "|---|---:|---:|---:|---:|",
    ]
    lookup = {(row["sampling_process"], row["case"]): row for row in payload["summary"]}
    for process in SAMPLING_PROCESSES:
        row = lookup[(process, "all_zero_controls")]
        frozen = row["methods"]["frozen_jitter_threshold"]
        stratified = row["methods"]["sampling_stratified_threshold"]
        cal = payload["calibration"][process]
        lines.append(
            f"| {process} | {cal['threshold']:.6f} | {cal['bootstrap_interval95']} | "
            f"{frozen['refusals']}/{row['trials']} | {stratified['refusals']}/{row['trials']} |"
        )
    (RESULTS / "sampling_stratified_null_law.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    source = json.loads(SOURCE_ARTIFACT.read_text(encoding="utf-8"))
    frozen = float(source["calibration"]["methods"]["index_lag1"]["threshold"])
    calibration_records = []
    evaluation_records = []
    for process in SAMPLING_PROCESSES:
        for case in ("signed_zero", "oscillation_zero"):
            for repeat in range(CALIBRATION_REPEATS_PER_FAMILY):
                calibration_records.append(fit_null(case, repeat, process, "calibration"))
            for repeat in range(EVALUATION_REPEATS_PER_FAMILY):
                evaluation_records.append(fit_null(case, repeat, process, "evaluation"))
    calibration = {
        process: calibrate(calibration_records, process) for process in SAMPLING_PROCESSES
    }
    apply_decisions(evaluation_records, frozen, calibration)
    summary = summarize(evaluation_records)
    assessment = assess(calibration, summary)
    payload = {
        "experiment": "sampling_stratified_null_law",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "design": {
            "frozen_jitter_threshold": frozen,
            "threshold_source": str(SOURCE_ARTIFACT),
            "calibration_fits_per_sampling_process": 60,
            "heldout_fits_per_sampling_process": 40,
            "calibration_and_evaluation_seeds_disjoint": True,
            "candidate_rank": 1,
            "starts_per_fit": 2,
            "threshold_unit": "independent per-fit three-window maximum",
        },
        "calibration": calibration,
        "calibration_records": calibration_records,
        "evaluation_records": evaluation_records,
        "summary": summary,
        "assessment": assessment,
    }
    write_outputs(payload)
    print(json.dumps({
        "calibration": calibration,
        "summary": summary,
        "assessment": assessment,
    }, indent=2))


if __name__ == "__main__":
    main()
