"""Stress-test a frozen jitter-calibrated refusal rule across sampling processes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank
from probe_out_of_class_refusal import mean_lag1, prediction_from_fit
from probe_refusal_calibration import RESULTS, wilson_interval
from probe_multiwindow_external_calibration import (
    HORIZON,
    NOISE_STD,
    NUM_POINTS,
    WINDOW_FRACTIONS,
    response,
)


CALIBRATION_ARTIFACT = RESULTS / "sampling_aware_residual_calibration.json"
SAMPLING_PROCESSES = ("random_missing", "clustered", "long_gap")
ZERO_REPEATS = 6
PRIMARY_REPEATS = 4
SECONDARY_REPEATS = 4
FIT_RMSE_LIMIT = max(4.0 * NOISE_STD, 3.0e-3)
FIT_CONDITION_LIMIT = 1.0e8
SEED_OFFSET = 141000


def make_sampling_times(seed: int, process: str) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    dt = HORIZON / (NUM_POINTS - 1)
    dense = np.linspace(0.0, HORIZON, NUM_POINTS)
    dense[1:-1] += rng.uniform(-0.35 * dt, 0.35 * dt, size=NUM_POINTS - 2)
    dense = np.sort(dense)

    if process == "random_missing":
        keep = np.ones(NUM_POINTS, dtype=bool)
        keep[1:-1] = rng.random(NUM_POINTS - 2) >= 0.30
        times = dense[keep]
    elif process == "clustered":
        uniform = rng.uniform(0.0, HORIZON, size=30)
        support = np.concatenate([
            np.linspace(0.02, 0.14, 6) * HORIZON,
            np.linspace(0.36, 0.44, 6) * HORIZON,
            np.linspace(0.67, 0.78, 6) * HORIZON,
        ])
        centers = np.asarray([0.22, 0.54, 0.84]) * HORIZON
        clustered = np.concatenate([
            rng.normal(center, 0.045 * HORIZON, size=15) for center in centers
        ])
        times = np.unique(np.clip(
            np.concatenate(([0.0, HORIZON], support, uniform, clustered)),
            0.0,
            HORIZON,
        ))
    elif process == "long_gap":
        normalized = dense / HORIZON
        times = dense[(normalized < 0.36) | (normalized > 0.54)]
    else:
        raise ValueError(f"Unknown sampling process: {process}")
    return torch.tensor(np.sort(times), dtype=DTYPE, device=DEVICE)


def split_indices(times: torch.Tensor, seed: int):
    normalized = times.detach().cpu().numpy() / HORIZON
    windows = {
        name: np.flatnonzero((normalized >= lo) & (normalized < hi))
        for name, (lo, hi) in WINDOW_FRACTIONS.items()
    }
    if any(index.size < 4 for index in windows.values()):
        raise RuntimeError("Sampling process left fewer than four points in a diagnostic window")
    diagnostic = np.unique(np.concatenate(list(windows.values())))
    training_pool = np.setdiff1d(np.arange(1, len(times)), diagnostic)
    rng = np.random.default_rng(seed)
    train_size = min(42, training_pool.size)
    if train_size < 18:
        raise RuntimeError("Sampling process left too few training points")
    train = np.sort(rng.choice(training_pool, size=train_size, replace=False))
    return train, diagnostic, windows


def generate(case: str, repeat: int, process: str):
    seed = SEED_OFFSET + 1009 * SAMPLING_PROCESSES.index(process) + 101 * repeat + sum(map(ord, case))
    times = make_sampling_times(seed, process)
    clean = response(case, times)
    rng = np.random.default_rng(seed + 17)
    observations = clean + NOISE_STD * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )
    train, diagnostic, windows = split_indices(times, seed + 29)
    return seed, times, observations, train, diagnostic, windows


def fit_case(case: str, repeat: int, process: str, threshold: float) -> dict:
    seed, times, observations, train_np, diagnostic_np, windows_np = generate(
        case, repeat, process
    )
    train_idx = torch.tensor(train_np, dtype=torch.long, device=DEVICE)
    diagnostic_idx = torch.tensor(diagnostic_np, dtype=torch.long, device=DEVICE)
    fits = []
    for rank in (1, 2, 3):
        candidates = [
            fit_rank(
                times,
                observations,
                train_idx,
                diagnostic_idx,
                rank=rank,
                seed=seed * 100 + start,
                adam_steps=190,
                lbfgs_steps=55,
            )
            for start in range(2)
        ]
        fits.append(min(candidates, key=lambda item: item.bic))
    winner = min(fits, key=lambda item: item.bic)
    prediction = prediction_from_fit(times, winner)
    statistics = {}
    window_counts = {}
    for name, index_np in windows_np.items():
        index = torch.tensor(index_np, dtype=torch.long, device=DEVICE)
        statistics[name] = mean_lag1(prediction[index] - observations[index])
        window_counts[name] = int(index_np.size)
    max_abs = max(abs(value) for value in statistics.values())
    quality_pass = (
        np.isfinite(winner.val_rmse)
        and np.isfinite(winner.jacobian_condition)
        and winner.val_rmse <= FIT_RMSE_LIMIT
        and winner.jacobian_condition <= FIT_CONDITION_LIMIT
    )
    rank2 = next(item for item in fits if item.rank == 2)
    rank3 = next(item for item in fits if item.rank == 3)
    rank_cap = winner.rank == 3 and rank2.bic - rank3.bic >= 6.0
    decision = (
        "ACCEPT_CONTRACT"
        if quality_pass and not rank_cap and max_abs <= threshold
        else "REFUSE_CONTRACT"
    )
    return {
        "case": case,
        "repeat": repeat,
        "seed": seed,
        "sampling_process": process,
        "observation_count": int(len(times)),
        "training_count": int(len(train_np)),
        "window_counts": window_counts,
        "selected_rank": winner.rank,
        "validation_rmse": winner.val_rmse,
        "jacobian_condition": winner.jacobian_condition,
        "fit_quality_pass": bool(quality_pass),
        "rank3_vs_rank2_bic_gain": rank2.bic - rank3.bic,
        "statistics": statistics,
        "max_abs_statistic": max_abs,
        "strongest_window": max(statistics, key=lambda name: abs(statistics[name])),
        "decision": decision,
        "candidate_bic": {str(item.rank): item.bic for item in fits},
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for process in SAMPLING_PROCESSES:
        for case in sorted({record["case"] for record in records}):
            group = [
                record for record in records
                if record["sampling_process"] == process and record["case"] == case
            ]
            refused = sum(record["decision"] == "REFUSE_CONTRACT" for record in group)
            invalid = sum(not record["fit_quality_pass"] for record in group)
            rows.append({
                "sampling_process": process,
                "case": case,
                "trials": len(group),
                "refusal_fraction": refused / len(group),
                "refusal_wilson95": wilson_interval(refused, len(group)),
                "invalid_fit_fraction": invalid / len(group),
                "elevated_rank_fraction": sum(record["selected_rank"] > 1 for record in group) / len(group),
                "median_observation_count": float(np.median([record["observation_count"] for record in group])),
                "median_max_abs_statistic": float(np.median([record["max_abs_statistic"] for record in group])),
            })
    return rows


def assess(summary: list[dict]) -> dict:
    lookup = {(row["sampling_process"], row["case"]): row for row in summary}
    primary = ("oscillation_decay_016", "shifted_transient_020", "shifted_transient_055")
    checks = {}
    process_pass = {}
    for process in SAMPLING_PROCESSES:
        local = {
            "each_zero_false_refusal_at_most_1_of_6": all(
                lookup[(process, case)]["refusal_fraction"] <= 1.0 / 6.0
                for case in ("signed_zero", "oscillation_zero")
            ),
            "each_primary_stress_refused_at_least_3_of_4": all(
                lookup[(process, case)]["refusal_fraction"] >= 3.0 / 4.0
                for case in primary
            ),
            "invalid_fit_fraction_at_most_0.10": all(
                lookup[(process, case)]["invalid_fit_fraction"] <= 0.10
                for case in ("signed_zero", "oscillation_zero", *primary)
            ),
            "no_systematic_rank_absorption": all(
                lookup[(process, case)]["elevated_rank_fraction"] <= 0.50
                for case in primary
            ),
        }
        checks[process] = local
        process_pass[process] = all(local.values())
    return {
        "checks": checks,
        "process_pass": process_pass,
        "route_pass": all(process_pass.values()),
        "fast_decay_secondary_boundary": {
            process: lookup[(process, "oscillation_decay_036")]["refusal_fraction"]
            for process in SAMPLING_PROCESSES
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "sampling_process_stress.json"
    md_path = RESULTS / "sampling_process_stress.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Frozen-rule sampling-process stress test",
        "",
        f"- Frozen threshold: {payload['design']['frozen_threshold']:.6f}",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        "",
        "| Sampling | Case | Trials | Refusal | Invalid fit | Elevated rank | Median n |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['sampling_process']} | {row['case']} | {row['trials']} | "
            f"{row['refusal_fraction']:.3f} | {row['invalid_fit_fraction']:.3f} | "
            f"{row['elevated_rank_fraction']:.3f} | {row['median_observation_count']:.0f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    calibration = json.loads(CALIBRATION_ARTIFACT.read_text(encoding="utf-8"))
    threshold = float(calibration["calibration"]["methods"]["index_lag1"]["threshold"])
    records = []
    for process in SAMPLING_PROCESSES:
        for case in ("signed_zero", "oscillation_zero"):
            for repeat in range(ZERO_REPEATS):
                records.append(fit_case(case, repeat, process, threshold))
        for case in ("oscillation_decay_016", "shifted_transient_020", "shifted_transient_055"):
            for repeat in range(PRIMARY_REPEATS):
                records.append(fit_case(case, repeat, process, threshold))
        for repeat in range(SECONDARY_REPEATS):
            records.append(fit_case("oscillation_decay_036", repeat, process, threshold))
    summary = summarize(records)
    assessment = assess(summary)
    payload = {
        "experiment": "sampling_process_stress",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "design": {
            "frozen_threshold": threshold,
            "threshold_source": str(CALIBRATION_ARTIFACT),
            "threshold_reestimated": False,
            "sampling_processes": list(SAMPLING_PROCESSES),
            "random_missing_fraction": 0.30,
            "clustered_design": "18 support, 30 uniform, and 15 observations around each of three centers",
            "long_gap_fraction": [0.36, 0.54],
            "evaluation_candidate_ranks": [1, 2, 3],
            "starts_per_rank": 2,
            "primary_repeats_per_case": PRIMARY_REPEATS,
            "zero_repeats_per_case": ZERO_REPEATS,
            "fast_decay_is_secondary_boundary": True,
        },
        "records": records,
        "summary": summary,
        "assessment": assessment,
    }
    write_outputs(payload)
    print(json.dumps({"summary": summary, "assessment": assessment}, indent=2))


if __name__ == "__main__":
    main()
