"""Large-null calibration and external stress test for the multi-window gate."""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank
from probe_out_of_class_refusal import mean_lag1, oscillatory_response, prediction_from_fit
from probe_refusal_calibration import RESULTS, wilson_interval
from probe_joint_horizon_noise_effects import clean_response


HORIZON = 18.0
NOISE_STD = 6.0e-4
NUM_POINTS = 124
CALIBRATION_REPEATS_PER_FAMILY = 60
ZERO_TEST_REPEATS = 12
STRESS_TEST_REPEATS = 6
ALPHA_FAMILYWISE = 0.05
WINDOW_FRACTIONS = {
    "early": (0.15, 0.35),
    "middle": (0.45, 0.65),
    "late": (0.80, 1.001),
}


def make_times(seed: int, sampling: str) -> torch.Tensor:
    base = np.linspace(0.0, HORIZON, NUM_POINTS)
    if sampling == "jittered":
        rng = np.random.default_rng(seed)
        dt = HORIZON / (NUM_POINTS - 1)
        base[1:-1] += rng.uniform(-0.35 * dt, 0.35 * dt, size=NUM_POINTS - 2)
        base = np.sort(base)
    elif sampling != "regular":
        raise ValueError(f"Unknown sampling mode: {sampling}")
    return torch.tensor(base, dtype=DTYPE, device=DEVICE)


def split_indices(times: torch.Tensor, seed: int):
    normalized = times.detach().cpu().numpy() / HORIZON
    windows = {}
    for name, (lo, hi) in WINDOW_FRACTIONS.items():
        windows[name] = np.flatnonzero((normalized >= lo) & (normalized < hi))
    diagnostic = np.unique(np.concatenate(list(windows.values())))
    training_pool = np.setdiff1d(np.arange(1, NUM_POINTS), diagnostic)
    rng = np.random.default_rng(seed)
    train = np.sort(rng.choice(training_pool, size=min(48, training_pool.size), replace=False))
    return train, diagnostic, windows


def response(case: str, times: torch.Tensor) -> torch.Tensor:
    if case == "signed_zero":
        return clean_response("signed_residue", 0.0, times)
    if case == "oscillation_zero":
        return clean_response("oscillation", 0.0, times)
    if case == "oscillation_decay_016":
        scale = torch.linspace(0.72, 1.28, 6, dtype=DTYPE, device=DEVICE)
        return oscillatory_response(times, 0.38 * scale, decay=0.16, frequency=0.04)
    if case == "oscillation_decay_036":
        scale = torch.linspace(0.72, 1.28, 6, dtype=DTYPE, device=DEVICE)
        return oscillatory_response(times, 0.38 * scale, decay=0.36, frequency=0.04)
    if case.startswith("shifted_transient_"):
        onset_fraction = 0.20 if case.endswith("020") else 0.55
        base = clean_response("oscillation", 0.0, times)
        shifted = torch.clamp(times - onset_fraction * HORIZON, min=0.0)
        active = (times >= onset_fraction * HORIZON).to(DTYPE)
        envelope = 0.0035 * torch.exp(-0.45 * shifted) * torch.sin(1.2 * shifted) * active
        channel_scale = torch.linspace(0.8, 1.2, 6, dtype=DTYPE, device=DEVICE)
        return base + envelope[:, None] * channel_scale[None, :]
    raise ValueError(f"Unknown case: {case}")


def generate(
    case: str,
    repeat: int,
    split: str,
    sampling: str,
    offset_override: int | None = None,
):
    offset = offset_override if offset_override is not None else (
        61000 if split == "calibration" else 71000
    )
    seed = offset + 101 * repeat + sum(map(ord, case))
    times = make_times(seed, sampling)
    clean = response(case, times)
    rng = np.random.default_rng(seed + 17)
    observations = clean + NOISE_STD * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )
    train_np, diagnostic_np, windows_np = split_indices(times, seed + 29)
    return seed, times, observations, train_np, diagnostic_np, windows_np


def residual_statistics(
    prediction: torch.Tensor,
    observations: torch.Tensor,
    windows_np: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    lag1 = {}
    rmse = {}
    for name, index_np in windows_np.items():
        index = torch.tensor(index_np, dtype=torch.long, device=DEVICE)
        residual = prediction[index] - observations[index]
        lag1[name] = mean_lag1(residual)
        rmse[name] = float(torch.sqrt(torch.mean(residual.square())).cpu())
    return lag1, rmse


def calibration_fit(case: str, repeat: int) -> dict:
    seed, times, observations, train_np, diagnostic_np, windows_np = generate(
        case, repeat, "calibration", "regular"
    )
    train_idx = torch.tensor(train_np, dtype=torch.long, device=DEVICE)
    diagnostic_idx = torch.tensor(diagnostic_np, dtype=torch.long, device=DEVICE)
    fit = fit_rank(
        times,
        observations,
        train_idx,
        diagnostic_idx,
        rank=1,
        seed=seed * 100,
        adam_steps=145,
        lbfgs_steps=40,
    )
    lag1, rmse = residual_statistics(
        prediction_from_fit(times, fit), observations, windows_np
    )
    return {
        "case": case,
        "repeat": repeat,
        "seed": seed,
        "sampling": "regular",
        "window_residual_lag1": lag1,
        "window_rmse": rmse,
        "max_abs_lag1": max(abs(value) for value in lag1.values()),
    }


def joint_test_fit(case: str, repeat: int, seed_offset: int = 71000) -> dict:
    seed, times, observations, train_np, diagnostic_np, windows_np = generate(
        case, repeat, "evaluation", "jittered", offset_override=seed_offset
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
    lag1, rmse = residual_statistics(
        prediction_from_fit(times, winner), observations, windows_np
    )
    rank2 = next(item for item in fits if item.rank == 2)
    rank3 = next(item for item in fits if item.rank == 3)
    return {
        "case": case,
        "repeat": repeat,
        "seed": seed,
        "sampling": "jittered",
        "selected_rank": winner.rank,
        "validation_rmse": winner.val_rmse,
        "jacobian_condition": winner.jacobian_condition,
        "rank3_vs_rank2_bic_gain": rank2.bic - rank3.bic,
        "window_residual_lag1": lag1,
        "window_rmse": rmse,
        "terminal_abs_lag1": abs(lag1["late"]),
        "multiwindow_max_abs_lag1": max(abs(value) for value in lag1.values()),
        "strongest_window": max(lag1, key=lambda name: abs(lag1[name])),
        "candidate_bic": {str(item.rank): item.bic for item in fits},
    }


def calibrate(records: list[dict]) -> dict:
    pooled = np.asarray([
        abs(value)
        for record in records
        for value in record["window_residual_lag1"].values()
    ])
    quantile = 1.0 - ALPHA_FAMILYWISE / len(WINDOW_FRACTIONS)
    threshold = float(np.quantile(pooled, quantile, method="higher"))
    bootstrap_rng = np.random.default_rng(80819)
    bootstrap_thresholds = []
    for _ in range(2000):
        sample = bootstrap_rng.choice(pooled, size=pooled.size, replace=True)
        bootstrap_thresholds.append(float(np.quantile(sample, quantile, method="higher")))
    return {
        "method": "pooled regular-sampling zero controls; Bonferroni empirical quantile",
        "familywise_alpha": ALPHA_FAMILYWISE,
        "quantile": quantile,
        "threshold": threshold,
        "window_statistics": int(pooled.size),
        "independent_zero_control_fits": len(records),
        "bootstrap_threshold_interval95": [
            float(np.quantile(bootstrap_thresholds, 0.025)),
            float(np.quantile(bootstrap_thresholds, 0.975)),
        ],
        "calibration_max": float(np.max(pooled)),
    }


def apply_decisions(records: list[dict], threshold: float) -> None:
    for record in records:
        prediction_ok = record["validation_rmse"] <= max(4.0 * NOISE_STD, 3.0e-3)
        condition_ok = record["jacobian_condition"] <= 1.0e8
        rank_cap = record["selected_rank"] == 3 and record["rank3_vs_rank2_bic_gain"] >= 6.0
        common_ok = prediction_ok and condition_ok and not rank_cap
        record["terminal_decision"] = (
            "ACCEPT_CONTRACT"
            if common_ok and record["terminal_abs_lag1"] <= 0.55
            else "REFUSE_CONTRACT"
        )
        record["multiwindow_decision"] = (
            "ACCEPT_CONTRACT"
            if common_ok and record["multiwindow_max_abs_lag1"] <= threshold
            else "REFUSE_CONTRACT"
        )


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for case in sorted({record["case"] for record in records}):
        group = [record for record in records if record["case"] == case]
        terminal = sum(record["terminal_decision"] == "REFUSE_CONTRACT" for record in group)
        multi = sum(record["multiwindow_decision"] == "REFUSE_CONTRACT" for record in group)
        rows.append({
            "case": case,
            "trials": len(group),
            "terminal_refusal_fraction": terminal / len(group),
            "multiwindow_refusal_fraction": multi / len(group),
            "multiwindow_refusal_wilson95": wilson_interval(multi, len(group)),
            "elevated_rank_fraction": sum(record["selected_rank"] > 1 for record in group) / len(group),
            "median_max_abs_lag1": float(np.median([
                record["multiwindow_max_abs_lag1"] for record in group
            ])),
            "strongest_window_counts": {
                name: sum(record["strongest_window"] == name for record in group)
                for name in WINDOW_FRACTIONS
            },
        })
    return rows


def assess(summary: list[dict], calibration: dict) -> dict:
    lookup = {row["case"]: row for row in summary}
    zero_cases = ("oscillation_zero", "signed_zero")
    stress_cases = (
        "oscillation_decay_016",
        "oscillation_decay_036",
        "shifted_transient_020",
        "shifted_transient_055",
    )
    checks = {
        "at_least_100_independent_calibration_fits": (
            calibration["independent_zero_control_fits"] >= 100
        ),
        "jittered_zero_false_refusal_at_most_1_of_12": all(
            lookup[case]["multiwindow_refusal_fraction"] <= 1.0 / 12.0
            for case in zero_cases
        ),
        "each_unseen_stress_refused_at_least_5_of_6": all(
            lookup[case]["multiwindow_refusal_fraction"] >= 5.0 / 6.0
            for case in stress_cases
        ),
        "no_systematic_rank_absorption": all(
            lookup[case]["elevated_rank_fraction"] <= 1.0 / 3.0
            for case in stress_cases
        ),
        "threshold_bootstrap_width_at_most_0.15": (
            calibration["bootstrap_threshold_interval95"][1]
            - calibration["bootstrap_threshold_interval95"][0]
            <= 0.15
        ),
    }
    return {"checks": checks, "route_pass": all(checks.values())}


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "multiwindow_external_calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Multi-window external calibration and stress test",
        "",
        f"- Calibration fits: {payload['calibration']['independent_zero_control_fits']}",
        f"- Window statistics: {payload['calibration']['window_statistics']}",
        f"- Frozen threshold: {payload['calibration']['threshold']:.6f}",
        f"- Threshold bootstrap 95% interval: {payload['calibration']['bootstrap_threshold_interval95']}",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        "",
        "| Case | Trials | Terminal refusal | Multi-window refusal | Elevated rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['case']} | {row['trials']} | "
            f"{row['terminal_refusal_fraction']:.3f} | "
            f"{row['multiwindow_refusal_fraction']:.3f} | "
            f"{row['elevated_rank_fraction']:.3f} |"
        )
    lines.extend(["", "## Prespecified checks", ""])
    for key, value in payload["assessment"]["checks"].items():
        lines.append(f"- {key}: **{value}**")
    (RESULTS / "multiwindow_external_calibration.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    calibration_records = []
    for case in ("signed_zero", "oscillation_zero"):
        for repeat in range(CALIBRATION_REPEATS_PER_FAMILY):
            calibration_records.append(calibration_fit(case, repeat))
    calibration = calibrate(calibration_records)

    evaluation_records = []
    for case in ("signed_zero", "oscillation_zero"):
        for repeat in range(ZERO_TEST_REPEATS):
            evaluation_records.append(joint_test_fit(case, repeat))
    for case in (
        "oscillation_decay_016",
        "oscillation_decay_036",
        "shifted_transient_020",
        "shifted_transient_055",
    ):
        for repeat in range(STRESS_TEST_REPEATS):
            evaluation_records.append(joint_test_fit(case, repeat))
    apply_decisions(evaluation_records, calibration["threshold"])
    summary = summarize(evaluation_records)
    assessment = assess(summary, calibration)
    payload = {
        "experiment": "multiwindow_external_calibration",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "design": {
            "calibration_sampling": "regular",
            "evaluation_sampling": "jittered by up to 0.35 nominal time steps",
            "calibration_rank": 1,
            "evaluation_candidate_ranks": [1, 2, 3],
            "windows": WINDOW_FRACTIONS,
            "calibration_and_evaluation_seed_ranges_are_disjoint": True,
            "shifted_transient_is_an_output_level_contract_violation": True,
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
