"""Calibrate and test a prespecified multi-window residual refusal gate.

The calibration split contains only in-class zero controls.  Its observations
are never reused in the held-out evaluation split.  This keeps the diagnostic
from being tuned directly on the H=18 oscillatory failure that motivated it.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank
from probe_out_of_class_refusal import mean_lag1, prediction_from_fit
from probe_refusal_calibration import RESULTS, wilson_interval
from probe_joint_horizon_noise_effects import clean_response


HORIZON = 18.0
NOISE_STD = 6.0e-4
NUM_POINTS = 124
CALIBRATION_REPEATS = 6
TEST_REPEATS = 8
ALPHA_FAMILYWISE = 0.10
LEVELS = {
    "signed_residue": {"zero": 0.0, "above": 0.0075},
    "oscillation": {"zero": 0.0, "above": 0.04},
}


def diagnostic_windows(num_points: int) -> dict[str, np.ndarray]:
    """Return three prespecified, disjoint diagnostic windows."""
    fractions = {
        "early": (0.15, 0.35),
        "middle": (0.45, 0.65),
        "late": (0.80, 1.00),
    }
    return {
        name: np.arange(int(np.floor(lo * num_points)), int(np.floor(hi * num_points)))
        for name, (lo, hi) in fractions.items()
    }


def generate_case(family: str, strength: float, seed: int):
    rng = np.random.default_rng(seed)
    times = torch.linspace(0.0, HORIZON, NUM_POINTS, dtype=DTYPE, device=DEVICE)
    clean = clean_response(family, strength, times)
    observations = clean + NOISE_STD * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )

    windows_np = diagnostic_windows(NUM_POINTS)
    diagnostic_np = np.unique(np.concatenate(list(windows_np.values())))
    training_pool = np.setdiff1d(np.arange(1, NUM_POINTS), diagnostic_np)
    train_size = min(48, training_pool.size)
    train_np = np.sort(rng.choice(training_pool, size=train_size, replace=False))
    diagnostic_idx = torch.tensor(diagnostic_np, dtype=torch.long, device=DEVICE)
    return times, observations, train_np, diagnostic_idx, windows_np


def evaluate_raw(family: str, level: str, strength: float, repeat: int, split: str) -> dict:
    split_offset = 41000 if split == "calibration" else 51000
    seed = split_offset + 101 * repeat + sum(map(ord, family))
    times, observations, train_np, diagnostic_idx, windows_np = generate_case(
        family, strength, seed
    )
    train_idx = torch.tensor(train_np, dtype=torch.long, device=DEVICE)

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
    window_lag1 = {}
    window_rmse = {}
    for name, index_np in windows_np.items():
        index = torch.tensor(index_np, dtype=torch.long, device=DEVICE)
        residual = prediction[index] - observations[index]
        window_lag1[name] = mean_lag1(residual)
        window_rmse[name] = float(torch.sqrt(torch.mean(residual.square())).cpu())

    rank2 = next(fit for fit in fits if fit.rank == 2)
    rank3 = next(fit for fit in fits if fit.rank == 3)
    return {
        "split": split,
        "family": family,
        "level": level,
        "strength": strength,
        "repeat": repeat,
        "seed": seed,
        "selected_rank": winner.rank,
        "validation_rmse": winner.val_rmse,
        "jacobian_condition": winner.jacobian_condition,
        "rank3_vs_rank2_bic_gain": rank2.bic - rank3.bic,
        "window_residual_lag1": window_lag1,
        "window_rmse": window_rmse,
        "terminal_abs_lag1": abs(window_lag1["late"]),
        "multiwindow_max_abs_lag1": max(abs(value) for value in window_lag1.values()),
        "candidate_bic": {str(fit.rank): fit.bic for fit in fits},
    }


def calibrate_threshold(records: list[dict]) -> dict:
    pooled = np.asarray([
        abs(value)
        for record in records
        for value in record["window_residual_lag1"].values()
    ])
    per_window_alpha = ALPHA_FAMILYWISE / 3.0
    quantile = 1.0 - per_window_alpha
    threshold = float(np.quantile(pooled, quantile, method="higher"))
    return {
        "method": "pooled-zero-control Bonferroni empirical quantile",
        "familywise_alpha": ALPHA_FAMILYWISE,
        "windows": 3,
        "per_window_alpha": per_window_alpha,
        "quantile": quantile,
        "threshold": threshold,
        "calibration_window_statistics": int(pooled.size),
        "calibration_max": float(np.max(pooled)),
    }


def apply_decisions(records: list[dict], threshold: float) -> None:
    for record in records:
        prediction_ok = record["validation_rmse"] <= max(4.0 * NOISE_STD, 3.0e-3)
        condition_ok = record["jacobian_condition"] <= 1.0e8
        rank_cap = (
            record["selected_rank"] == 3
            and record["rank3_vs_rank2_bic_gain"] >= 6.0
        )
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
    for family in LEVELS:
        for level in LEVELS[family]:
            group = [
                row for row in records
                if row["family"] == family and row["level"] == level
            ]
            terminal_refused = sum(
                row["terminal_decision"] == "REFUSE_CONTRACT" for row in group
            )
            multi_refused = sum(
                row["multiwindow_decision"] == "REFUSE_CONTRACT" for row in group
            )
            absorbed = sum(
                row["multiwindow_decision"] == "ACCEPT_CONTRACT"
                and row["selected_rank"] > 1
                for row in group
            )
            rows.append({
                "family": family,
                "level": level,
                "trials": len(group),
                "terminal_refusal_fraction": terminal_refused / len(group),
                "terminal_refusal_wilson95": wilson_interval(terminal_refused, len(group)),
                "multiwindow_refusal_fraction": multi_refused / len(group),
                "multiwindow_refusal_wilson95": wilson_interval(multi_refused, len(group)),
                "multi_minus_terminal_refusal": (
                    multi_refused - terminal_refused
                ) / len(group),
                "elevated_rank_fraction": sum(
                    row["selected_rank"] > 1 for row in group
                ) / len(group),
                "accepted_elevated_rank_fraction": absorbed / len(group),
                "median_terminal_abs_lag1": float(np.median([
                    row["terminal_abs_lag1"] for row in group
                ])),
                "median_multiwindow_max_abs_lag1": float(np.median([
                    row["multiwindow_max_abs_lag1"] for row in group
                ])),
            })
    return rows


def assess(summary: list[dict]) -> dict:
    lookup = {(row["family"], row["level"]): row for row in summary}
    zero_ok = all(
        lookup[(family, "zero")]["multiwindow_refusal_fraction"] <= 0.25
        for family in LEVELS
    )
    oscill_gain = lookup[("oscillation", "above")]["multi_minus_terminal_refusal"]
    signed_change = lookup[("signed_residue", "above")]["multi_minus_terminal_refusal"]
    absorption_ok = all(
        lookup[(family, "above")]["accepted_elevated_rank_fraction"] <= 0.25
        for family in LEVELS
    )
    checks = {
        "heldout_zero_false_refusal_at_most_0.25": zero_ok,
        "oscillation_detection_gain_at_least_0.50": oscill_gain >= 0.50,
        "signed_detection_not_degraded": signed_change >= 0.0,
        "accepted_elevated_rank_absorption_at_most_0.25": absorption_ok,
    }
    return {"checks": checks, "route_pass": all(checks.values())}


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "calibrated_multiwindow_refusal.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Calibrated multi-window refusal probe",
        "",
        f"- Horizon: {HORIZON:g}",
        f"- Noise std: {NOISE_STD:.1e}",
        "- Diagnostic windows: early [15%,35%), middle [45%,65%), late [80%,100%)",
        f"- Calibrated lag-1 threshold: {payload['calibration']['threshold']:.6f}",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        "",
        "| Family | Level | Terminal refusal | Multi-window refusal | Change |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['family']} | {row['level']} | "
            f"{row['terminal_refusal_fraction']:.3f} | "
            f"{row['multiwindow_refusal_fraction']:.3f} | "
            f"{row['multi_minus_terminal_refusal']:+.3f} |"
        )
    lines.extend(["", "## Checks", ""])
    for key, value in payload["assessment"]["checks"].items():
        lines.append(f"- {key}: **{value}**")
    (RESULTS / "calibrated_multiwindow_refusal.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    calibration = []
    for family in LEVELS:
        for repeat in range(CALIBRATION_REPEATS):
            calibration.append(evaluate_raw(
                family, "zero", LEVELS[family]["zero"], repeat, "calibration"
            ))
    calibration_info = calibrate_threshold(calibration)

    evaluation = []
    for family, levels in LEVELS.items():
        for level, strength in levels.items():
            for repeat in range(TEST_REPEATS):
                evaluation.append(evaluate_raw(
                    family, level, strength, repeat, "evaluation"
                ))
    apply_decisions(evaluation, calibration_info["threshold"])
    summary = summarize(evaluation)
    payload = {
        "experiment": "calibrated_multiwindow_refusal",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "design": {
            "calibration_repeats_per_family": CALIBRATION_REPEATS,
            "test_repeats_per_family_level": TEST_REPEATS,
            "training_points": 48,
            "diagnostic_windows": {
                name: index.tolist()
                for name, index in diagnostic_windows(NUM_POINTS).items()
            },
            "separation": "calibration and evaluation use disjoint seed ranges",
        },
        "calibration": calibration_info,
        "calibration_records": calibration,
        "evaluation_records": evaluation,
        "summary": summary,
        "assessment": assess(summary),
    }
    write_outputs(payload)
    print(json.dumps({
        "calibration": calibration_info,
        "summary": summary,
        "assessment": payload["assessment"],
    }, indent=2))


if __name__ == "__main__":
    main()
