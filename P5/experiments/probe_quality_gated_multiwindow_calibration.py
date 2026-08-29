"""Quality-gated null calibration with a fresh external evaluation split."""

from __future__ import annotations

import json

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank
from probe_out_of_class_refusal import prediction_from_fit
from probe_refusal_calibration import RESULTS
from probe_multiwindow_external_calibration import (
    ALPHA_FAMILYWISE,
    CALIBRATION_REPEATS_PER_FAMILY,
    NOISE_STD,
    STRESS_TEST_REPEATS,
    WINDOW_FRACTIONS,
    ZERO_TEST_REPEATS,
    apply_decisions,
    assess,
    generate,
    joint_test_fit,
    residual_statistics,
    summarize,
)


CALIBRATION_SEED_OFFSET = 91000
EVALUATION_SEED_OFFSET = 101000
FIT_RMSE_LIMIT = max(4.0 * NOISE_STD, 3.0e-3)
FIT_CONDITION_LIMIT = 1.0e8


def quality_gated_calibration_fit(case: str, repeat: int) -> dict:
    seed, times, observations, train_np, diagnostic_np, windows_np = generate(
        case,
        repeat,
        "calibration",
        "regular",
        offset_override=CALIBRATION_SEED_OFFSET,
    )
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
            adam_steps=170,
            lbfgs_steps=50,
        )
        for start in range(2)
    ]
    fit = min(candidates, key=lambda item: item.bic)
    lag1, rmse = residual_statistics(
        prediction_from_fit(times, fit), observations, windows_np
    )
    fit_quality_pass = (
        fit.val_rmse <= FIT_RMSE_LIMIT
        and fit.jacobian_condition <= FIT_CONDITION_LIMIT
        and np.isfinite(fit.val_rmse)
        and np.isfinite(fit.jacobian_condition)
    )
    return {
        "case": case,
        "repeat": repeat,
        "seed": seed,
        "sampling": "regular",
        "fit_quality_pass": bool(fit_quality_pass),
        "validation_rmse": fit.val_rmse,
        "jacobian_condition": fit.jacobian_condition,
        "candidate_bic": [candidate.bic for candidate in candidates],
        "window_residual_lag1": lag1,
        "window_rmse": rmse,
        "max_abs_lag1": max(abs(value) for value in lag1.values()),
    }


def calibrate(records: list[dict]) -> dict:
    valid = [record for record in records if record["fit_quality_pass"]]
    pooled = np.asarray([
        abs(value)
        for record in valid
        for value in record["window_residual_lag1"].values()
    ])
    quantile = 1.0 - ALPHA_FAMILYWISE / len(WINDOW_FRACTIONS)
    threshold = float(np.quantile(pooled, quantile, method="higher"))
    bootstrap_rng = np.random.default_rng(90819)
    bootstrap_thresholds = []
    for _ in range(2000):
        sample = bootstrap_rng.choice(pooled, size=pooled.size, replace=True)
        bootstrap_thresholds.append(float(np.quantile(sample, quantile, method="higher")))
    return {
        "method": "quality-gated two-start zero controls; Bonferroni empirical quantile",
        "familywise_alpha": ALPHA_FAMILYWISE,
        "quantile": quantile,
        "threshold": threshold,
        "independent_zero_control_fits": len(records),
        "valid_calibration_fits": len(valid),
        "invalid_calibration_fits": len(records) - len(valid),
        "invalid_calibration_fraction": (len(records) - len(valid)) / len(records),
        "window_statistics": int(pooled.size),
        "bootstrap_threshold_interval95": [
            float(np.quantile(bootstrap_thresholds, 0.025)),
            float(np.quantile(bootstrap_thresholds, 0.975)),
        ],
        "calibration_max": float(np.max(pooled)),
        "fit_quality_limits": {
            "validation_rmse": FIT_RMSE_LIMIT,
            "jacobian_condition": FIT_CONDITION_LIMIT,
        },
    }


def local_assess(summary: list[dict], calibration: dict) -> dict:
    base = assess(summary, calibration)
    base["checks"]["invalid_calibration_fraction_at_most_0.02"] = (
        calibration["invalid_calibration_fraction"] <= 0.02
    )
    base["route_pass"] = all(base["checks"].values())
    return base


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "quality_gated_multiwindow_calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Quality-gated multi-window calibration",
        "",
        f"- Calibration fits: {payload['calibration']['independent_zero_control_fits']}",
        f"- Invalid calibration fits: {payload['calibration']['invalid_calibration_fits']}",
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
    (RESULTS / "quality_gated_multiwindow_calibration.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    calibration_records = []
    for case in ("signed_zero", "oscillation_zero"):
        for repeat in range(CALIBRATION_REPEATS_PER_FAMILY):
            calibration_records.append(quality_gated_calibration_fit(case, repeat))
    calibration = calibrate(calibration_records)

    evaluation_records = []
    for case in ("signed_zero", "oscillation_zero"):
        for repeat in range(ZERO_TEST_REPEATS):
            evaluation_records.append(joint_test_fit(
                case, repeat, seed_offset=EVALUATION_SEED_OFFSET
            ))
    for case in (
        "oscillation_decay_016",
        "oscillation_decay_036",
        "shifted_transient_020",
        "shifted_transient_055",
    ):
        for repeat in range(STRESS_TEST_REPEATS):
            evaluation_records.append(joint_test_fit(
                case, repeat, seed_offset=EVALUATION_SEED_OFFSET
            ))
    apply_decisions(evaluation_records, calibration["threshold"])
    summary = summarize(evaluation_records)
    assessment = local_assess(summary, calibration)
    payload = {
        "experiment": "quality_gated_multiwindow_calibration",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "design": {
            "calibration_sampling": "regular",
            "evaluation_sampling": "jittered by up to 0.35 nominal time steps",
            "calibration_starts": 2,
            "calibration_rank": 1,
            "evaluation_candidate_ranks": [1, 2, 3],
            "evaluation_starts_per_rank": 2,
            "quality_gate_precedes_statistical_calibration": True,
            "fresh_seed_ranges_after_failed_ungated_probe": True,
            "windows": WINDOW_FRACTIONS,
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
