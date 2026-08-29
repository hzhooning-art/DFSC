"""Compare stratified index-lag and time-gap-weighted residual calibration."""

from __future__ import annotations

import json

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank
from probe_out_of_class_refusal import mean_lag1, prediction_from_fit
from probe_refusal_calibration import RESULTS, wilson_interval
from probe_multiwindow_external_calibration import (
    ALPHA_FAMILYWISE,
    CALIBRATION_REPEATS_PER_FAMILY,
    NOISE_STD,
    STRESS_TEST_REPEATS,
    WINDOW_FRACTIONS,
    ZERO_TEST_REPEATS,
    generate,
)


CALIBRATION_SEED_OFFSET = 111000
EVALUATION_SEED_OFFSET = 121000
FIT_RMSE_LIMIT = max(4.0 * NOISE_STD, 3.0e-3)
FIT_CONDITION_LIMIT = 1.0e8
METHODS = ("index_lag1", "time_weighted_lag1")


def time_weighted_lag1(residual: torch.Tensor, times: torch.Tensor) -> float:
    """Adjacent residual correlation weighted by the actual time increments."""
    values = []
    pair_weight = torch.diff(times).clamp_min(torch.finfo(times.dtype).eps)
    pair_weight = pair_weight / pair_weight.sum()
    for channel in range(residual.shape[1]):
        x = residual[:-1, channel]
        y = residual[1:, channel]
        x = x - torch.sum(pair_weight * x)
        y = y - torch.sum(pair_weight * y)
        covariance = torch.sum(pair_weight * x * y)
        denominator = torch.sqrt(
            torch.sum(pair_weight * x.square())
            * torch.sum(pair_weight * y.square())
        ).clamp_min(1.0e-20)
        values.append(float((covariance / denominator).detach().cpu()))
    return float(np.mean(values))


def window_statistics(
    prediction: torch.Tensor,
    observations: torch.Tensor,
    times: torch.Tensor,
    windows_np: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    output = {method: {} for method in METHODS}
    for name, index_np in windows_np.items():
        index = torch.tensor(index_np, dtype=torch.long, device=DEVICE)
        residual = prediction[index] - observations[index]
        output["index_lag1"][name] = mean_lag1(residual)
        output["time_weighted_lag1"][name] = time_weighted_lag1(
            residual, times[index]
        )
    return output


def fit_case(case: str, repeat: int, split: str) -> dict:
    offset = CALIBRATION_SEED_OFFSET if split == "calibration" else EVALUATION_SEED_OFFSET
    seed, times, observations, train_np, diagnostic_np, windows_np = generate(
        case,
        repeat,
        split,
        "jittered",
        offset_override=offset,
    )
    train_idx = torch.tensor(train_np, dtype=torch.long, device=DEVICE)
    diagnostic_idx = torch.tensor(diagnostic_np, dtype=torch.long, device=DEVICE)
    ranks = (1,) if split == "calibration" else (1, 2, 3)
    fits = []
    for rank in ranks:
        candidates = [
            fit_rank(
                times,
                observations,
                train_idx,
                diagnostic_idx,
                rank=rank,
                seed=seed * 100 + start,
                adam_steps=170 if split == "calibration" else 190,
                lbfgs_steps=50 if split == "calibration" else 55,
            )
            for start in range(2)
        ]
        fits.append(min(candidates, key=lambda item: item.bic))
    winner = min(fits, key=lambda item: item.bic)
    statistics = window_statistics(
        prediction_from_fit(times, winner), observations, times, windows_np
    )
    quality_pass = (
        winner.val_rmse <= FIT_RMSE_LIMIT
        and winner.jacobian_condition <= FIT_CONDITION_LIMIT
        and np.isfinite(winner.val_rmse)
        and np.isfinite(winner.jacobian_condition)
    )
    record = {
        "split": split,
        "case": case,
        "repeat": repeat,
        "seed": seed,
        "sampling": "jittered",
        "fit_quality_pass": bool(quality_pass),
        "selected_rank": winner.rank,
        "validation_rmse": winner.val_rmse,
        "jacobian_condition": winner.jacobian_condition,
        "statistics": statistics,
        "max_abs_statistic": {
            method: max(abs(value) for value in statistics[method].values())
            for method in METHODS
        },
        "strongest_window": {
            method: max(statistics[method], key=lambda name: abs(statistics[method][name]))
            for method in METHODS
        },
        "candidate_bic": {str(item.rank): item.bic for item in fits},
    }
    if split == "evaluation":
        rank2 = next(item for item in fits if item.rank == 2)
        rank3 = next(item for item in fits if item.rank == 3)
        record["rank3_vs_rank2_bic_gain"] = rank2.bic - rank3.bic
    return record


def empirical_threshold(values: np.ndarray, seed: int) -> dict:
    quantile = 1.0 - ALPHA_FAMILYWISE / len(WINDOW_FRACTIONS)
    threshold = float(np.quantile(values, quantile, method="higher"))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(2000):
        sample = rng.choice(values, size=values.size, replace=True)
        boot.append(float(np.quantile(sample, quantile, method="higher")))
    return {
        "threshold": threshold,
        "quantile": quantile,
        "bootstrap_interval95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "bootstrap_width": float(np.quantile(boot, 0.975) - np.quantile(boot, 0.025)),
        "calibration_max": float(np.max(values)),
    }


def calibrate(records: list[dict]) -> dict:
    valid = [record for record in records if record["fit_quality_pass"]]
    methods = {}
    for method_index, method in enumerate(METHODS):
        pooled = np.asarray([
            abs(value)
            for record in valid
            for value in record["statistics"][method].values()
        ])
        methods[method] = empirical_threshold(pooled, 12819 + method_index)
        methods[method]["window_statistics"] = int(pooled.size)
    return {
        "sampling_stratum": "jittered",
        "familywise_alpha": ALPHA_FAMILYWISE,
        "independent_zero_control_fits": len(records),
        "valid_calibration_fits": len(valid),
        "invalid_calibration_fits": len(records) - len(valid),
        "invalid_calibration_fraction": (len(records) - len(valid)) / len(records),
        "methods": methods,
    }


def apply_decisions(records: list[dict], calibration: dict) -> None:
    for record in records:
        rank_cap = (
            record["selected_rank"] == 3
            and record["rank3_vs_rank2_bic_gain"] >= 6.0
        )
        common_ok = record["fit_quality_pass"] and not rank_cap
        record["decisions"] = {}
        for method in METHODS:
            passed = (
                record["max_abs_statistic"][method]
                <= calibration["methods"][method]["threshold"]
            )
            record["decisions"][method] = (
                "ACCEPT_CONTRACT" if common_ok and passed else "REFUSE_CONTRACT"
            )


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for case in sorted({record["case"] for record in records}):
        group = [record for record in records if record["case"] == case]
        row = {
            "case": case,
            "trials": len(group),
            "elevated_rank_fraction": sum(record["selected_rank"] > 1 for record in group) / len(group),
            "methods": {},
        }
        for method in METHODS:
            refused = sum(
                record["decisions"][method] == "REFUSE_CONTRACT" for record in group
            )
            row["methods"][method] = {
                "refusal_fraction": refused / len(group),
                "refusal_wilson95": wilson_interval(refused, len(group)),
                "median_max_abs_statistic": float(np.median([
                    record["max_abs_statistic"][method] for record in group
                ])),
                "strongest_window_counts": {
                    name: sum(
                        record["strongest_window"][method] == name for record in group
                    )
                    for name in WINDOW_FRACTIONS
                },
            }
        rows.append(row)
    return rows


def assess(summary: list[dict], calibration: dict) -> dict:
    lookup = {row["case"]: row for row in summary}
    zero_cases = ("signed_zero", "oscillation_zero")
    primary_stress = (
        "oscillation_decay_016",
        "shifted_transient_020",
        "shifted_transient_055",
    )
    methods = {}
    for method in METHODS:
        checks = {
            "zero_false_refusal_at_most_1_of_12": all(
                lookup[case]["methods"][method]["refusal_fraction"] <= 1.0 / 12.0
                for case in zero_cases
            ),
            "primary_stress_refused_at_least_5_of_6": all(
                lookup[case]["methods"][method]["refusal_fraction"] >= 5.0 / 6.0
                for case in primary_stress
            ),
            "bootstrap_width_at_most_0.15": (
                calibration["methods"][method]["bootstrap_width"] <= 0.15
            ),
        }
        methods[method] = {"checks": checks, "pass": all(checks.values())}
    shared_checks = {
        "at_least_100_independent_calibration_fits": (
            calibration["independent_zero_control_fits"] >= 100
        ),
        "invalid_calibration_fraction_at_most_0.02": (
            calibration["invalid_calibration_fraction"] <= 0.02
        ),
        "no_systematic_rank_absorption": all(
            lookup[case]["elevated_rank_fraction"] <= 1.0 / 3.0
            for case in primary_stress
        ),
    }
    return {
        "shared_checks": shared_checks,
        "methods": methods,
        "route_pass": all(shared_checks.values()) and any(
            result["pass"] for result in methods.values()
        ),
        "fast_decay_is_secondary_boundary_case": {
            method: lookup["oscillation_decay_036"]["methods"][method]["refusal_fraction"]
            for method in METHODS
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sampling_aware_residual_calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Sampling-aware residual calibration",
        "",
        f"- Calibration fits: {payload['calibration']['independent_zero_control_fits']}",
        f"- Invalid calibration fits: {payload['calibration']['invalid_calibration_fits']}",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        "",
    ]
    for method in METHODS:
        details = payload["calibration"]["methods"][method]
        lines.append(
            f"- {method}: threshold={details['threshold']:.6f}, "
            f"bootstrap95={details['bootstrap_interval95']}"
        )
    lines.extend([
        "",
        "| Case | Trials | Index refusal | Time-weighted refusal | Elevated rank |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in payload["summary"]:
        lines.append(
            f"| {row['case']} | {row['trials']} | "
            f"{row['methods']['index_lag1']['refusal_fraction']:.3f} | "
            f"{row['methods']['time_weighted_lag1']['refusal_fraction']:.3f} | "
            f"{row['elevated_rank_fraction']:.3f} |"
        )
    (RESULTS / "sampling_aware_residual_calibration.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    calibration_records = []
    for case in ("signed_zero", "oscillation_zero"):
        for repeat in range(CALIBRATION_REPEATS_PER_FAMILY):
            calibration_records.append(fit_case(case, repeat, "calibration"))
    calibration = calibrate(calibration_records)

    evaluation_records = []
    for case in ("signed_zero", "oscillation_zero"):
        for repeat in range(ZERO_TEST_REPEATS):
            evaluation_records.append(fit_case(case, repeat, "evaluation"))
    for case in (
        "oscillation_decay_016",
        "oscillation_decay_036",
        "shifted_transient_020",
        "shifted_transient_055",
    ):
        for repeat in range(STRESS_TEST_REPEATS):
            evaluation_records.append(fit_case(case, repeat, "evaluation"))
    apply_decisions(evaluation_records, calibration)
    summary = summarize(evaluation_records)
    assessment = assess(summary, calibration)
    payload = {
        "experiment": "sampling_aware_residual_calibration",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "design": {
            "calibration_and_evaluation_sampling": "independently jittered",
            "jitter": "uniform up to 0.35 nominal time steps",
            "calibration_rank": 1,
            "calibration_starts": 2,
            "evaluation_candidate_ranks": [1, 2, 3],
            "evaluation_starts_per_rank": 2,
            "primary_stress_excludes_known_fast_decay_boundary": True,
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
