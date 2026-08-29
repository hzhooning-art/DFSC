"""Evaluate exact-observation-design parametric-bootstrap refusal calibration."""

from __future__ import annotations

import json

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank
from probe_out_of_class_refusal import mean_lag1, prediction_from_fit
from probe_refusal_calibration import RESULTS, wilson_interval
from probe_multiwindow_external_calibration import NOISE_STD, response
from probe_sampling_process_stress import FIT_CONDITION_LIMIT, FIT_RMSE_LIMIT, split_indices
from probe_cluster_geometry_conditional_null import (
    SOURCE_ARTIFACT,
    geometry_features,
    make_variable_clustered_times,
)


OUTER_REPEATS_PER_FAMILY = 15
BOOTSTRAP_REPLICATES = 19
OUTER_SEED_OFFSET = 271000
MONTE_CARLO_ALPHA = 0.05


def seed_for(case: str, repeat: int) -> int:
    return OUTER_SEED_OFFSET + 101 * repeat + sum(map(ord, case))


def fit_observations(
    times: torch.Tensor,
    observations: torch.Tensor,
    train_np: np.ndarray,
    diagnostic_np: np.ndarray,
    windows_np: dict[str, np.ndarray],
    seed: int,
) -> tuple[dict, torch.Tensor]:
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
    return ({
        "validation_rmse": winner.val_rmse,
        "jacobian_condition": winner.jacobian_condition,
        "fit_quality_pass": bool(quality_pass),
        "statistics": statistics,
        "max_abs_statistic": max(abs(value) for value in statistics.values()),
        "strongest_window": max(statistics, key=lambda name: abs(statistics[name])),
    }, prediction.detach())


def evaluate_outer(case: str, repeat: int, legacy: float, global_threshold: float) -> dict:
    seed = seed_for(case, repeat)
    times = make_variable_clustered_times(seed)
    clean = response(case, times)
    rng = np.random.default_rng(seed + 17)
    observations = clean + NOISE_STD * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )
    train_np, diagnostic_np, windows_np = split_indices(times, seed + 29)
    outer, fitted_null = fit_observations(
        times, observations, train_np, diagnostic_np, windows_np, seed
    )

    bootstrap = []
    for bootstrap_index in range(BOOTSTRAP_REPLICATES):
        bootstrap_seed = seed * 1000 + 100 + bootstrap_index
        bootstrap_rng = np.random.default_rng(bootstrap_seed)
        bootstrap_observations = fitted_null + NOISE_STD * torch.tensor(
            bootstrap_rng.standard_normal(fitted_null.shape), dtype=DTYPE, device=DEVICE
        )
        fitted, _ = fit_observations(
            times,
            bootstrap_observations,
            train_np,
            diagnostic_np,
            windows_np,
            bootstrap_seed,
        )
        fitted["bootstrap_index"] = bootstrap_index
        fitted["seed"] = bootstrap_seed
        bootstrap.append(fitted)

    valid_bootstrap = [record for record in bootstrap if record["fit_quality_pass"]]
    calibration_pass = outer["fit_quality_pass"] and len(valid_bootstrap) == BOOTSTRAP_REPLICATES
    bootstrap_statistics = np.asarray([
        record["max_abs_statistic"] for record in valid_bootstrap
    ])
    if calibration_pass:
        exceedances = int(np.sum(bootstrap_statistics >= outer["max_abs_statistic"]))
        monte_carlo_p = (1 + exceedances) / (BOOTSTRAP_REPLICATES + 1)
        conditional_threshold = float(np.max(bootstrap_statistics))
        bootstrap_decision = (
            "REFUSE_CONTRACT" if monte_carlo_p <= MONTE_CARLO_ALPHA else "ACCEPT_CONTRACT"
        )
    else:
        exceedances = None
        monte_carlo_p = None
        conditional_threshold = None
        bootstrap_decision = "REFUSE_CALIBRATION"

    return {
        "case": case,
        "repeat": repeat,
        "seed": seed,
        "observation_count": int(len(times)),
        "training_count": int(len(train_np)),
        "geometry": geometry_features(times, windows_np),
        "outer_fit": outer,
        "bootstrap_replicates": bootstrap,
        "valid_bootstrap_replicates": len(valid_bootstrap),
        "calibration_pass": bool(calibration_pass),
        "conditional_threshold": conditional_threshold,
        "exceedances": exceedances,
        "monte_carlo_p": monte_carlo_p,
        "decisions": {
            "legacy_clustered": (
                "ACCEPT_CONTRACT"
                if outer["fit_quality_pass"] and outer["max_abs_statistic"] <= legacy
                else "REFUSE_CONTRACT"
            ),
            "new_global_clustered": (
                "ACCEPT_CONTRACT"
                if outer["fit_quality_pass"] and outer["max_abs_statistic"] <= global_threshold
                else "REFUSE_CONTRACT"
            ),
            "exact_design_bootstrap": bootstrap_decision,
        },
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for case in ("all_zero_controls", "signed_zero", "oscillation_zero"):
        group = records if case == "all_zero_controls" else [
            record for record in records if record["case"] == case
        ]
        methods = {}
        for method in ("legacy_clustered", "new_global_clustered", "exact_design_bootstrap"):
            refused = sum(record["decisions"][method] != "ACCEPT_CONTRACT" for record in group)
            methods[method] = {
                "refusals": refused,
                "fraction": refused / len(group),
                "wilson95": wilson_interval(refused, len(group)),
            }
        rows.append({
            "case": case,
            "trials": len(group),
            "outer_invalid_fits": sum(not record["outer_fit"]["fit_quality_pass"] for record in group),
            "conditional_calibration_failures": sum(not record["calibration_pass"] for record in group),
            "median_conditional_threshold": float(np.median([
                record["conditional_threshold"]
                for record in group if record["conditional_threshold"] is not None
            ])),
            "methods": methods,
        })
    return rows


def assess(records: list[dict], summary: list[dict]) -> dict:
    overall = next(row for row in summary if row["case"] == "all_zero_controls")
    bootstrap_result = overall["methods"]["exact_design_bootstrap"]
    checks = {
        "all_30_outer_fits_valid": all(
            record["outer_fit"]["fit_quality_pass"] for record in records
        ),
        "all_570_bootstrap_fits_valid": all(
            record["valid_bootstrap_replicates"] == BOOTSTRAP_REPLICATES
            for record in records
        ),
        "heldout_false_refusals_at_most_1_of_30": bootstrap_result["refusals"] <= 1,
        "heldout_wilson_upper_at_most_0.20": bootstrap_result["wilson95"][1] <= 0.20,
        "neither_family_has_more_than_1_false_refusal": all(
            row["methods"]["exact_design_bootstrap"]["refusals"] <= 1
            for row in summary if row["case"] != "all_zero_controls"
        ),
    }
    return {"checks": checks, "route_pass": all(checks.values())}


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "exact_design_conditional_bootstrap.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Exact-design conditional bootstrap",
        "",
        f"- Bootstrap replicates per outer fit: {BOOTSTRAP_REPLICATES}",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        "",
        "| Case | Trials | Calibration failures | Legacy FR | Global FR | Bootstrap FR | Median conditional threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['case']} | {row['trials']} | {row['conditional_calibration_failures']} | "
            f"{row['methods']['legacy_clustered']['refusals']} | "
            f"{row['methods']['new_global_clustered']['refusals']} | "
            f"{row['methods']['exact_design_bootstrap']['refusals']} | "
            f"{row['median_conditional_threshold']:.6f} |"
        )
    (RESULTS / "exact_design_conditional_bootstrap.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    source = json.loads(SOURCE_ARTIFACT.read_text(encoding="utf-8"))
    legacy = float(source["calibration"]["clustered"]["threshold"])
    geometry_artifact = json.loads(
        (RESULTS / "cluster_geometry_conditional_null.json").read_text(encoding="utf-8")
    )
    global_threshold = float(geometry_artifact["global_calibration"]["threshold"])
    records = [
        evaluate_outer(case, repeat, legacy, global_threshold)
        for case in ("signed_zero", "oscillation_zero")
        for repeat in range(OUTER_REPEATS_PER_FAMILY)
    ]
    summary = summarize(records)
    assessment = assess(records, summary)
    payload = {
        "experiment": "exact_design_conditional_bootstrap",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "design": {
            "outer_zero_fits": len(records),
            "outer_repeats_per_family": OUTER_REPEATS_PER_FAMILY,
            "bootstrap_replicates_per_outer_fit": BOOTSTRAP_REPLICATES,
            "total_bootstrap_fits": len(records) * BOOTSTRAP_REPLICATES,
            "monte_carlo_alpha": MONTE_CARLO_ALPHA,
            "finite_sample_p_value": "(1 + count(T_boot >= T_observed)) / (B + 1)",
            "exact_design_preserved": [
                "observation_times",
                "training_split",
                "diagnostic_windows",
                "noise_level",
            ],
            "candidate_rank": 1,
            "starts_per_fit": 2,
            "legacy_clustered_threshold": legacy,
            "new_global_clustered_threshold": global_threshold,
            "power_evaluation_deferred": True,
        },
        "records": records,
        "summary": summary,
        "assessment": assessment,
    }
    write_outputs(payload)
    print(json.dumps({
        "summary": summary,
        "assessment": assessment,
        "conditional_threshold_range": [
            min(record["conditional_threshold"] for record in records),
            max(record["conditional_threshold"] for record in records),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
