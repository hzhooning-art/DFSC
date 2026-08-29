"""Replicate exact-design bootstrap inference with a frozen retry-on-failure rule."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank
from probe_out_of_class_refusal import mean_lag1, prediction_from_fit
from probe_refusal_calibration import RESULTS, wilson_interval
from probe_multiwindow_external_calibration import NOISE_STD, response
from probe_sampling_process_stress import FIT_CONDITION_LIMIT, FIT_RMSE_LIMIT, split_indices
from probe_cluster_geometry_conditional_null import geometry_features, make_variable_clustered_times
from probe_exact_design_conditional_bootstrap import BOOTSTRAP_REPLICATES, MONTE_CARLO_ALPHA


NULL_CASES = ("signed_zero", "oscillation_zero")
PRIMARY_CASES = (
    "oscillation_decay_016",
    "shifted_transient_020",
    "shifted_transient_055",
)
NULL_REPEATS = 10
PRIMARY_REPEATS = 4
SEED_OFFSET = 351000
INITIAL_STARTS = 2
RETRY_STARTS = 4


def seed_for(case: str, repeat: int) -> int:
    cases = NULL_CASES + PRIMARY_CASES
    return SEED_OFFSET + 2003 * cases.index(case) + 101 * repeat


def quality_pass(fit) -> bool:
    return bool(
        np.isfinite(fit.val_rmse)
        and np.isfinite(fit.jacobian_condition)
        and fit.val_rmse <= FIT_RMSE_LIMIT
        and fit.jacobian_condition <= FIT_CONDITION_LIMIT
    )


def fit_with_frozen_retry(
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
        for start in range(INITIAL_STARTS)
    ]
    initial_winner = min(candidates, key=lambda item: item.bic)
    retry_triggered = not quality_pass(initial_winner)
    if retry_triggered:
        candidates.extend(
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
            for start in range(INITIAL_STARTS, INITIAL_STARTS + RETRY_STARTS)
        )

    winner = min(candidates, key=lambda item: item.bic)
    prediction = prediction_from_fit(times, winner)
    statistics = {}
    for name, index_np in windows_np.items():
        index = torch.tensor(index_np, dtype=torch.long, device=DEVICE)
        statistics[name] = mean_lag1(prediction[index] - observations[index])

    return ({
        "validation_rmse": winner.val_rmse,
        "jacobian_condition": winner.jacobian_condition,
        "fit_quality_pass": quality_pass(winner),
        "statistics": statistics,
        "max_abs_statistic": max(abs(value) for value in statistics.values()),
        "strongest_window": max(statistics, key=lambda name: abs(statistics[name])),
        "retry_triggered": retry_triggered,
        "starts_evaluated": len(candidates),
        "initial_validation_rmse": initial_winner.val_rmse,
        "initial_jacobian_condition": initial_winner.jacobian_condition,
        "initial_fit_quality_pass": quality_pass(initial_winner),
        "retry_recovered_fit": retry_triggered and quality_pass(winner),
    }, prediction.detach())


def evaluate_case(case: str, repeat: int) -> dict:
    started = time.perf_counter()
    seed = seed_for(case, repeat)
    times = make_variable_clustered_times(seed)
    clean = response(case, times)
    rng = np.random.default_rng(seed + 17)
    observations = clean + NOISE_STD * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )
    train_np, diagnostic_np, windows_np = split_indices(times, seed + 29)
    outer, fitted_null = fit_with_frozen_retry(
        times, observations, train_np, diagnostic_np, windows_np, seed
    )

    bootstrap = []
    for bootstrap_index in range(BOOTSTRAP_REPLICATES):
        bootstrap_seed = seed * 1000 + 100 + bootstrap_index
        bootstrap_rng = np.random.default_rng(bootstrap_seed)
        bootstrap_observations = fitted_null + NOISE_STD * torch.tensor(
            bootstrap_rng.standard_normal(fitted_null.shape), dtype=DTYPE, device=DEVICE
        )
        fitted, _ = fit_with_frozen_retry(
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
    if calibration_pass:
        bootstrap_statistics = np.asarray([
            record["max_abs_statistic"] for record in valid_bootstrap
        ])
        exceedances = int(np.sum(bootstrap_statistics >= outer["max_abs_statistic"]))
        monte_carlo_p = (1 + exceedances) / (BOOTSTRAP_REPLICATES + 1)
        conditional_threshold = float(np.max(bootstrap_statistics))
        decision = "REFUSE_CONTRACT" if monte_carlo_p <= MONTE_CARLO_ALPHA else "ACCEPT_CONTRACT"
    else:
        exceedances = None
        monte_carlo_p = None
        conditional_threshold = None
        decision = "REFUSE_CALIBRATION"

    return {
        "case": case,
        "role": "null" if case in NULL_CASES else "primary_alternative",
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
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for case in NULL_CASES + PRIMARY_CASES:
        group = [record for record in records if record["case"] == case]
        refused = sum(record["decision"] == "REFUSE_CONTRACT" for record in group)
        calibration_failures = sum(record["decision"] == "REFUSE_CALIBRATION" for record in group)
        fit_records = [record["outer_fit"] for record in group]
        fit_records.extend(
            fit for record in group for fit in record["bootstrap_replicates"]
        )
        rows.append({
            "case": case,
            "role": group[0]["role"],
            "trials": len(group),
            "contract_refusals": refused,
            "refusal_fraction": refused / len(group),
            "refusal_wilson95": wilson_interval(refused, len(group)),
            "calibration_failures": calibration_failures,
            "retry_triggers": sum(fit["retry_triggered"] for fit in fit_records),
            "retry_recoveries": sum(fit["retry_recovered_fit"] for fit in fit_records),
            "total_fits": len(fit_records),
            "median_elapsed_seconds": float(np.median([
                record["elapsed_seconds"] for record in group
            ])),
        })
    return rows


def assess(records: list[dict], summary: list[dict]) -> dict:
    null_rows = [row for row in summary if row["role"] == "null"]
    primary_rows = [row for row in summary if row["role"] == "primary_alternative"]
    all_fits = [record["outer_fit"] for record in records]
    all_fits.extend(fit for record in records for fit in record["bootstrap_replicates"])
    null_refusals = sum(row["contract_refusals"] for row in null_rows)
    null_trials = sum(row["trials"] for row in null_rows)
    null_interval = wilson_interval(null_refusals, null_trials)
    retry_triggers = sum(fit["retry_triggered"] for fit in all_fits)
    checks = {
        "all_outer_and_bootstrap_fits_valid_after_frozen_retry": all(
            fit["fit_quality_pass"] for fit in all_fits
        ),
        "no_calibration_failures": all(record["calibration_pass"] for record in records),
        "zero_false_refusals_among_20_null_controls": null_refusals == 0,
        "null_wilson_upper_at_most_0.20": null_interval[1] <= 0.20,
        "each_primary_case_refused_at_least_3_of_4": all(
            row["contract_refusals"] >= 3 for row in primary_rows
        ),
        "retry_trigger_rate_at_most_0.02": retry_triggers / len(all_fits) <= 0.02,
    }
    return {
        "checks": checks,
        "route_pass": all(checks.values()),
        "null_refusals": null_refusals,
        "null_trials": null_trials,
        "null_wilson95": null_interval,
        "total_fits": len(all_fits),
        "retry_triggers": retry_triggers,
        "retry_trigger_fraction": retry_triggers / len(all_fits),
        "retry_recoveries": sum(fit["retry_recovered_fit"] for fit in all_fits),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "quality_triggered_retry_replication.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Quality-triggered retry replication",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        f"- Total runtime: {payload['runtime']['total_elapsed_seconds']:.1f} s",
        f"- Retry triggers: {payload['assessment']['retry_triggers']}/{payload['assessment']['total_fits']}",
        "",
        "| Case | Role | Trials | Contract refusals | Calibration failures | Retry triggers | Retry recoveries | Median seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['case']} | {row['role']} | {row['trials']} | "
            f"{row['contract_refusals']} | {row['calibration_failures']} | "
            f"{row['retry_triggers']} | {row['retry_recoveries']} | "
            f"{row['median_elapsed_seconds']:.2f} |"
        )
    (RESULTS / "quality_triggered_retry_replication.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    records = [
        evaluate_case(case, repeat)
        for case in NULL_CASES
        for repeat in range(NULL_REPEATS)
    ]
    records.extend(
        evaluate_case(case, repeat)
        for case in PRIMARY_CASES
        for repeat in range(PRIMARY_REPEATS)
    )
    summary = summarize(records)
    assessment = assess(records, summary)
    payload = {
        "experiment": "quality_triggered_retry_replication",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "design": {
            "null_cases": list(NULL_CASES),
            "primary_cases": list(PRIMARY_CASES),
            "null_repeats_per_case": NULL_REPEATS,
            "primary_repeats_per_case": PRIMARY_REPEATS,
            "bootstrap_replicates_per_outer_fit": BOOTSTRAP_REPLICATES,
            "initial_starts": INITIAL_STARTS,
            "additional_starts_only_after_quality_failure": RETRY_STARTS,
            "retry_rule_frozen_before_fresh_seed_evaluation": True,
            "seed_offset": SEED_OFFSET,
            "monte_carlo_alpha": MONTE_CARLO_ALPHA,
        },
        "records": records,
        "summary": summary,
        "assessment": assessment,
        "runtime": {
            "total_elapsed_seconds": time.perf_counter() - started,
            "outer_decisions": len(records),
            "nominal_refits": len(records) * (BOOTSTRAP_REPLICATES + 1),
        },
    }
    write_outputs(payload)
    print(json.dumps({
        "summary": summary,
        "assessment": assessment,
        "runtime": payload["runtime"],
    }, indent=2))


if __name__ == "__main__":
    main()
