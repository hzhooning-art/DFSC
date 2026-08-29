"""Stress-test the frozen retry gate under a constrained optimization budget."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank
from probe_refusal_calibration import RESULTS, wilson_interval
from probe_multiwindow_external_calibration import NOISE_STD, response
from probe_sampling_process_stress import FIT_CONDITION_LIMIT, FIT_RMSE_LIMIT, split_indices
from probe_cluster_geometry_conditional_null import geometry_features, make_variable_clustered_times


CASES = ("signed_zero", "oscillation_zero")
REPEATS_PER_CASE = 60
SEED_OFFSET = 401000
TRAINING_POINTS = 20
ADAM_STEPS = 75
LBFGS_STEPS = 20
INITIAL_STARTS = 2
ADDITIONAL_STARTS = 4


def seed_for(case: str, repeat: int) -> int:
    return SEED_OFFSET + 2003 * CASES.index(case) + 101 * repeat


def quality_pass(fit) -> bool:
    return bool(
        np.isfinite(fit.val_rmse)
        and np.isfinite(fit.jacobian_condition)
        and fit.val_rmse <= FIT_RMSE_LIMIT
        and fit.jacobian_condition <= FIT_CONDITION_LIMIT
    )


def evaluate(case: str, repeat: int) -> dict:
    started = time.perf_counter()
    seed = seed_for(case, repeat)
    times = make_variable_clustered_times(seed)
    clean = response(case, times)
    rng = np.random.default_rng(seed + 17)
    observations = clean + NOISE_STD * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )
    train_pool, diagnostic_np, windows_np = split_indices(times, seed + 29)
    split_rng = np.random.default_rng(seed + 41)
    train_np = np.sort(split_rng.choice(
        train_pool, size=min(TRAINING_POINTS, len(train_pool)), replace=False
    ))
    train_idx = torch.tensor(train_np, dtype=torch.long, device=DEVICE)
    diagnostic_idx = torch.tensor(diagnostic_np, dtype=torch.long, device=DEVICE)

    fits = [
        fit_rank(
            times,
            observations,
            train_idx,
            diagnostic_idx,
            rank=1,
            seed=seed * 100 + start,
            adam_steps=ADAM_STEPS,
            lbfgs_steps=LBFGS_STEPS,
        )
        for start in range(INITIAL_STARTS + ADDITIONAL_STARTS)
    ]
    initial = min(fits[:INITIAL_STARTS], key=lambda item: item.bic)
    retry_triggered = not quality_pass(initial)
    policy_pool = fits if retry_triggered else fits[:INITIAL_STARTS]
    policy = min(policy_pool, key=lambda item: item.bic)
    exhaustive = min(fits, key=lambda item: item.bic)

    return {
        "case": case,
        "repeat": repeat,
        "seed": seed,
        "observation_count": int(len(times)),
        "training_count": int(len(train_np)),
        "geometry": geometry_features(times, windows_np),
        "all_starts": [
            {
                "start": start,
                "bic": fit.bic,
                "validation_rmse": fit.val_rmse,
                "jacobian_condition": fit.jacobian_condition,
                "quality_pass": quality_pass(fit),
            }
            for start, fit in enumerate(fits)
        ],
        "initial": {
            "validation_rmse": initial.val_rmse,
            "jacobian_condition": initial.jacobian_condition,
            "quality_pass": quality_pass(initial),
        },
        "retry_triggered": retry_triggered,
        "policy": {
            "starts_charged": INITIAL_STARTS + (ADDITIONAL_STARTS if retry_triggered else 0),
            "validation_rmse": policy.val_rmse,
            "jacobian_condition": policy.jacobian_condition,
            "quality_pass": quality_pass(policy),
            "recovered": retry_triggered and quality_pass(policy),
            "decision": "ACCEPT_CALIBRATION" if quality_pass(policy) else "REFUSE_CALIBRATION",
        },
        "exhaustive": {
            "validation_rmse": exhaustive.val_rmse,
            "jacobian_condition": exhaustive.jacobian_condition,
            "quality_pass": quality_pass(exhaustive),
            "would_improve_valid_initial_fit": (
                quality_pass(initial)
                and exhaustive.bic < initial.bic
                and exhaustive.val_rmse < initial.val_rmse
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for case in ("all",) + CASES:
        group = records if case == "all" else [record for record in records if record["case"] == case]
        triggers = sum(record["retry_triggered"] for record in group)
        recoveries = sum(record["policy"]["recovered"] for record in group)
        final_failures = sum(not record["policy"]["quality_pass"] for record in group)
        extra_start_cost = sum(record["policy"]["starts_charged"] - INITIAL_STARTS for record in group)
        rows.append({
            "case": case,
            "trials": len(group),
            "initial_failures": triggers,
            "initial_failure_fraction": triggers / len(group),
            "initial_failure_wilson95": wilson_interval(triggers, len(group)),
            "retry_recoveries": recoveries,
            "recovery_fraction_given_trigger": recoveries / triggers if triggers else None,
            "final_calibration_failures": final_failures,
            "final_failure_fraction": final_failures / len(group),
            "final_failure_wilson95": wilson_interval(final_failures, len(group)),
            "projected_total_starts": INITIAL_STARTS * len(group) + extra_start_cost,
            "projected_start_overhead_fraction": extra_start_cost / (INITIAL_STARTS * len(group)),
            "exhaustive_improvements_without_trigger": sum(
                record["exhaustive"]["would_improve_valid_initial_fit"] for record in group
            ),
            "median_elapsed_seconds_for_six_start_audit": float(np.median([
                record["elapsed_seconds"] for record in group
            ])),
        })
    return rows


def assess(summary: list[dict]) -> dict:
    overall = next(row for row in summary if row["case"] == "all")
    checks = {
        "stress_exposes_at_least_3_initial_failures": overall["initial_failures"] >= 3,
        "retry_recovers_at_least_75_percent_of_triggers": (
            overall["recovery_fraction_given_trigger"] is not None
            and overall["recovery_fraction_given_trigger"] >= 0.75
        ),
        "final_calibration_failure_fraction_at_most_0.025": (
            overall["final_failure_fraction"] <= 0.025
        ),
        "projected_start_overhead_at_most_0.15": (
            overall["projected_start_overhead_fraction"] <= 0.15
        ),
    }
    return {"checks": checks, "route_pass": all(checks.values())}


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retry_numerical_stress.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Retry numerical stress test",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        f"- Total runtime: {payload['runtime']['total_elapsed_seconds']:.1f} s",
        "",
        "| Case | Trials | Initial failures | Recoveries | Final failures | Projected start overhead | Six-start audit seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['case']} | {row['trials']} | {row['initial_failures']} | "
            f"{row['retry_recoveries']} | {row['final_calibration_failures']} | "
            f"{row['projected_start_overhead_fraction']:.3f} | "
            f"{row['median_elapsed_seconds_for_six_start_audit']:.2f} |"
        )
    (RESULTS / "retry_numerical_stress.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    records = [
        evaluate(case, repeat)
        for case in CASES
        for repeat in range(REPEATS_PER_CASE)
    ]
    summary = summarize(records)
    assessment = assess(summary)
    payload = {
        "experiment": "retry_numerical_stress",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "design": {
            "correctly_specified_rank_one_null_cases": list(CASES),
            "repeats_per_case": REPEATS_PER_CASE,
            "training_points": TRAINING_POINTS,
            "sampling": "continuously_variable_clustered",
            "noise_std": NOISE_STD,
            "adam_steps_per_start": ADAM_STEPS,
            "lbfgs_steps_per_start": LBFGS_STEPS,
            "initial_starts": INITIAL_STARTS,
            "additional_starts_after_quality_failure": ADDITIONAL_STARTS,
            "all_six_starts_run_for_audit_only": True,
            "policy_cost_charges_additional_starts_only_after_failure": True,
            "seed_offset": SEED_OFFSET,
            "fit_rmse_limit": FIT_RMSE_LIMIT,
            "fit_condition_limit": FIT_CONDITION_LIMIT,
        },
        "records": records,
        "summary": summary,
        "assessment": assessment,
        "runtime": {
            "total_elapsed_seconds": time.perf_counter() - started,
            "six_start_fits": len(records),
            "individual_optimizer_runs": len(records) * (INITIAL_STARTS + ADDITIONAL_STARTS),
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
