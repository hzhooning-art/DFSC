"""Measure power of the frozen exact-design conditional-bootstrap rule."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE
from probe_refusal_calibration import RESULTS, wilson_interval
from probe_multiwindow_external_calibration import NOISE_STD, response
from probe_sampling_process_stress import split_indices
from probe_cluster_geometry_conditional_null import (
    SOURCE_ARTIFACT,
    geometry_features,
    make_variable_clustered_times,
)
from probe_exact_design_conditional_bootstrap import (
    BOOTSTRAP_REPLICATES,
    MONTE_CARLO_ALPHA,
    fit_observations,
)


PRIMARY_CASES = (
    "oscillation_decay_016",
    "shifted_transient_020",
    "shifted_transient_055",
)
SECONDARY_CASES = ("oscillation_decay_036",)
REPEATS_PER_CASE = 6
SEED_OFFSET = 311000


def seed_for(case: str, repeat: int) -> int:
    return SEED_OFFSET + 1009 * (PRIMARY_CASES + SECONDARY_CASES).index(case) + 101 * repeat


def evaluate_case(case: str, repeat: int, legacy: float, global_threshold: float) -> dict:
    started = time.perf_counter()
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
        conditional_decision = (
            "REFUSE_CONTRACT" if monte_carlo_p <= MONTE_CARLO_ALPHA else "ACCEPT_CONTRACT"
        )
    else:
        exceedances = None
        monte_carlo_p = None
        conditional_threshold = None
        conditional_decision = "REFUSE_CALIBRATION"

    return {
        "case": case,
        "role": "primary" if case in PRIMARY_CASES else "secondary_boundary",
        "repeat": repeat,
        "seed": seed,
        "observation_count": int(len(times)),
        "training_count": int(len(train_np)),
        "geometry": geometry_features(times, windows_np),
        "outer_fit": outer,
        "valid_bootstrap_replicates": len(valid_bootstrap),
        "bootstrap_statistics": bootstrap_statistics.tolist(),
        "calibration_pass": bool(calibration_pass),
        "conditional_threshold": conditional_threshold,
        "exceedances": exceedances,
        "monte_carlo_p": monte_carlo_p,
        "elapsed_seconds": time.perf_counter() - started,
        "decisions": {
            "legacy_clustered": (
                "REFUSE_CONTRACT"
                if not outer["fit_quality_pass"] or outer["max_abs_statistic"] > legacy
                else "ACCEPT_CONTRACT"
            ),
            "new_global_clustered": (
                "REFUSE_CONTRACT"
                if not outer["fit_quality_pass"] or outer["max_abs_statistic"] > global_threshold
                else "ACCEPT_CONTRACT"
            ),
            "exact_design_bootstrap": conditional_decision,
        },
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for case in PRIMARY_CASES + SECONDARY_CASES:
        group = [record for record in records if record["case"] == case]
        methods = {}
        for method in ("legacy_clustered", "new_global_clustered", "exact_design_bootstrap"):
            refused = sum(record["decisions"][method] == "REFUSE_CONTRACT" for record in group)
            methods[method] = {
                "refusals": refused,
                "fraction": refused / len(group),
                "wilson95": wilson_interval(refused, len(group)),
            }
        rows.append({
            "case": case,
            "role": group[0]["role"],
            "trials": len(group),
            "outer_invalid_fits": sum(not record["outer_fit"]["fit_quality_pass"] for record in group),
            "conditional_calibration_failures": sum(not record["calibration_pass"] for record in group),
            "median_observed_statistic": float(np.median([
                record["outer_fit"]["max_abs_statistic"] for record in group
            ])),
            "median_conditional_threshold": float(np.median([
                record["conditional_threshold"]
                for record in group if record["conditional_threshold"] is not None
            ])),
            "median_elapsed_seconds": float(np.median([
                record["elapsed_seconds"] for record in group
            ])),
            "methods": methods,
        })
    return rows


def assess(records: list[dict], summary: list[dict]) -> dict:
    primary = [row for row in summary if row["role"] == "primary"]
    checks = {
        "all_24_outer_fits_valid": all(
            record["outer_fit"]["fit_quality_pass"] for record in records
        ),
        "all_456_bootstrap_fits_valid": all(
            record["valid_bootstrap_replicates"] == BOOTSTRAP_REPLICATES
            for record in records
        ),
        "each_primary_case_refused_at_least_5_of_6": all(
            row["methods"]["exact_design_bootstrap"]["refusals"] >= 5
            for row in primary
        ),
        "no_primary_case_loses_more_than_1_detection_vs_global": all(
            row["methods"]["new_global_clustered"]["refusals"]
            - row["methods"]["exact_design_bootstrap"]["refusals"] <= 1
            for row in primary
        ),
    }
    return {"checks": checks, "route_pass": all(checks.values())}


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "exact_design_conditional_power.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Exact-design conditional-bootstrap power",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        f"- Total elapsed seconds: {payload['runtime']['total_elapsed_seconds']:.1f}",
        "",
        "| Case | Role | Trials | Calibration failures | Legacy refusal | Global refusal | Conditional refusal | Median statistic | Median threshold | Median seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['case']} | {row['role']} | {row['trials']} | "
            f"{row['conditional_calibration_failures']} | "
            f"{row['methods']['legacy_clustered']['refusals']} | "
            f"{row['methods']['new_global_clustered']['refusals']} | "
            f"{row['methods']['exact_design_bootstrap']['refusals']} | "
            f"{row['median_observed_statistic']:.6f} | "
            f"{row['median_conditional_threshold']:.6f} | "
            f"{row['median_elapsed_seconds']:.2f} |"
        )
    (RESULTS / "exact_design_conditional_power.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    source = json.loads(SOURCE_ARTIFACT.read_text(encoding="utf-8"))
    legacy = float(source["calibration"]["clustered"]["threshold"])
    geometry_artifact = json.loads(
        (RESULTS / "cluster_geometry_conditional_null.json").read_text(encoding="utf-8")
    )
    global_threshold = float(geometry_artifact["global_calibration"]["threshold"])
    records = [
        evaluate_case(case, repeat, legacy, global_threshold)
        for case in PRIMARY_CASES + SECONDARY_CASES
        for repeat in range(REPEATS_PER_CASE)
    ]
    summary = summarize(records)
    assessment = assess(records, summary)
    payload = {
        "experiment": "exact_design_conditional_power",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "design": {
            "primary_cases": list(PRIMARY_CASES),
            "secondary_boundary_cases": list(SECONDARY_CASES),
            "repeats_per_case": REPEATS_PER_CASE,
            "outer_fits": len(records),
            "bootstrap_replicates_per_outer_fit": BOOTSTRAP_REPLICATES,
            "total_bootstrap_fits": len(records) * BOOTSTRAP_REPLICATES,
            "monte_carlo_alpha": MONTE_CARLO_ALPHA,
            "rule_changed_after_null_experiment": False,
            "candidate_rank": 1,
            "starts_per_fit": 2,
            "legacy_clustered_threshold": legacy,
            "new_global_clustered_threshold": global_threshold,
        },
        "records": records,
        "summary": summary,
        "assessment": assessment,
        "runtime": {
            "total_elapsed_seconds": time.perf_counter() - started,
            "total_refits": len(records) * (BOOTSTRAP_REPLICATES + 1),
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
