"""Paired audit of scientific-decision stability across optimizer budgets.

Stage 49 showed that stronger deterministic refinement resolves severe-case
ambiguity but can over-refuse mild drift.  This experiment holds each dataset,
initialization scheme, calibration, and decision threshold fixed while varying
only the symmetric L-BFGS budget for shared and grouped candidates.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from probe_cross_start_consistency_transfer import select_consistent_candidate
from probe_decomposed_tolerance_transfer import (
    CHANNEL_COUNTS,
    HETEROGENEITY_CONSTRUCTIONS,
    NOISE_CORRELATION,
    PROXY_SCOPE,
    build_transfer_observation,
    frozen_total_tolerance,
)
from probe_extended_refinement_transfer import ADAM_STEPS, load_frozen_stage48
from probe_high_dimensional_shared_spectrum import DEVICE, DTYPE, fit_candidate
from probe_nested_group_sharing_gate import fit_grouped_candidate
from probe_noise_aware_sharing_gate import (
    classify_with_limit,
    second_difference_correlation_proxy,
)
from probe_noise_scale_optimizer_transfer import (
    GROUPED_STARTS,
    PROJECT_NOISE_STDS,
    PROJECT_REPEATS,
    SHARED_STARTS,
    mixed_difference_noise_scale,
    scale_correction,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROJECT_SEED_BASE = 131000
MILD_DRIFT = 0.05
LBFGS_BUDGETS = (80, 160, 240)
MIN_EXACT_AGREEMENT_FRACTION = 0.75
MAX_DIRECT_REVERSAL_FRACTION = 1.0 / 24.0
MAX_CELL_DIRECT_REVERSAL_FRACTION = 1.0 / 3.0


def decision_class(decision: str) -> str:
    if decision in {"ACCEPT_SHARED_MECHANISM", "ACCEPT_WITH_SCOPE_LIMITS"}:
        return "RETAIN"
    if decision == "REFUSE_SHARED_MECHANISM":
        return "REFUSE"
    if decision == "INDETERMINATE_OPTIMIZATION":
        return "INDETERMINATE"
    raise ValueError(f"Unknown decision: {decision}")


def project_record_for_budget(
    channels: int,
    noise_std: float,
    construction: str,
    repeat: int,
    lbfgs_steps: int,
    noise_calibration: dict,
    consistency_calibration: dict,
    log_spectral_drift: float = MILD_DRIFT,
    project_seed_base: int = PROJECT_SEED_BASE,
) -> dict:
    construction_code = HETEROGENEITY_CONSTRUCTIONS.index(construction)
    seed = (
        project_seed_base
        + channels * 10
        + int(noise_std * 1.0e6)
        + construction_code * 1000
        + int(log_spectral_drift * 1000)
        + repeat
    )
    times, observations, train_idx, val_idx, _, block_labels = (
        build_transfer_observation(
            channels, noise_std, construction, log_spectral_drift, seed
        )
    )
    shared_candidates = [
        fit_candidate(
            times,
            observations,
            train_idx,
            val_idx,
            2,
            True,
            seed * 10 + start,
            adam_steps=ADAM_STEPS,
            lbfgs_steps=lbfgs_steps,
        )
        for start in range(SHARED_STARTS)
    ]
    shared, diagnostics, consistent_starts, signal_scale = select_consistent_candidate(
        shared_candidates, times, observations, consistency_calibration
    )
    grouped = min(
        [
            fit_grouped_candidate(
                times,
                observations,
                train_idx,
                val_idx,
                torch.tensor(block_labels, dtype=torch.long, device=DEVICE),
                seed * 10 + 20 + start,
                adam_steps=ADAM_STEPS,
                lbfgs_steps=lbfgs_steps,
            )
            for start in range(GROUPED_STARTS)
        ],
        key=lambda candidate: candidate.bic,
    )
    noise_proxy = mixed_difference_noise_scale(observations)
    correlation_proxy = second_difference_correlation_proxy(observations)
    base_tolerance = frozen_total_tolerance(correlation_proxy)
    correction = scale_correction(noise_proxy, noise_calibration)
    total_tolerance = base_tolerance + correction
    in_scope = (
        noise_calibration["noise_proxy_min"]
        <= noise_proxy
        <= noise_calibration["noise_proxy_max"]
        and PROXY_SCOPE[0] <= correlation_proxy <= PROXY_SCOPE[1]
    )
    if shared is None:
        decision = "INDETERMINATE_OPTIMIZATION"
        support = None
        shared_val_rmse = None
    else:
        support = shared.bic - grouped.bic
        decision = classify_with_limit(
            support, shared.val_rmse, grouped.val_rmse, total_tolerance
        )
        shared_val_rmse = shared.val_rmse
    return {
        "channels": channels,
        "noise_std_diagnostic": noise_std,
        "heterogeneity_construction": construction,
        "log_spectral_drift": log_spectral_drift,
        "repeat": repeat,
        "seed": seed,
        "lbfgs_steps": lbfgs_steps,
        "decision": decision,
        "decision_class": decision_class(decision),
        "adequate_shared_starts": consistent_starts,
        "diagnostics_in_calibration_scope": in_scope,
        "noise_scale_proxy": noise_proxy,
        "correlation_proxy": correlation_proxy,
        "signal_scale": signal_scale,
        "cross_start_diagnostics": diagnostics,
        "augmented_total_tolerance": total_tolerance,
        "group_bic_support": support,
        "shared_val_rmse": shared_val_rmse,
        "grouped_val_rmse": grouped.val_rmse,
    }


def paired_summary(records: list[dict]) -> dict:
    paired = defaultdict(list)
    for record in records:
        key = (
            record["channels"],
            record["noise_std_diagnostic"],
            record["heterogeneity_construction"],
            record["repeat"],
            record["seed"],
        )
        paired[key].append(record)

    pairs = []
    for key, items in sorted(paired.items()):
        ordered = sorted(items, key=lambda item: item["lbfgs_steps"])
        classes = [item["decision_class"] for item in ordered]
        direct_reversal = "RETAIN" in classes and "REFUSE" in classes
        pairs.append(
            {
                "channels": key[0],
                "noise_std": key[1],
                "heterogeneity_construction": key[2],
                "repeat": key[3],
                "seed": key[4],
                "budgets": [item["lbfgs_steps"] for item in ordered],
                "decisions": [item["decision"] for item in ordered],
                "decision_classes": classes,
                "exact_agreement": len(set(classes)) == 1,
                "direct_reversal": direct_reversal,
                "contains_indeterminate": "INDETERMINATE" in classes,
            }
        )

    cell_groups = defaultdict(list)
    for item in pairs:
        cell_groups[
            (
                item["channels"],
                item["noise_std"],
                item["heterogeneity_construction"],
            )
        ].append(item)
    cells = []
    for key, items in sorted(cell_groups.items()):
        cells.append(
            {
                "channels": key[0],
                "noise_std": key[1],
                "heterogeneity_construction": key[2],
                "trials": len(items),
                "exact_agreement_fraction": float(
                    np.mean([item["exact_agreement"] for item in items])
                ),
                "direct_reversal_fraction": float(
                    np.mean([item["direct_reversal"] for item in items])
                ),
                "indeterminate_fraction": float(
                    np.mean([item["contains_indeterminate"] for item in items])
                ),
            }
        )

    exact_agreement_fraction = float(
        np.mean([item["exact_agreement"] for item in pairs])
    )
    direct_reversal_fraction = float(
        np.mean([item["direct_reversal"] for item in pairs])
    )
    all_in_scope = all(
        record["diagnostics_in_calibration_scope"] for record in records
    )
    checks = {
        "complete_paired_matrix": len(pairs) == 24
        and all(len(items) == len(LBFGS_BUDGETS) for items in paired.values()),
        "exact_agreement_fraction_at_least_0_75": exact_agreement_fraction
        >= MIN_EXACT_AGREEMENT_FRACTION,
        "direct_reversal_fraction_at_most_1_of_24": direct_reversal_fraction
        <= MAX_DIRECT_REVERSAL_FRACTION,
        "each_cell_direct_reversal_fraction_at_most_1_of_3": all(
            cell["direct_reversal_fraction"]
            <= MAX_CELL_DIRECT_REVERSAL_FRACTION
            for cell in cells
        ),
        "all_diagnostics_in_calibration_scope": all_in_scope,
    }
    return {
        "pairs": pairs,
        "cells": cells,
        "exact_agreement_fraction": exact_agreement_fraction,
        "direct_reversal_fraction": direct_reversal_fraction,
        "indeterminate_pair_fraction": float(
            np.mean([item["contains_indeterminate"] for item in pairs])
        ),
        "checks": checks,
        "route_pass": all(checks.values()),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "optimizer_budget_stability.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    summary = payload["summary"]
    lines = [
        "# Paired optimizer-budget decision-stability audit",
        "",
        f"Device: `{payload['device']}`; route pass: **{summary['route_pass']}**.",
        "",
        f"Exact tri-state agreement: {summary['exact_agreement_fraction']:.3f}; direct retain/refuse reversal: {summary['direct_reversal_fraction']:.3f}; pairs containing an indeterminate result: {summary['indeterminate_pair_fraction']:.3f}.",
        "",
        "| Channels | Noise std | Construction | Agreement | Direct reversal | Indeterminate |",
        "|---:|---:|:---|---:|---:|---:|",
    ]
    for row in summary["cells"]:
        lines.append(
            f"| {row['channels']} | {row['noise_std']:.1e} | "
            f"{row['heterogeneity_construction']} | "
            f"{row['exact_agreement_fraction']:.2f} | "
            f"{row['direct_reversal_fraction']:.2f} | "
            f"{row['indeterminate_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "All observations, initializations, calibrations, and scientific-decision thresholds are paired across budgets. No budget is selected after observing its result.",
        ]
    )
    (RESULTS / "optimizer_budget_stability.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    noise_calibration, consistency_calibration = load_frozen_stage48()
    records = []
    for channels in CHANNEL_COUNTS:
        for noise_std in PROJECT_NOISE_STDS:
            for construction in HETEROGENEITY_CONSTRUCTIONS:
                for repeat in range(PROJECT_REPEATS):
                    for budget in LBFGS_BUDGETS:
                        record = project_record_for_budget(
                            channels,
                            noise_std,
                            construction,
                            repeat,
                            budget,
                            noise_calibration,
                            consistency_calibration,
                        )
                        records.append(record)
                        print(
                            f"channels={channels} noise={noise_std:.1e} "
                            f"construction={construction} repeat={repeat} "
                            f"budget={budget} decision={record['decision']}",
                            flush=True,
                        )
    summary = paired_summary(records)
    payload = {
        "experiment": "optimizer_budget_stability",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "protocol": {
            "project_seed_base": PROJECT_SEED_BASE,
            "adam_steps": ADAM_STEPS,
            "lbfgs_budgets": list(LBFGS_BUDGETS),
            "mild_log_spectral_drift": MILD_DRIFT,
            "paired_datasets": 24,
            "total_budget_evaluations": 72,
            "shared_starts": SHARED_STARTS,
            "grouped_starts": GROUPED_STARTS,
        },
        "frozen_noise_calibration": noise_calibration,
        "frozen_consistency_calibration": consistency_calibration,
        "records": records,
        "summary": summary,
        "exit_rule": {
            "minimum_exact_agreement_fraction": MIN_EXACT_AGREEMENT_FRACTION,
            "maximum_direct_reversal_fraction": MAX_DIRECT_REVERSAL_FRACTION,
            "maximum_cell_direct_reversal_fraction": MAX_CELL_DIRECT_REVERSAL_FRACTION,
            "failure_action": "optimizer configuration remains decision-relevant contract metadata",
        },
    }
    write_outputs(payload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
