"""Test whether stronger symmetric refinement resolves Stage 48 ambiguity.

All Stage 48 calibration values remain frozen.  Shared and grouped candidates
receive the same extended L-BFGS budget, and a fresh 72-record matrix determines
whether optimization indeterminacy falls enough to restore the strict route.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from probe_cross_start_consistency_transfer import (
    cross_start_diagnostics,
    select_consistent_candidate,
)
from probe_decomposed_tolerance_transfer import (
    CHANNEL_COUNTS,
    HETEROGENEITY_CONSTRUCTIONS,
    LOG_SPECTRAL_DRIFTS,
    NOISE_CORRELATION,
    PROXY_SCOPE,
    build_transfer_observation,
    frozen_total_tolerance,
)
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
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FROZEN_STAGE48 = RESULTS / "cross_start_consistency_transfer.json"
PROJECT_SEED_BASE = 111000
ADAM_STEPS = 280
BASELINE_LBFGS_STEPS = 80
EXTENDED_LBFGS_STEPS = 240


def load_frozen_stage48(path: Path = FROZEN_STAGE48) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["frozen_noise_calibration"], payload[
        "frozen_consistency_calibration"
    ]


def extended_shared_candidates(times, observations, train_idx, val_idx, seed: int):
    return [
        fit_candidate(
            times,
            observations,
            train_idx,
            val_idx,
            2,
            True,
            seed * 10 + start,
            adam_steps=ADAM_STEPS,
            lbfgs_steps=EXTENDED_LBFGS_STEPS,
        )
        for start in range(SHARED_STARTS)
    ]


def project_record(
    channels: int,
    noise_std: float,
    construction: str,
    drift: float,
    repeat: int,
    noise_calibration: dict,
    consistency_calibration: dict,
) -> dict:
    construction_code = HETEROGENEITY_CONSTRUCTIONS.index(construction)
    seed = (
        PROJECT_SEED_BASE
        + channels * 10
        + int(noise_std * 1.0e6)
        + construction_code * 1000
        + int(drift * 1000)
        + repeat
    )
    times, observations, train_idx, val_idx, _, block_labels = (
        build_transfer_observation(channels, noise_std, construction, drift, seed)
    )
    candidates = extended_shared_candidates(
        times, observations, train_idx, val_idx, seed
    )
    shared, diagnostics, consistent_starts, signal_scale = select_consistent_candidate(
        candidates, times, observations, consistency_calibration
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
                lbfgs_steps=EXTENDED_LBFGS_STEPS,
            )
            for start in range(GROUPED_STARTS)
        ],
        key=lambda candidate: candidate.bic,
    )
    if shared is None:
        decision = "INDETERMINATE_OPTIMIZATION"
        support = None
        shared_bic = None
        shared_train_rmse = None
        shared_val_rmse = None
    else:
        support = shared.bic - grouped.bic
        decision = classify_with_limit(
            support, shared.val_rmse, grouped.val_rmse, total_tolerance
        )
        shared_bic = shared.bic
        shared_train_rmse = shared.train_rmse
        shared_val_rmse = shared.val_rmse
    return {
        "channels": channels,
        "noise_std_diagnostic": noise_std,
        "heterogeneity_construction": construction,
        "log_spectral_drift": drift,
        "noise_correlation_diagnostic": NOISE_CORRELATION,
        "repeat": repeat,
        "seed": seed,
        "noise_scale_proxy": noise_proxy,
        "correlation_proxy": correlation_proxy,
        "signal_scale": signal_scale,
        "diagnostics_in_calibration_scope": in_scope,
        "adequate_shared_starts": consistent_starts,
        "cross_start_diagnostics": diagnostics,
        "stage45_total_tolerance": base_tolerance,
        "noise_scale_correction": correction,
        "augmented_total_tolerance": total_tolerance,
        "decision": decision,
        "group_bic_support": support,
        "shared_bic": shared_bic,
        "shared_train_rmse": shared_train_rmse,
        "shared_val_rmse": shared_val_rmse,
        "grouped_val_rmse": grouped.val_rmse,
    }


def severe_rates(summary: dict) -> dict:
    rows = [
        row for row in summary["rows"] if row["log_spectral_drift"] == 0.15
    ]
    return {
        "mean_refuse_fraction": float(
            np.mean([row["refuse_fraction"] for row in rows])
        ),
        "mean_indeterminate_fraction": float(
            np.mean([row["indeterminate_fraction"] for row in rows])
        ),
        "cells_meeting_explicit_refusal_rule": sum(
            row["refuse_fraction"] >= 2.0 / 3.0 for row in rows
        ),
        "severe_cells": len(rows),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "extended_refinement_transfer.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Extended deterministic-refinement transfer audit",
        "",
        f"Device: `{payload['device']}`; route pass: **{payload['summary']['route_pass']}**.",
        "",
        f"L-BFGS steps per shared/grouped start: {EXTENDED_LBFGS_STEPS} (Stage 48: {BASELINE_LBFGS_STEPS}).",
        "",
        "| Channels | Noise std | Construction | Drift | Refused | Indeterminate | Adverse | Consistent starts | In scope |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["rows"]:
        lines.append(
            f"| {row['channels']} | {row['noise_std']:.1e} | "
            f"{row['heterogeneity_construction']} | {row['log_spectral_drift']:.2f} | "
            f"{row['refuse_fraction']:.2f} | {row['indeterminate_fraction']:.2f} | "
            f"{row['adverse_fraction']:.2f} | "
            f"{row['median_adequate_shared_starts']:.1f} | "
            f"{row['in_scope_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Stage 48 calibration values were loaded unchanged. Shared and grouped candidates received the same extended deterministic-refinement budget.",
            "",
            f"Stage 48 severe mean refusal/indeterminate: {payload['comparison']['stage48']['mean_refuse_fraction']:.3f}/{payload['comparison']['stage48']['mean_indeterminate_fraction']:.3f}.",
            f"Stage 49 severe mean refusal/indeterminate: {payload['comparison']['stage49']['mean_refuse_fraction']:.3f}/{payload['comparison']['stage49']['mean_indeterminate_fraction']:.3f}.",
        ]
    )
    (RESULTS / "extended_refinement_transfer.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    noise_calibration, consistency_calibration = load_frozen_stage48()
    stage48_payload = json.loads(FROZEN_STAGE48.read_text(encoding="utf-8"))
    records = []
    for channels in CHANNEL_COUNTS:
        for noise_std in PROJECT_NOISE_STDS:
            for construction in HETEROGENEITY_CONSTRUCTIONS:
                for drift in LOG_SPECTRAL_DRIFTS:
                    for repeat in range(PROJECT_REPEATS):
                        record = project_record(
                            channels,
                            noise_std,
                            construction,
                            drift,
                            repeat,
                            noise_calibration,
                            consistency_calibration,
                        )
                        records.append(record)
                        print(
                            f"channels={channels} noise={noise_std:.1e} "
                            f"construction={construction} drift={drift:.2f} "
                            f"repeat={repeat} decision={record['decision']} "
                            f"consistent={record['adequate_shared_starts']}",
                            flush=True,
                        )
    summary = summarize(records)
    comparison = {
        "stage48": severe_rates(stage48_payload["summary"]),
        "stage49": severe_rates(summary),
    }
    comparison["indeterminate_fraction_reduction"] = (
        comparison["stage48"]["mean_indeterminate_fraction"]
        - comparison["stage49"]["mean_indeterminate_fraction"]
    )
    comparison["refuse_fraction_change"] = (
        comparison["stage49"]["mean_refuse_fraction"]
        - comparison["stage48"]["mean_refuse_fraction"]
    )
    payload = {
        "experiment": "extended_refinement_transfer",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "protocol": {
            "project_seed_base": PROJECT_SEED_BASE,
            "adam_steps": ADAM_STEPS,
            "baseline_lbfgs_steps": BASELINE_LBFGS_STEPS,
            "extended_lbfgs_steps": EXTENDED_LBFGS_STEPS,
            "shared_starts": SHARED_STARTS,
            "grouped_starts": GROUPED_STARTS,
            "project_records": len(records),
            "frozen_stage48_artifact": str(FROZEN_STAGE48.relative_to(ROOT)),
        },
        "frozen_noise_calibration": noise_calibration,
        "frozen_consistency_calibration": consistency_calibration,
        "project_records": records,
        "summary": summary,
        "comparison": comparison,
        "exit_rule": {
            "continue_binary_refusal_route_only_if_route_passes": True,
            "otherwise_retain_tri_state_decision": True,
        },
    }
    write_outputs(payload)
    print(json.dumps({"summary": summary, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
