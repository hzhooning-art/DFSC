"""Stage 52: disjoint high-noise calibration of the validation budget.

The calibration bank contains only new exact-sharing and declared mild-drift
controls.  It cannot inspect the Stage 51 failures or any severe-drift record.
After one multiplicative validation allowance is frozen, a second seed bank
tests exact, mild, and severe drift at all preregistered optimizer budgets.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from probe_budget_consensus_abstention import consensus_decision
from probe_decomposed_tolerance_transfer import (
    CHANNEL_COUNTS,
    HETEROGENEITY_CONSTRUCTIONS,
)
from probe_extended_refinement_transfer import load_frozen_stage48
from probe_high_dimensional_shared_spectrum import DEVICE, DTYPE
from probe_noise_aware_sharing_gate import classify_with_limit
from probe_optimizer_budget_stability import (
    LBFGS_BUDGETS,
    decision_class,
    project_record_for_budget,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CALIBRATION_SEED_BASE = 161000
VALIDATION_SEED_BASE = 171000
HIGH_NOISE_STD = 1.6e-3
CALIBRATION_DRIFTS = (0.0, 0.05)
VALIDATION_DRIFTS = (0.0, 0.05, 0.15)
CALIBRATION_REPEATS = 2
VALIDATION_REPEATS = 2
TARGET_COVERAGE = 0.95
MULTIPLIER_CEILING = 2.5

MAX_EXACT_MILD_FALSE_REFUSAL = 0.05
MIN_EXACT_MILD_RETAIN = 0.70
MIN_SEVERE_REFUSAL = 0.75
MAX_BUDGET_SENSITIVE = 0.35


def validation_ratio(record: dict) -> float:
    value = record.get("shared_val_rmse")
    tolerance = record.get("augmented_total_tolerance")
    if value is None or tolerance is None or tolerance <= 0.0:
        return math.inf
    return float(value / tolerance)


def fit_conditional_multiplier(records: list[dict]) -> dict:
    """Freeze one conservative multiplier from disjoint acceptable controls."""
    if not records:
        raise ValueError("calibration records are required")
    if any(record["log_spectral_drift"] not in CALIBRATION_DRIFTS for record in records):
        raise ValueError("calibration may use only exact and declared mild controls")
    ratios = np.asarray([validation_ratio(record) for record in records], dtype=float)
    if not np.all(np.isfinite(ratios)):
        raise ValueError("all calibration controls require a determinate shared fit")
    order = min(len(ratios) - 1, math.ceil((len(ratios) + 1) * TARGET_COVERAGE) - 1)
    raw = max(1.0, float(np.sort(ratios)[order]))
    return {
        "validation_tolerance_multiplier": min(raw, MULTIPLIER_CEILING),
        "uncapped_multiplier": raw,
        "ceiling": MULTIPLIER_CEILING,
        "coverage_target": TARGET_COVERAGE,
        "calibration_records": len(records),
        "ratio_min": float(ratios.min()),
        "ratio_median": float(np.median(ratios)),
        "ratio_max": float(ratios.max()),
        "used_stage51_failures": False,
        "severe_drift_used_for_calibration": False,
    }


def apply_frozen_calibration(record: dict, calibration: dict) -> dict:
    updated = dict(record)
    multiplier = calibration["validation_tolerance_multiplier"]
    adjusted = record["augmented_total_tolerance"] * multiplier
    if record["shared_val_rmse"] is None or record["group_bic_support"] is None:
        decision = "INDETERMINATE_OPTIMIZATION"
    else:
        decision = classify_with_limit(
            record["group_bic_support"],
            record["shared_val_rmse"],
            record["grouped_val_rmse"],
            adjusted,
        )
    updated.update(
        {
            "stage52_validation_multiplier": multiplier,
            "stage52_adjusted_tolerance": adjusted,
            "stage52_decision": decision,
            "stage52_decision_class": decision_class(decision),
        }
    )
    return updated


def summarize(records: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (
            record["channels"],
            record["heterogeneity_construction"],
            record["log_spectral_drift"],
            record["repeat"],
            record["seed"],
        )
        grouped[key].append(record)

    pairs = []
    for key, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: item["lbfgs_steps"])
        classes = [item["stage52_decision_class"] for item in ordered]
        consensus, reason = consensus_decision(classes)
        pairs.append(
            {
                "channels": key[0],
                "noise_std": HIGH_NOISE_STD,
                "heterogeneity_construction": key[1],
                "log_spectral_drift": key[2],
                "repeat": key[3],
                "seed": key[4],
                "budget_classes": classes,
                "class_counts": dict(Counter(classes)),
                "consensus_class": consensus,
                "consensus_reason": reason,
                "budget_sensitive": reason == "BUDGET_SENSITIVE_BINARY_CONFLICT",
            }
        )

    acceptable = [item for item in pairs if item["log_spectral_drift"] <= 0.05]
    severe = [item for item in pairs if item["log_spectral_drift"] == 0.15]
    false_refusal = float(np.mean([item["consensus_class"] == "REFUSE" for item in acceptable]))
    retain = float(np.mean([item["consensus_class"] == "RETAIN" for item in acceptable]))
    severe_refusal = float(np.mean([item["consensus_class"] == "REFUSE" for item in severe]))
    sensitive = float(np.mean([item["budget_sensitive"] for item in pairs]))
    checks = {
        "complete_locked_matrix": len(pairs) == 24
        and all(len(items) == len(LBFGS_BUDGETS) for items in grouped.values()),
        "exact_mild_false_refusal_at_most_0_05": false_refusal <= MAX_EXACT_MILD_FALSE_REFUSAL,
        "exact_mild_retain_at_least_0_70": retain >= MIN_EXACT_MILD_RETAIN,
        "severe_refusal_at_least_0_75": severe_refusal >= MIN_SEVERE_REFUSAL,
        "budget_sensitive_at_most_0_35": sensitive <= MAX_BUDGET_SENSITIVE,
        "all_diagnostics_in_scope": all(record["diagnostics_in_calibration_scope"] for record in records),
    }
    return {
        "pairs": pairs,
        "exact_mild_false_refusal_fraction": false_refusal,
        "exact_mild_retain_fraction": retain,
        "severe_refusal_fraction": severe_refusal,
        "budget_sensitive_fraction": sensitive,
        "checks": checks,
        "route_pass": all(checks.values()),
    }


def make_records(seed_base: int, drifts: tuple[float, ...], repeats: int, calibrate: bool, calibration: dict | None = None) -> list[dict]:
    noise_calibration, consistency_calibration = load_frozen_stage48()
    output = []
    for channels in CHANNEL_COUNTS:
        for construction in HETEROGENEITY_CONSTRUCTIONS:
            for drift in drifts:
                for repeat in range(repeats):
                    for budget in LBFGS_BUDGETS:
                        record = project_record_for_budget(
                            channels,
                            HIGH_NOISE_STD,
                            construction,
                            repeat,
                            budget,
                            noise_calibration,
                            consistency_calibration,
                            log_spectral_drift=drift,
                            project_seed_base=seed_base,
                        )
                        if not calibrate:
                            assert calibration is not None
                            record = apply_frozen_calibration(record, calibration)
                        output.append(record)
                        print(
                            f"stage52 {'cal' if calibrate else 'val'} channels={channels} "
                            f"construction={construction} drift={drift:.2f} "
                            f"repeat={repeat} budget={budget}",
                            flush=True,
                        )
    return output


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "high_noise_conditional_calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    s = payload["summary"]
    lines = [
        "# Disjoint high-noise conditional calibration",
        "",
        f"Route pass: **{s['route_pass']}**.",
        f"Frozen validation multiplier: {payload['calibration']['validation_tolerance_multiplier']:.4f}.",
        f"Exact/mild false refusal: {s['exact_mild_false_refusal_fraction']:.3f}; retained: {s['exact_mild_retain_fraction']:.3f}; severe refused: {s['severe_refusal_fraction']:.3f}; budget-sensitive: {s['budget_sensitive_fraction']:.3f}.",
        "",
        "Calibration and validation use disjoint seeds. Stage 51 failures and severe-drift records are excluded from calibration.",
    ]
    (RESULTS / "high_noise_conditional_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    calibration_records = make_records(
        CALIBRATION_SEED_BASE, CALIBRATION_DRIFTS, CALIBRATION_REPEATS, True
    )
    calibration = fit_conditional_multiplier(calibration_records)
    validation_records = make_records(
        VALIDATION_SEED_BASE,
        VALIDATION_DRIFTS,
        VALIDATION_REPEATS,
        False,
        calibration,
    )
    summary = summarize(validation_records)
    payload = {
        "experiment": "high_noise_conditional_calibration",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "protocol": {
            "calibration_seed_base": CALIBRATION_SEED_BASE,
            "validation_seed_base": VALIDATION_SEED_BASE,
            "high_noise_std": HIGH_NOISE_STD,
            "budgets": list(LBFGS_BUDGETS),
            "calibration_drifts": list(CALIBRATION_DRIFTS),
            "validation_drifts": list(VALIDATION_DRIFTS),
        },
        "calibration": calibration,
        "calibration_records": calibration_records,
        "validation_records": validation_records,
        "summary": summary,
        "exit_rule": {
            "maximum_exact_mild_false_refusal": MAX_EXACT_MILD_FALSE_REFUSAL,
            "minimum_exact_mild_retain": MIN_EXACT_MILD_RETAIN,
            "minimum_severe_refusal": MIN_SEVERE_REFUSAL,
            "maximum_budget_sensitive": MAX_BUDGET_SENSITIVE,
            "failure_action": "close universal binary repair and retain a scoped abstaining protocol",
        },
    }
    write_outputs(payload)
    print(json.dumps({"calibration": calibration, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
