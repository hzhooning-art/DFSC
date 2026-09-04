"""Sensitivity curve for the Stage 69 power-certificate operating point.

The Stage 69 evaluation set is frozen. This script is a sensitivity audit, not
a second confirmatory selection exercise: no point on the curve replaces the
predeclared 0.70 power-lower-bound operating point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_power_certified_order_detection import (  # noqa: E402
    POWER_LOWER_BOUND,
    SPECIFICITY_LOWER_BOUND,
    certificate_key,
    run as run_stage69,
)


POWER_THRESHOLDS = (0.0, 0.30, 0.50, 0.70, 0.80, 0.90)


def summarize_decisions(records: list[dict], decisions: list[int | None]) -> dict:
    truths = [row["true_rank"] for row in records]
    covered = [decision is not None for decision in decisions]
    correct = [decision == truth for decision, truth in zip(decisions, truths)]
    accepted = sum(covered)
    rank_one = [decision for decision, truth in zip(decisions, truths) if truth == 1]
    rank_two = [decision for decision, truth in zip(decisions, truths) if truth == 2]
    return {
        "coverage": float(np.mean(covered)),
        "overall_accuracy_abstention_as_error": float(np.mean(correct)),
        "selective_accuracy": float(sum(correct) / accepted) if accepted else None,
        "selective_risk": float(1.0 - sum(correct) / accepted) if accepted else None,
        "false_order_elevation_rate": float(np.mean([value == 2 for value in rank_one])),
        "false_order_reduction_rate": float(np.mean([value == 1 for value in rank_two])),
        "rank_two_detection_rate": float(np.mean([value == 2 for value in rank_two])),
        "abstention_rate": float(np.mean([value is None for value in decisions])),
    }


def curve_point(records: list[dict], certificates: dict, power_threshold: float) -> dict:
    decisions = []
    for row in records:
        raw = row["methods"]["stage68_selective"]["decision"]
        certificate = certificates[certificate_key(
            row["horizon"], row["samples_per_channel"], row["noise_std"], row["noise_model"]
        )]
        qualified = (
            certificate["power_wilson_lower"] >= power_threshold
            and certificate["specificity_wilson_lower"] >= SPECIFICITY_LOWER_BOUND
        )
        if raw == 2:
            decision = 2
        elif raw == 1 and qualified:
            decision = 1
        else:
            decision = None
        decisions.append(decision)
    metrics = summarize_decisions(records, decisions)
    return {
        "power_wilson_lower_threshold": power_threshold,
        "qualified_designs": sum(
            row["power_wilson_lower"] >= power_threshold
            and row["specificity_wilson_lower"] >= SPECIFICITY_LOWER_BOUND
            for row in certificates.values()
        ),
        **metrics,
    }


def pareto_frontier(points: list[dict]) -> list[dict]:
    frontier = []
    for candidate in points:
        dominated = any(
            other["coverage"] >= candidate["coverage"]
            and other["selective_risk"] <= candidate["selective_risk"]
            and (
                other["coverage"] > candidate["coverage"]
                or other["selective_risk"] < candidate["selective_risk"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def run() -> dict:
    stage69, records = run_stage69()
    certificates = {
        certificate_key(
            row["horizon"], row["samples_per_channel"], row["noise_std"], row["noise_model"]
        ): row
        for row in stage69["certificates"]
    }
    overall = [curve_point(records, certificates, threshold) for threshold in POWER_THRESHOLDS]
    by_noise = {}
    for noise_model in ("white", "ar1"):
        local = [row for row in records if row["noise_model"] == noise_model]
        by_noise[noise_model] = [curve_point(local, certificates, threshold) for threshold in POWER_THRESHOLDS]
    return {
        "schema": "P5-Power-Certificate-Risk-Coverage-v1",
        "frozen_stage69_operating_point": POWER_LOWER_BOUND,
        "specificity_wilson_lower_threshold": SPECIFICITY_LOWER_BOUND,
        "power_threshold_grid": list(POWER_THRESHOLDS),
        "evaluation_trials": len(records),
        "curve": overall,
        "pareto_frontier": pareto_frontier(overall),
        "curve_by_noise_model": by_noise,
        "interpretation_rule": (
            "This grid audits sensitivity on the already frozen Stage 69 evaluation set. "
            "It must not be used to replace the 0.70 confirmatory operating point."
        ),
        "record_storage": (
            "Aggregate-only; Stage 69 decisions are regenerated deterministically while runtime is ignored."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-summary", action="store_true")
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
