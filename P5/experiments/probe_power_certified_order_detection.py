"""Disjoint calibration/evaluation study for power-certified order decisions.

Stage 68 showed that absence of rank-two evidence is not evidence for rank one.
This experiment therefore permits a rank-one claim only in observation designs
whose calibration data had adequate power for the declared relevant rate gap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p5_memory_protocol import matrix_pencil_order_selection  # noqa: E402
from probe_common_budget_order_detection import (  # noqa: E402
    HORIZONS,
    NOISE_LEVELS,
    NOISE_MODELS,
    SAMPLE_COUNTS,
    _selective_detection,
    make_curves,
)


RELEVANT_RATE_GAP = 0.32
CALIBRATION_REPEATS = 64
EVALUATION_REPEATS = 16
CALIBRATION_SEED_OFFSET = 3_000_000
EVALUATION_SEED_OFFSET = 4_000_000
POWER_LOWER_BOUND = 0.70
SPECIFICITY_LOWER_BOUND = 0.90
CONFIDENCE_Z = 1.96
MIN_COVERAGE_BY_NOISE_MODEL = 0.20
MIN_RISK_IMPROVEMENT_OVER_AICC = 0.10
MAX_FALSE_ELEVATION = 0.05


def wilson_lower(successes: int, trials: int, z: float = CONFIDENCE_Z) -> float:
    """Return the two-sided 95% Wilson interval lower endpoint."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2))
    return (centre - radius) / denominator


def design_seed(horizon: float, samples: int, noise: float, noise_model: str) -> int:
    return int(
        10_000 * horizon
        + 100 * samples
        + round(1_000_000 * noise)
        + (500_000 if noise_model == "ar1" else 0)
    )


def observed_evidence(curves: list, noise: float) -> dict:
    details, _ = _selective_detection(curves, noise)
    return details


def calibrate_design(horizon: float, samples: int, noise: float, noise_model: str) -> dict:
    base = design_seed(horizon, samples, noise, noise_model) + CALIBRATION_SEED_OFFSET
    power_hits = 0
    specificity_hits = 0
    for repeat in range(CALIBRATION_REPEATS):
        rank_two, _ = make_curves(2, horizon, samples, noise, noise_model, RELEVANT_RATE_GAP, base + repeat)
        power_hits += observed_evidence(rank_two, noise)["decision"] == 2

        rank_one, _ = make_curves(1, horizon, samples, noise, noise_model, None, base + 100_000 + repeat)
        specificity_hits += observed_evidence(rank_one, noise)["criterion_improvement_rank1_to_rank2"] <= -10.0

    power_lower = wilson_lower(power_hits, CALIBRATION_REPEATS)
    specificity_lower = wilson_lower(specificity_hits, CALIBRATION_REPEATS)
    return {
        "horizon": horizon,
        "samples_per_channel": samples,
        "noise_std": noise,
        "noise_model": noise_model,
        "relevant_rate_gap": RELEVANT_RATE_GAP,
        "calibration_repeats_per_rank": CALIBRATION_REPEATS,
        "rank_two_detections": power_hits,
        "rank_one_specific_decisions": specificity_hits,
        "power_estimate": power_hits / CALIBRATION_REPEATS,
        "power_wilson_lower": power_lower,
        "specificity_estimate": specificity_hits / CALIBRATION_REPEATS,
        "specificity_wilson_lower": specificity_lower,
        "qualified_for_rank_one_claim": (
            power_lower >= POWER_LOWER_BOUND and specificity_lower >= SPECIFICITY_LOWER_BOUND
        ),
    }


def certificate_key(horizon: float, samples: int, noise: float, noise_model: str) -> tuple:
    return horizon, samples, noise, noise_model


def evaluate_trial(
    true_rank: int,
    gap: float | None,
    horizon: float,
    samples: int,
    noise: float,
    noise_model: str,
    seed: int,
    certificate: dict,
) -> dict:
    curves, rates = make_curves(true_rank, horizon, samples, noise, noise_model, gap, seed)
    started = time.perf_counter()
    aicc = matrix_pencil_order_selection(curves, ranks=(1, 2), criterion="aicc")
    aicc_runtime = time.perf_counter() - started
    selective = observed_evidence(curves, noise)

    if selective["decision"] == 2:
        certified_decision: int | None = 2
    elif (
        selective["criterion_improvement_rank1_to_rank2"] <= -10.0
        and certificate["qualified_for_rank_one_claim"]
    ):
        certified_decision = 1
    else:
        certified_decision = None

    return {
        "true_rank": true_rank,
        "true_rates": list(rates),
        "rate_gap": gap,
        "horizon": horizon,
        "samples_per_channel": samples,
        "noise_std": noise,
        "noise_model": noise_model,
        "seed": seed,
        "certificate_qualified": certificate["qualified_for_rank_one_claim"],
        "methods": {
            "matrix_pencil_aicc": {
                "decision": aicc["selected_rank"],
                "runtime_seconds": aicc_runtime,
            },
            "stage68_selective": {
                "decision": selective["decision"],
                "runtime_seconds": selective["runtime_seconds"],
            },
            "power_certified_selective": {
                "decision": certified_decision,
                "runtime_seconds": selective["runtime_seconds"],
            },
        },
    }


def summarize(records: list[dict]) -> dict:
    output = {}
    for method in records[0]["methods"]:
        decisions = [row["methods"][method]["decision"] for row in records]
        truths = [row["true_rank"] for row in records]
        covered = [decision is not None for decision in decisions]
        correct = [decision == truth for decision, truth in zip(decisions, truths)]
        accepted = sum(covered)
        rank_one_decisions = [decision for decision, truth in zip(decisions, truths) if truth == 1]
        rank_two_decisions = [decision for decision, truth in zip(decisions, truths) if truth == 2]
        output[method] = {
            "trials": len(records),
            "coverage": float(np.mean(covered)),
            "overall_accuracy_abstention_as_error": float(np.mean(correct)),
            "selective_accuracy": float(sum(correct) / accepted) if accepted else None,
            "selective_risk": float(1.0 - sum(correct) / accepted) if accepted else None,
            "false_order_elevation_rate": float(np.mean([value == 2 for value in rank_one_decisions])),
            "false_order_reduction_rate": float(np.mean([value == 1 for value in rank_two_decisions])),
            "rank_two_detection_rate": float(np.mean([value == 2 for value in rank_two_decisions])),
            "abstention_rate": float(np.mean([value is None for value in decisions])),
            "median_runtime_seconds": float(np.median([
                row["methods"][method]["runtime_seconds"] for row in records
            ])),
        }
    return output


def factor_summary(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[row["noise_model"]].append(row)
    return {name: summarize(local) for name, local in sorted(grouped.items())}


def success_assessment(by_noise_model: dict) -> dict:
    checks = {}
    for noise_model, methods in by_noise_model.items():
        baseline = methods["matrix_pencil_aicc"]
        candidate = methods["power_certified_selective"]
        checks[noise_model] = {
            "coverage_at_least_floor": candidate["coverage"] >= MIN_COVERAGE_BY_NOISE_MODEL,
            "risk_improvement_at_least_margin": (
                baseline["selective_risk"] - candidate["selective_risk"]
                >= MIN_RISK_IMPROVEMENT_OVER_AICC
            ),
            "false_elevation_below_ceiling": (
                candidate["false_order_elevation_rate"] <= MAX_FALSE_ELEVATION
            ),
        }
    return {
        "thresholds": {
            "minimum_coverage_each_noise_model": MIN_COVERAGE_BY_NOISE_MODEL,
            "minimum_selective_risk_improvement_over_aicc": MIN_RISK_IMPROVEMENT_OVER_AICC,
            "maximum_false_order_elevation": MAX_FALSE_ELEVATION,
        },
        "checks_by_noise_model": checks,
        "passes_all": all(all(local.values()) for local in checks.values()),
    }


def run() -> tuple[dict, list[dict]]:
    certificates = []
    lookup = {}
    for horizon in HORIZONS:
        for samples in SAMPLE_COUNTS:
            for noise in NOISE_LEVELS:
                for noise_model in NOISE_MODELS:
                    certificate = calibrate_design(horizon, samples, noise, noise_model)
                    certificates.append(certificate)
                    lookup[certificate_key(horizon, samples, noise, noise_model)] = certificate

    records = []
    for horizon in HORIZONS:
        for samples in SAMPLE_COUNTS:
            for noise in NOISE_LEVELS:
                for noise_model in NOISE_MODELS:
                    certificate = lookup[certificate_key(horizon, samples, noise, noise_model)]
                    base = design_seed(horizon, samples, noise, noise_model) + EVALUATION_SEED_OFFSET
                    for true_rank in (1, 2):
                        gaps = (None,) if true_rank == 1 else (0.08, RELEVANT_RATE_GAP)
                        for gap_index, gap in enumerate(gaps):
                            for repeat in range(EVALUATION_REPEATS):
                                records.append(evaluate_trial(
                                    true_rank,
                                    gap,
                                    horizon,
                                    samples,
                                    noise,
                                    noise_model,
                                    base + true_rank * 100_000 + gap_index * 10_000 + repeat,
                                    certificate,
                                ))

    by_noise_model = factor_summary(records)
    aggregate = {
        "schema": "P5-Power-Certified-Order-Detection-v1",
        "design": {
            "relevant_rate_gap": RELEVANT_RATE_GAP,
            "calibration_repeats_per_rank_and_design": CALIBRATION_REPEATS,
            "evaluation_repeats_per_truth_and_design": EVALUATION_REPEATS,
            "calibration_seed_offset": CALIBRATION_SEED_OFFSET,
            "evaluation_seed_offset": EVALUATION_SEED_OFFSET,
            "seeds_are_disjoint": True,
            "power_wilson_lower_threshold": POWER_LOWER_BOUND,
            "specificity_wilson_lower_threshold": SPECIFICITY_LOWER_BOUND,
            "discarded_dry_run": {
                "calibration_seed_offset": 1_000_000,
                "evaluation_seed_offset": 2_000_000,
                "calibration_repeats": 32,
                "reason": (
                    "The maximum attainable 95% Wilson specificity lower bound was 0.8928, "
                    "below the frozen 0.90 qualification threshold. All dry-run seeds were retired."
                ),
            },
        },
        "certificates": certificates,
        "qualified_designs": sum(row["qualified_for_rank_one_claim"] for row in certificates),
        "total_designs": len(certificates),
        "evaluation_summary": summarize(records),
        "evaluation_by_noise_model": by_noise_model,
        "success_assessment": success_assessment(by_noise_model),
        "claim_boundary": (
            "The certificate supports rank-one decisions only relative to a declared 0.32 rate gap "
            "under this grouped exponential generator; it is not a universal identifiability guarantee."
        ),
    }
    return aggregate, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-summary", action="store_true")
    args = parser.parse_args()
    aggregate, records = run()
    if args.stdout_summary:
        aggregate["record_storage"] = (
            "Aggregate-only artifact; decisions are deterministically regenerated, while runtime is remeasured."
        )
        print(json.dumps(aggregate, indent=2))
        return
    payload = {**aggregate, "records": records, "record_storage": "Complete individual records included."}
    output = RESULTS / "power_certified_order_detection.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": aggregate["evaluation_summary"]}, indent=2))


if __name__ == "__main__":
    main()
