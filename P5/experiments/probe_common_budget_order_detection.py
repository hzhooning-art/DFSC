"""Common-budget model-order benchmark for selective exponential detection.

Every detector receives the same grouped multichannel observations.  The
benchmark separates false order elevation, false order reduction, abstention,
selective risk, and runtime instead of treating abstention as a hidden error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from p5_memory_protocol import (  # noqa: E402
    CurveRecord,
    identifiability_certificate,
    matrix_pencil_consensus,
    matrix_pencil_order_selection,
)


HORIZONS = (4.0, 8.0, 16.0)
SAMPLE_COUNTS = (24, 48)
NOISE_LEVELS = (0.001, 0.005)
NOISE_MODELS = ("white", "ar1")
RATE_GAPS = (0.08, 0.32)
SEEDS = tuple(range(12))
BASE_RATE = 0.25
CHANNELS = 6
AR1_RHO = 0.65
BOUNDARY_THRESHOLD = 0.49085
MIN_RATE_RATIO = 1.20
MAX_LOG_RATE_STD = 0.15


def _noise(rng: np.random.Generator, samples: int, scale: float, model: str) -> np.ndarray:
    if model == "white":
        return rng.normal(0.0, scale, size=samples)
    innovations = rng.normal(0.0, scale * np.sqrt(1.0 - AR1_RHO**2), size=samples)
    output = np.empty(samples)
    output[0] = rng.normal(0.0, scale)
    for index in range(1, samples):
        output[index] = AR1_RHO * output[index - 1] + innovations[index]
    return output


def make_curves(
    true_rank: int,
    horizon: float,
    samples: int,
    noise: float,
    noise_model: str,
    gap: float | None,
    seed: int,
) -> tuple[list[CurveRecord], tuple[float, ...]]:
    rng = np.random.default_rng(seed)
    time_grid = np.linspace(0.0, horizon, samples)
    rates = (BASE_RATE,) if true_rank == 1 else (BASE_RATE, BASE_RATE + float(gap))
    curves = []
    for channel in range(CHANNELS):
        amplitudes = rng.uniform(0.45, 1.25, size=true_rank)
        offset = rng.uniform(-0.1, 0.25)
        value = offset + sum(
            amplitude * np.exp(-rate * time_grid)
            for amplitude, rate in zip(amplitudes, rates)
        )
        value = value + _noise(rng, samples, noise, noise_model)
        curves.append(CurveRecord(f"u{channel}", f"g{channel}", f"c{channel}", time_grid, value))
    return curves, rates


def _run_selection(curves: list[CurveRecord], criterion: str, threshold: float) -> tuple[int | None, float]:
    started = time.perf_counter()
    result = matrix_pencil_order_selection(
        curves,
        ranks=(1, 2),
        criterion=criterion,
        minimum_improvement=threshold,
    )
    return result["selected_rank"], time.perf_counter() - started


def _selective_detection(curves: list[CurveRecord], noise: float) -> tuple[dict, dict]:
    started = time.perf_counter()
    selection = matrix_pencil_order_selection(
        curves,
        ranks=(1, 2),
        criterion="bic",
        minimum_improvement=10.0,
    )
    consensus = matrix_pencil_consensus(
        curves,
        rank=2,
        delta_bic=10.0,
        max_log_rate_std=MAX_LOG_RATE_STD,
    )
    rank_two = selection["rank_records"]["2"]
    checks = {
        "strong_bic": selection["selected_rank"] == 2,
        "admissible_rank_two": bool(rank_two["success"]),
        "separation": False,
        "local_information": False,
        "cross_pencil_stability": bool(consensus["passes_consensus"]),
    }
    boundary = 0.0
    if rank_two["success"]:
        certificate = identifiability_certificate(curves, rank_two["rates"], noise_std=noise)
        boundary = certificate["normalized_local_boundary_index"]
        checks["separation"] = rank_two["minimum_rate_ratio"] >= MIN_RATE_RATIO
        checks["local_information"] = boundary >= BOUNDARY_THRESHOLD

    criterion_improvement = selection["transitions"][0]["criterion_improvement"]
    if criterion_improvement <= -10.0:
        decision: int | None = 1
    elif criterion_improvement >= 10.0 and all(checks.values()):
        decision = 2
    else:
        decision = None
    details = {
        "decision": decision,
        "rank_one_bic_margin": 10.0,
        "rank_two_bic_margin": 10.0,
        "criterion_improvement_rank1_to_rank2": criterion_improvement,
        "checks": checks,
        "normalized_local_boundary_index": boundary,
        "runtime_seconds": time.perf_counter() - started,
    }
    return details, consensus


def run_trial(
    true_rank: int,
    horizon: float,
    samples: int,
    noise: float,
    noise_model: str,
    gap: float | None,
    seed: int,
) -> dict:
    curves, rates = make_curves(true_rank, horizon, samples, noise, noise_model, gap, seed)
    methods = {}
    for name, criterion, threshold in (
        ("matrix_pencil_aic", "aic", 0.0),
        ("matrix_pencil_aicc", "aicc", 0.0),
        ("matrix_pencil_bic", "bic", 0.0),
        ("matrix_pencil_strong_bic", "bic", 10.0),
    ):
        decision, runtime = _run_selection(curves, criterion, threshold)
        methods[name] = {"decision": decision, "runtime_seconds": runtime}

    selective, consensus = _selective_detection(curves, noise)
    if consensus["passes_consensus"]:
        stability_decision: int | None = 2
    else:
        central = consensus["records"][len(consensus["records"]) // 2]["selected_rank"]
        stability_decision = 1 if central == 1 else None
    methods["cross_pencil_stability"] = {
        "decision": stability_decision,
        "runtime_seconds": selective["runtime_seconds"],
    }
    methods["selective_detector"] = selective
    return {
        "true_rank": true_rank,
        "true_rates": list(rates),
        "horizon": horizon,
        "samples_per_channel": samples,
        "noise_std": noise,
        "noise_model": noise_model,
        "rate_gap": gap,
        "seed": seed,
        "methods": methods,
    }


def summarize(records: list[dict]) -> dict:
    def mean_or_none(values: list[bool]) -> float | None:
        return float(np.mean(values)) if values else None

    method_names = tuple(records[0]["methods"])
    summary = {}
    for method in method_names:
        decisions = [row["methods"][method]["decision"] for row in records]
        truths = [row["true_rank"] for row in records]
        covered = [decision is not None for decision in decisions]
        correct = [decision == truth for decision, truth in zip(decisions, truths)]
        rank_one = [truth == 1 for truth in truths]
        rank_two = [truth == 2 for truth in truths]
        accepted_count = sum(covered)
        summary[method] = {
            "trials": len(records),
            "coverage": float(np.mean(covered)),
            "overall_accuracy_abstention_as_error": float(np.mean(correct)),
            "selective_accuracy": float(sum(correct) / accepted_count) if accepted_count else None,
            "selective_risk": float(1.0 - sum(correct) / accepted_count) if accepted_count else None,
            "false_order_elevation_rate": mean_or_none([
                decision == 2 for decision, keep in zip(decisions, rank_one) if keep
            ]),
            "false_order_reduction_rate": mean_or_none([
                decision == 1 for decision, keep in zip(decisions, rank_two) if keep
            ]),
            "rank_two_detection_rate": mean_or_none([
                decision == 2 for decision, keep in zip(decisions, rank_two) if keep
            ]),
            "abstention_rate": float(np.mean([decision is None for decision in decisions])),
            "median_runtime_seconds": float(np.median([
                row["methods"][method]["runtime_seconds"] for row in records
            ])),
        }
    return summary


def stratified_summary(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        key = (
            row["true_rank"],
            row["horizon"],
            row["samples_per_channel"],
            row["noise_std"],
            row["noise_model"],
            row["rate_gap"],
        )
        groups[key].append(row)
    output = []
    for key, local in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        output.append({
            "true_rank": key[0],
            "horizon": key[1],
            "samples_per_channel": key[2],
            "noise_std": key[3],
            "noise_model": key[4],
            "rate_gap": key[5],
            "trials": len(local),
            "methods": summarize(local),
        })
    return output


def factor_summary(records: list[dict]) -> dict:
    factors = {}
    definitions = {
        "true_rank": lambda row: row["true_rank"],
        "noise_model": lambda row: row["noise_model"],
        "noise_std": lambda row: row["noise_std"],
        "horizon": lambda row: row["horizon"],
        "samples_per_channel": lambda row: row["samples_per_channel"],
        "rate_gap": lambda row: row["rate_gap"] if row["true_rank"] == 2 else "rank1_not_applicable",
    }
    for name, getter in definitions.items():
        values = sorted({getter(row) for row in records}, key=str)
        factors[name] = {
            str(value): summarize([row for row in records if getter(row) == value])
            for value in values
        }
    return factors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stdout-summary",
        action="store_true",
        help="emit aggregate decisions without writing individual records",
    )
    args = parser.parse_args()
    records = []
    for true_rank in (1, 2):
        gaps = (None,) if true_rank == 1 else RATE_GAPS
        for horizon in HORIZONS:
            for samples in SAMPLE_COUNTS:
                for noise in NOISE_LEVELS:
                    for noise_model in NOISE_MODELS:
                        for gap in gaps:
                            cell_seed = int(
                                100000 * true_rank
                                + 1000 * horizon
                                + 10 * samples
                                + round(1e5 * noise)
                                + (500000 if noise_model == "ar1" else 0)
                                + (0 if gap is None else round(1e4 * gap))
                            )
                            for seed in SEEDS:
                                records.append(run_trial(
                                    true_rank,
                                    horizon,
                                    samples,
                                    noise,
                                    noise_model,
                                    gap,
                                    cell_seed + seed,
                                ))
    aggregate = {
        "schema": "P5-Common-Budget-Order-Detection-v1",
        "design": {
            "true_ranks": [1, 2],
            "horizons": list(HORIZONS),
            "samples_per_channel": list(SAMPLE_COUNTS),
            "noise_std": list(NOISE_LEVELS),
            "noise_models": list(NOISE_MODELS),
            "ar1_rho": AR1_RHO,
            "rank_two_log_rate_gaps": list(RATE_GAPS),
            "channels": CHANNELS,
            "seeds_per_cell": len(SEEDS),
            "same_observations_for_every_method": True,
        },
        "summary": summarize(records),
        "factor_summary": factor_summary(records),
        "claim_boundary": (
            "This is a controlled common-budget benchmark for one grouped exponential generator. "
            "Abstention is reported explicitly and is not removed from overall accuracy."
        ),
    }
    if args.stdout_summary:
        aggregate["record_storage"] = (
            "Aggregate-only artifact; decisions are deterministically regenerated, while runtime is remeasured."
        )
        print(json.dumps(aggregate, indent=2))
        return
    payload = {
        **aggregate,
        "cells": stratified_summary(records),
        "records": records,
        "record_storage": "Complete individual records included.",
    }
    output = RESULTS / "common_budget_order_detection.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "trials": len(records), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
