"""Finite-budget resolution atlas with a shared matrix-pencil comparator."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p5_memory_protocol import (  # noqa: E402
    CurveRecord,
    identifiability_certificate,
    matrix_pencil_consensus,
    matrix_pencil_order_selection,
)


HORIZONS = (4.0, 8.0, 16.0)
SAMPLE_COUNTS = (24, 48)
NOISE_LEVELS = (0.001, 0.005, 0.015)
RATE_GAPS = (0.02, 0.08, 0.32)
SEEDS = tuple(range(20))
BASE_RATE = 0.25
CHANNELS = 6
BOUNDARY_THRESHOLD = 0.49085


def make_curves(horizon: float, samples: int, noise: float, gap: float, seed: int) -> list[CurveRecord]:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, horizon, samples)
    rates = (BASE_RATE, BASE_RATE + gap)
    curves = []
    for channel in range(CHANNELS):
        amplitudes = rng.uniform(0.45, 1.25, size=2)
        offset = rng.uniform(-0.1, 0.25)
        value = offset + sum(amplitude * np.exp(-rate * time) for amplitude, rate in zip(amplitudes, rates))
        value = value + rng.normal(0.0, noise, size=samples)
        curves.append(CurveRecord(f"u{channel}", f"g{channel}", f"c{channel}", time, value))
    return curves


def matched_log_rate_error(estimated: list[float], truth: tuple[float, float]) -> float:
    if len(estimated) != 2:
        return float("inf")
    return float(np.max(np.abs(np.log(np.sort(estimated)) - np.log(np.sort(truth)))))


def run_cell(horizon: float, samples: int, noise: float, gap: float) -> dict:
    rows = []
    truth = (BASE_RATE, BASE_RATE + gap)
    for seed in SEEDS:
        curves = make_curves(horizon, samples, noise, gap, seed)
        selection = matrix_pencil_order_selection(curves, ranks=(1, 2), delta_bic=10.0)
        consensus = matrix_pencil_consensus(curves, rank=2, delta_bic=10.0, max_log_rate_std=0.15)
        rank_two = selection["rank_records"]["2"]
        forced_rank_two = selection["selected_rank"] == 2
        recovery_error = matched_log_rate_error(rank_two.get("rates", []), truth)
        accurate = recovery_error <= 0.20
        if rank_two["success"]:
            certificate = identifiability_certificate(curves, rank_two["rates"], noise_std=noise)
            boundary = certificate["normalized_local_boundary_index"]
            separated = rank_two["minimum_rate_ratio"] >= 1.20
        else:
            boundary = 0.0
            separated = False
        evidence_supported = bool(
            forced_rank_two
            and separated
            and boundary >= BOUNDARY_THRESHOLD
            and consensus["passes_consensus"]
        )
        rows.append(
            {
                "seed": seed,
                "matrix_pencil_selected_rank": selection["selected_rank"],
                "rank_two_success": rank_two["success"],
                "estimated_rates": rank_two.get("rates", []),
                "maximum_log_rate_error": recovery_error,
                "accurate_two_rate_recovery": accurate,
                "normalized_boundary_index": boundary,
                "cross_pencil_consensus": consensus["passes_consensus"],
                "maximum_cross_pencil_log_rate_std": consensus["maximum_cross_pencil_log_rate_std"],
                "evidence_supported_rank_two": evidence_supported,
            }
        )
    forced = np.asarray([row["matrix_pencil_selected_rank"] == 2 for row in rows])
    supported = np.asarray([row["evidence_supported_rank_two"] for row in rows])
    accurate = np.asarray([row["accurate_two_rate_recovery"] for row in rows])
    return {
        "horizon": horizon,
        "samples_per_channel": samples,
        "noise_std": noise,
        "rate_gap": gap,
        "true_rate_ratio": (BASE_RATE + gap) / BASE_RATE,
        "trials": len(rows),
        "matrix_pencil_rank_two_rate": float(np.mean(forced)),
        "evidence_supported_rank_two_rate": float(np.mean(supported)),
        "accurate_two_rate_recovery_rate": float(np.mean(accurate)),
        "matrix_pencil_silent_overclaim_rate": float(np.mean(forced & ~accurate)),
        "evidence_record_silent_overclaim_rate": float(np.mean(supported & ~accurate)),
        "median_normalized_boundary_index": float(np.median([row["normalized_boundary_index"] for row in rows])),
        "records": rows,
    }


def recommendations(cells: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for cell in cells:
        grouped[(cell["noise_std"], cell["rate_gap"])].append(cell)
    output = []
    for (noise, gap), candidates in sorted(grouped.items()):
        eligible = [
            cell for cell in candidates
            if cell["evidence_supported_rank_two_rate"] >= 0.80
            and cell["evidence_record_silent_overclaim_rate"] <= 0.10
        ]
        eligible.sort(key=lambda cell: (cell["samples_per_channel"] * cell["horizon"], cell["horizon"]))
        output.append(
            {
                "noise_std": noise,
                "rate_gap": gap,
                "status": "supported_budget_found" if eligible else "no_supported_budget_in_grid",
                "minimum_supported_budget": None if not eligible else {
                    "horizon": eligible[0]["horizon"],
                    "samples_per_channel": eligible[0]["samples_per_channel"],
                },
            }
        )
    return output


def main() -> None:
    cells = []
    for horizon in HORIZONS:
        for samples in SAMPLE_COUNTS:
            for noise in NOISE_LEVELS:
                for gap in RATE_GAPS:
                    cells.append(run_cell(horizon, samples, noise, gap))
    payload = {
        "schema": "P5-Finite-Budget-Resolution-Atlas-v1",
        "design": {
            "horizons": HORIZONS,
            "samples_per_channel": SAMPLE_COUNTS,
            "noise_std": NOISE_LEVELS,
            "rate_gaps": RATE_GAPS,
            "seeds": SEEDS,
            "channels": CHANNELS,
            "base_rate": BASE_RATE,
            "boundary_threshold": BOUNDARY_THRESHOLD,
            "accurate_recovery_rule": "maximum matched absolute log-rate error <= 0.20",
        },
        "cells": cells,
        "budget_recommendations": recommendations(cells),
        "claim_boundary": (
            "The atlas is conditional on positive two-rate signals, a uniform grid, Gaussian noise, "
            "the declared amplitude distribution, and the block matrix-pencil implementation."
        ),
    }
    output = ROOT / "results" / "finite_budget_resolution_atlas.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output} with {len(cells)} cells")


if __name__ == "__main__":
    main()
