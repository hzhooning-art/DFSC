"""Stage 71 common-budget order-selection comparator study.

Classical shared linear-prediction and block-Hankel subspace information
criteria receive exactly the same observations as the frozen Stage 69 rule.
The subspace snapshot criteria are comparators, not claimed likelihood models,
because overlapping Hankel columns are dependent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_common_budget_order_detection import make_curves  # noqa: E402
from probe_power_certified_order_detection import (  # noqa: E402
    factor_summary,
    run as run_stage69,
    summarize,
)


def block_hankel(curves: list, rows_count: int | None = None) -> np.ndarray:
    values = np.stack([np.asarray(curve.value, dtype=float) for curve in curves])
    differenced = np.diff(values, axis=1)
    samples = differenced.shape[1]
    rows = rows_count or samples // 2
    rows = int(np.clip(rows, 3, samples - 2))
    columns = samples - rows
    blocks = [
        np.column_stack([signal[index : index + rows] for index in range(columns)])
        for signal in differenced
    ]
    return np.concatenate(blocks, axis=1)


def subspace_order(curves: list, criterion: str) -> dict:
    """Select rank 1/2 using covariance-eigenvalue AIC or MDL."""
    criterion = criterion.lower()
    if criterion not in {"aic", "mdl"}:
        raise ValueError("criterion must be aic or mdl")
    started = time.perf_counter()
    hankel = block_hankel(curves)
    singular = np.linalg.svd(hankel, compute_uv=False)
    eigenvalues = np.maximum(singular**2 / hankel.shape[1], np.finfo(float).tiny)
    sensors = len(eigenvalues)
    snapshots = hankel.shape[1]
    scores = {}
    for rank in (1, 2):
        noise = eigenvalues[rank:]
        arithmetic = float(np.mean(noise))
        geometric = float(np.exp(np.mean(np.log(noise))))
        log_ratio = math.log(max(geometric / arithmetic, np.finfo(float).tiny))
        free_parameters = rank * (2 * sensors - rank)
        if criterion == "aic":
            score = -2.0 * snapshots * (sensors - rank) * log_ratio + 2.0 * free_parameters
        else:
            score = -snapshots * (sensors - rank) * log_ratio + 0.5 * free_parameters * math.log(snapshots)
        scores[str(rank)] = float(score)
    return {
        "decision": min((1, 2), key=lambda rank: scores[str(rank)]),
        "scores": scores,
        "hankel_rows": hankel.shape[0],
        "effective_columns": hankel.shape[1],
        "runtime_seconds": time.perf_counter() - started,
    }


def shared_prony_order(curves: list, criterion: str) -> dict:
    """Select a shared linear-prediction order with AICc or BIC."""
    criterion = criterion.lower()
    if criterion not in {"aicc", "bic"}:
        raise ValueError("criterion must be aicc or bic")
    started = time.perf_counter()
    values = np.stack([np.asarray(curve.value, dtype=float) for curve in curves])
    differenced = np.diff(values, axis=1)
    records = {}
    for rank in (1, 2):
        design_blocks = []
        target_blocks = []
        for signal in differenced:
            target_blocks.append(signal[rank:])
            design_blocks.append(np.column_stack([
                signal[rank - lag - 1 : len(signal) - lag - 1]
                for lag in range(rank)
            ]))
        design = np.vstack(design_blocks)
        target = np.concatenate(target_blocks)
        coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        sse = float(np.dot(residual, residual))
        observations = len(target)
        parameters = rank
        likelihood = observations * math.log(max(sse / observations, np.finfo(float).tiny))
        aic = likelihood + 2.0 * parameters
        denominator = observations - parameters - 1
        aicc = aic + 2.0 * parameters * (parameters + 1) / denominator
        bic = likelihood + parameters * math.log(observations)
        records[str(rank)] = {
            "coefficients": coefficients.tolist(),
            "sse": sse,
            "aicc": float(aicc),
            "bic": float(bic),
        }
    return {
        "decision": min((1, 2), key=lambda rank: records[str(rank)][criterion]),
        "records": records,
        "runtime_seconds": time.perf_counter() - started,
    }


def run() -> dict:
    stage69, records = run_stage69()
    augmented = []
    for row in records:
        curves, _ = make_curves(
            row["true_rank"],
            row["horizon"],
            row["samples_per_channel"],
            row["noise_std"],
            row["noise_model"],
            row["rate_gap"],
            row["seed"],
        )
        methods = dict(row["methods"])
        for name, criterion in (("block_hankel_aic", "aic"), ("block_hankel_mdl", "mdl")):
            result = subspace_order(curves, criterion)
            methods[name] = {
                "decision": result["decision"],
                "runtime_seconds": result["runtime_seconds"],
            }
        for name, criterion in (("shared_prony_aicc", "aicc"), ("shared_prony_bic", "bic")):
            result = shared_prony_order(curves, criterion)
            methods[name] = {
                "decision": result["decision"],
                "runtime_seconds": result["runtime_seconds"],
            }
        augmented.append({**row, "methods": methods})

    summary = summarize(augmented)
    by_noise = factor_summary(augmented)
    comparator_names = (
        "matrix_pencil_aicc",
        "block_hankel_aic",
        "block_hankel_mdl",
        "shared_prony_aicc",
        "shared_prony_bic",
    )
    return {
        "schema": "P5-Common-Budget-Subspace-Baselines-v1",
        "design": {
            **stage69["design"],
            "evaluation_trials": len(augmented),
            "same_observations_for_all_methods": True,
            "full_coverage_comparators": list(comparator_names),
            "frozen_selective_method": "power_certified_selective",
        },
        "summary": {name: summary[name] for name in (*comparator_names, "power_certified_selective")},
        "by_noise_model": {
            noise: {name: local[name] for name in (*comparator_names, "power_certified_selective")}
            for noise, local in by_noise.items()
        },
        "claim_boundary": (
            "Block-Hankel AIC/MDL treat overlapping columns as effective snapshots and are classical-style "
            "comparators, not exact independent-snapshot likelihoods. Selective accuracy is always reported "
            "with coverage and abstention."
        ),
        "record_storage": (
            "Aggregate-only; decisions are deterministically regenerated while runtime is remeasured."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-summary", action="store_true")
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
