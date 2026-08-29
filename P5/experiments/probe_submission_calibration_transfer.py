"""Submission hardening: large-sample calibration and noise-transfer audit."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p5_memory_protocol import CurveRecord, fit, identifiability_certificate, report  # noqa: E402


OUTPUT = ROOT / "results" / "submission_calibration_transfer.json"
SUMMARY = ROOT / "results" / "submission_calibration_transfer.md"
CALIBRATION_SEEDS = tuple(range(6601, 6621))
EVALUATION_SEEDS = tuple(range(6701, 6741))
NOISE_GENERATORS = ("iid_gaussian", "ar1", "ar2", "heteroscedastic")


def _noise(rng: np.random.Generator, time: np.ndarray, kind: str, sigma: float = 0.006) -> np.ndarray:
    if kind == "iid_gaussian":
        return rng.normal(scale=sigma, size=len(time))
    if kind == "ar1":
        rho = 0.55
        innovations = rng.normal(scale=sigma * math.sqrt(1.0 - rho**2), size=len(time))
        output = np.zeros(len(time))
        for index in range(1, len(time)):
            output[index] = rho * output[index - 1] + innovations[index]
        return output
    if kind == "ar2":
        phi1, phi2 = 0.65, -0.20
        innovations = rng.normal(scale=sigma * 0.75, size=len(time))
        output = np.zeros(len(time))
        for index in range(2, len(time)):
            output[index] = phi1 * output[index - 1] + phi2 * output[index - 2] + innovations[index]
        return output
    if kind == "heteroscedastic":
        scale = sigma * (0.55 + 0.90 * time / max(float(time[-1]), 1e-12))
        return rng.normal(scale=scale, size=len(time))
    raise ValueError(f"unknown noise generator: {kind}")


def simulate(seed: int, regime: str, noise_generator: str) -> list[CurveRecord]:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 14.0, 64)
    rates = np.asarray((0.12, 0.72) if regime == "separated" else (0.34, 0.37))
    curves: list[CurveRecord] = []
    for group_index in range(4):
        for replicate in range(3):
            amplitudes = rng.uniform(0.45, 1.25, size=2)
            signal = rng.uniform(-0.08, 0.12) + np.exp(-np.outer(time, rates)) @ amplitudes
            curves.append(
                CurveRecord(
                    unit=f"g{group_index}-r{replicate}",
                    group=f"g{group_index}",
                    channel="response",
                    time=time,
                    value=signal + _noise(rng, time, noise_generator),
                )
            )
    return curves


def one_run(seed: int, regime: str, noise_generator: str) -> dict:
    curves = simulate(seed, regime, noise_generator)
    fitted = fit(
        curves,
        rank=2,
        starts=2,
        rate_bounds=(0.04, 1.2),
        nonnegative_amplitudes=True,
    )
    certificate = identifiability_certificate(curves, fitted["rates"])
    return {
        "seed": seed,
        "regime": regime,
        "noise_generator": noise_generator,
        "fitted_rates": fitted["rates"],
        "fit_rmse": math.sqrt(fitted["sse"] / fitted["n_observations"]),
        "residual_rho_ar1": fitted["rho_ar1"],
        "effective_sample_size": fitted["effective_sample_size"],
        "normalized_local_boundary_index": certificate["normalized_local_boundary_index"],
    }


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [float("nan"), float("nan")]
    probability = successes / n
    denominator = 1.0 + z**2 / n
    center = (probability + z**2 / (2.0 * n)) / denominator
    half_width = z * math.sqrt(probability * (1.0 - probability) / n + z**2 / (4.0 * n**2)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def metric(rows: list[dict], threshold: float, expected: str) -> dict:
    if expected == "support":
        successes = sum(row["normalized_local_boundary_index"] > threshold for row in rows)
    elif expected == "refusal":
        successes = sum(row["normalized_local_boundary_index"] <= threshold for row in rows)
    else:
        raise ValueError(expected)
    return {
        "successes": successes,
        "n": len(rows),
        "rate": successes / len(rows),
        "wilson_95": wilson(successes, len(rows)),
    }


def calibrate(rows: list[dict]) -> dict:
    separated = [row["normalized_local_boundary_index"] for row in rows if row["regime"] == "separated"]
    coalesced = [row["normalized_local_boundary_index"] for row in rows if row["regime"] == "coalesced"]
    lower_support = min(separated)
    upper_refusal = max(coalesced)
    threshold = float(math.sqrt(max(lower_support, 1e-300) * max(upper_refusal, 1e-300)))
    return {
        "rule": "geometric midpoint between the smallest separated and largest coalesced calibration indices",
        "threshold": threshold,
        "smallest_separated_index": lower_support,
        "largest_coalesced_index": upper_refusal,
        "calibration_gap_ratio": lower_support / max(upper_refusal, 1e-300),
        "calibration_overlap": bool(lower_support <= upper_refusal),
    }


def main() -> None:
    calibration_rows = [
        one_run(seed, regime, "ar1")
        for regime in ("separated", "coalesced")
        for seed in CALIBRATION_SEEDS
    ]
    calibration = calibrate(calibration_rows)
    threshold = calibration["threshold"]

    transfer_rows: dict[str, list[dict]] = {}
    transfer_evaluation = {}
    for noise_generator in NOISE_GENERATORS:
        rows = [
            one_run(seed, regime, noise_generator)
            for regime in ("separated", "coalesced")
            for seed in EVALUATION_SEEDS
        ]
        transfer_rows[noise_generator] = rows
        separated = [row for row in rows if row["regime"] == "separated"]
        coalesced = [row for row in rows if row["regime"] == "coalesced"]
        transfer_evaluation[noise_generator] = {
            "frozen_threshold": threshold,
            "threshold_source": "AR(1) calibration seeds only",
            "separated_support": metric(separated, threshold, "support"),
            "coalesced_refusal": metric(coalesced, threshold, "refusal"),
        }

    principal = transfer_evaluation["ar1"]
    payload = {
        "schema_version": "1.0.0",
        "experiment": "submission_calibration_and_noise_transfer",
        "predeclared_design": {
            "true_rank": 2,
            "separated_rates": [0.12, 0.72],
            "coalesced_rates": [0.34, 0.37],
            "groups": 4,
            "curves_per_group": 3,
            "time_points": 64,
            "calibration_noise_generator": "ar1",
            "calibration_seeds": list(CALIBRATION_SEEDS),
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "noise_generators": list(NOISE_GENERATORS),
            "threshold_retuning_on_evaluation": False,
        },
        "calibration": calibration,
        "principal_evaluation": {
            "separated_support": principal["separated_support"],
            "coalesced_refusal": principal["coalesced_refusal"],
        },
        "transfer_evaluation": transfer_evaluation,
        "calibration_rows": calibration_rows,
        "transfer_rows": transfer_rows,
        "claim_boundary": "The frozen threshold is design conditional and is not a universal identifiability boundary.",
    }
    report(payload, OUTPUT)

    lines = [
        "# Submission calibration and noise-transfer audit",
        "",
        f"Frozen threshold: `{threshold:.8g}` (AR(1) calibration only).",
        "",
        "| Noise generator | Separated support | 95% Wilson | Coalesced refusal | 95% Wilson |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in transfer_evaluation.items():
        support = row["separated_support"]
        refusal = row["coalesced_refusal"]
        lines.append(
            f"| {name} | {support['successes']}/{support['n']} | "
            f"[{support['wilson_95'][0]:.3f}, {support['wilson_95'][1]:.3f}] | "
            f"{refusal['successes']}/{refusal['n']} | "
            f"[{refusal['wilson_95'][0]:.3f}, {refusal['wilson_95'][1]:.3f}] |"
        )
    lines.extend(["", f"Claim boundary: {payload['claim_boundary']}", ""])
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"calibration": calibration, "transfer_evaluation": transfer_evaluation}, indent=2))


if __name__ == "__main__":
    main()
