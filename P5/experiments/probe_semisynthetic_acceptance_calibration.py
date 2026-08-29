"""Stage 65: calibrate acceptance and refusal on known-rank grouped data."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p5_memory_protocol import (  # noqa: E402
    CurveRecord,
    GateConfig,
    decide,
    evaluate,
    fit,
    identifiability_certificate,
    report,
)


OUTPUT = ROOT / "results" / "semisynthetic_acceptance_calibration.json"
SUMMARY = ROOT / "results" / "semisynthetic_acceptance_calibration.md"
FIGURE = ROOT / "figures" / "fig_stage65_acceptance_calibration.pdf"
SEEDS = tuple(range(6501, 6509))


def simulate(seed: int, regime: str, rho: float = 0.55) -> list[CurveRecord]:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 14.0, 64)
    rates = np.asarray((0.12, 0.72) if regime == "separated" else (0.34, 0.37))
    curves = []
    for group_index in range(4):
        for replicate in range(3):
            amplitudes = rng.uniform(0.45, 1.25, size=2)
            value = rng.uniform(-0.08, 0.12) + np.exp(-np.outer(time, rates)) @ amplitudes
            noise = np.zeros(len(time))
            innovation = rng.normal(scale=0.006, size=len(time))
            for index in range(1, len(time)):
                noise[index] = rho * noise[index - 1] + innovation[index]
            curves.append(CurveRecord(
                unit=f"g{group_index}-r{replicate}",
                group=f"g{group_index}",
                channel="response",
                time=time,
                value=value + noise,
            ))
    return curves


def one_run(seed: int, regime: str, nonnegative: bool) -> dict:
    curves = simulate(seed, regime)
    evaluation = evaluate(
        curves,
        starts=2,
        rate_bounds=(0.04, 1.2),
        nonnegative_amplitudes=nonnegative,
    )
    ordinary = decide(evaluation)
    correlated = decide(evaluation, GateConfig(use_ar1_bic=True))
    fitted = fit(
        curves,
        rank=2,
        starts=2,
        rate_bounds=(0.04, 1.2),
        nonnegative_amplitudes=nonnegative,
    )
    certificate = identifiability_certificate(curves, fitted["rates"])
    return {
        "seed": seed,
        "regime": regime,
        "amplitudes": "nonnegative" if nonnegative else "signed",
        "ordinary_decision": ordinary["decision"],
        "ar1_decision": correlated["decision"],
        "fitted_rates": fitted["rates"],
        "residual_rho_ar1": fitted["rho_ar1"],
        "effective_sample_size": fitted["effective_sample_size"],
        "local_boundary_index": certificate["local_boundary_index"],
        "normalized_local_boundary_index": certificate["normalized_local_boundary_index"],
    }


def summarize(rows: list[dict]) -> dict:
    output = {}
    for regime in ("separated", "coalesced"):
        for amplitudes in ("signed", "nonnegative"):
            subset = [row for row in rows if row["regime"] == regime and row["amplitudes"] == amplitudes]
            key = f"{regime}_{amplitudes}"
            output[key] = {
                "n": len(subset),
                "ordinary_decisions": dict(Counter(row["ordinary_decision"] for row in subset)),
                "ar1_decisions": dict(Counter(row["ar1_decision"] for row in subset)),
                "median_rho_ar1": float(np.median([row["residual_rho_ar1"] for row in subset])),
                "median_normalized_boundary_index": float(np.median([
                    row["normalized_local_boundary_index"] for row in subset
                ])),
            }
    calibration_seeds = set(SEEDS[:4])
    evaluation_seeds = set(SEEDS[4:])
    near_calibration = np.asarray([
        row["normalized_local_boundary_index"] for row in rows
        if row["regime"] == "coalesced" and row["amplitudes"] == "nonnegative" and row["seed"] in calibration_seeds
    ])
    separated_calibration = np.asarray([
        row["normalized_local_boundary_index"] for row in rows
        if row["regime"] == "separated" and row["amplitudes"] == "nonnegative" and row["seed"] in calibration_seeds
    ])
    threshold = float(np.sqrt(np.max(near_calibration) * np.min(separated_calibration)))
    near_evaluation = np.asarray([
        row["normalized_local_boundary_index"] for row in rows
        if row["regime"] == "coalesced" and row["amplitudes"] == "nonnegative" and row["seed"] in evaluation_seeds
    ])
    separated_evaluation = np.asarray([
        row["normalized_local_boundary_index"] for row in rows
        if row["regime"] == "separated" and row["amplitudes"] == "nonnegative" and row["seed"] in evaluation_seeds
    ])
    output["boundary_calibration"] = {
        "rule": "geometric midpoint between calibration maxima for coalesced rates and minima for separated rates",
        "threshold": threshold,
        "calibration_seeds": sorted(calibration_seeds),
        "held_evaluation_seeds": sorted(evaluation_seeds),
        "coalesced_false_accept_rate": float(np.mean(near_evaluation > threshold)),
        "coalesced_refusal_rate": float(np.mean(near_evaluation <= threshold)),
        "separated_detection_rate": float(np.mean(separated_evaluation > threshold)),
        "scope": "this declared semi-synthetic design only",
    }
    return output


def plot(rows: list[dict], calibration: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    colors = {"separated": "#2f5aa0", "coalesced": "#d55e6a"}
    for position, regime in enumerate(("separated", "coalesced")):
        values = [row["normalized_local_boundary_index"] for row in rows if row["regime"] == regime and row["amplitudes"] == "nonnegative"]
        axes[0].scatter(np.full(len(values), position) + np.linspace(-0.08, 0.08, len(values)), values, color=colors[regime], alpha=0.8)
    axes[0].axhline(calibration["threshold"], color="black", linestyle="--", label="calibrated threshold")
    axes[0].set_yscale("log")
    axes[0].set_xticks((0, 1), ("Separated rates", "Coalesced rates"))
    axes[0].set_ylabel("Normalized local boundary index")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    labels, ordinary, corrected = [], [], []
    for regime in ("separated", "coalesced"):
        subset = [row for row in rows if row["regime"] == regime and row["amplitudes"] == "nonnegative"]
        labels.append(regime.capitalize())
        desired = "SUPPORTED_RANK_2" if regime == "separated" else "INDETERMINATE"
        ordinary.append(np.mean([row["ordinary_decision"] == desired for row in subset]))
        corrected.append(np.mean([row["ar1_decision"] == desired for row in subset]))
    x = np.arange(2)
    axes[1].bar(x - 0.18, ordinary, width=0.36, color="#8c8c8c", label="Ordinary BIC")
    axes[1].bar(x + 0.18, corrected, width=0.36, color="#e69f00", label="AR(1)-BIC")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Correct support/refusal fraction")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    for label, axis in zip(("a", "b"), axes):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = [
        one_run(seed, regime, nonnegative)
        for regime in ("separated", "coalesced")
        for nonnegative in (False, True)
        for seed in SEEDS
    ]
    summary = summarize(rows)
    payload = {
        "schema_version": "1.0.0",
        "experiment": "stage65_semisynthetic_acceptance_calibration",
        "predeclared_design": {
            "true_rank": 2,
            "separated_rates": [0.12, 0.72],
            "coalesced_rates": [0.34, 0.37],
            "groups": 4,
            "curves_per_group": 3,
            "time_points": 64,
            "noise": "Gaussian AR(1), rho=0.55",
            "seeds": list(SEEDS),
        },
        "rows": rows,
        "summary": summary,
        "claim_boundary": "Threshold calibration is conditional on this declared simulation design.",
    }
    report(payload, OUTPUT)
    plot(rows, summary["boundary_calibration"])
    lines = ["# Semi-synthetic acceptance and refusal calibration", ""]
    for key, value in summary.items():
        lines.extend([f"## {key}", "", f"```json\n{json.dumps(value, indent=2)}\n```", ""])
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
