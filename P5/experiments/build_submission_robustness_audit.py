"""Submission hardening audit for grouped units and noise sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "submission_robustness_audit.json"
SUMMARY = RESULTS / "submission_robustness_audit.md"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def bootstrap_rate(successes: np.ndarray, seed: int, repeats: int = 4000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(successes), size=(repeats, len(successes)))
    rates = np.mean(successes[indices], axis=1)
    return [float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))]


def transfer_summary(calibration: dict) -> dict:
    threshold = calibration["calibration"]["threshold"]
    output = {}
    for noise_index, (noise, rows) in enumerate(calibration["transfer_rows"].items()):
        separated = np.asarray([
            row["normalized_local_boundary_index"] > threshold
            for row in rows if row["regime"] == "separated"
        ], dtype=float)
        coalesced = np.asarray([
            row["normalized_local_boundary_index"] <= threshold
            for row in rows if row["regime"] == "coalesced"
        ], dtype=float)
        output[noise] = {
            "frozen_threshold": threshold,
            "separated_support": {
                **calibration["transfer_evaluation"][noise]["separated_support"],
                "seed_cluster_bootstrap_95": bootstrap_rate(separated, 9100 + noise_index),
            },
            "coalesced_refusal": {
                **calibration["transfer_evaluation"][noise]["coalesced_refusal"],
                "seed_cluster_bootstrap_95": bootstrap_rate(coalesced, 9200 + noise_index),
            },
        }
    return output


def preferred_rank(records: dict, key: str) -> int:
    available = [(int(rank), row[key]) for rank, row in records.items() if key in row]
    return min(available, key=lambda item: item[1])[0]


def main() -> None:
    pva = load("public_pva_relaxation.json")
    gas = load("public_uci_gas_recovery.json")
    hydraulic = load("public_uci_hydraulic_transients.json")
    calibration = load("submission_calibration_transfer.json")
    hydraulic_records = hydraulic["evaluation"]["rank_records"]
    ordinary_rank = preferred_rank(hydraulic_records, "mean_bic")
    ar1_rank = preferred_rank(hydraulic_records, "mean_ar1_bic")

    payload = {
        "schema_version": "1.0.0",
        "experiment": "submission_grouping_and_noise_robustness_audit",
        "public_task_audit": {
            "pva": {
                "independent_unit": "physical specimen",
                "independent_unit_count": int(pva["source"]["specimens"]),
                "grouped_measurements": "loading cycles within specimen",
                "split_unit": "leave one specimen out",
                "leakage_controls": [
                    "all cycles from a held specimen remain outside calibration",
                    "cycle-level measurements are not counted as independent inferential units",
                ],
            },
            "gas_sensor": {
                "independent_unit": "non-air exposure experiment",
                "independent_unit_count": int(gas["source"]["independent_non_air_experiments"]),
                "grouped_measurements": "16 sensor channels within exposure",
                "split_unit": "held acquisition batch",
                "leakage_controls": [
                    "sensor channels remain clustered within their exposure",
                    "all exposures from a held acquisition batch remain outside calibration",
                ],
            },
            "hydraulic": {
                "independent_unit": "load cycle",
                "independent_unit_count": int(hydraulic["source"]["independent_cycles"]),
                "grouped_measurements": "four measured channels within cycle",
                "split_unit": "held cooler-condition group",
                "leakage_controls": [
                    "channels remain clustered within their load cycle",
                    "all cycles from a held cooler condition remain outside calibration",
                ],
            },
        },
        "noise_generator_transfer": transfer_summary(calibration),
        "correlated_noise_model_sensitivity": {
            "hydraulic": {
                "ordinary_bic_preferred_rank": ordinary_rank,
                "ar1_profile_bic_preferred_rank": ar1_rank,
                "criterion_disagreement": ordinary_rank != ar1_rank,
                "ordinary_bic_by_rank": {
                    rank: row["mean_bic"] for rank, row in hydraulic_records.items()
                },
                "ar1_profile_bic_by_rank": {
                    rank: row["mean_ar1_bic"] for rank, row in hydraulic_records.items()
                },
                "interpretation": (
                    "The disagreement is a residual-correlation sensitivity result; "
                    "neither criterion overrides the frozen transfer and separation gates."
                ),
            }
        },
        "changes_primary_public_decisions": False,
        "scope_warnings": [
            "The PVA task contains only three independent specimens.",
            "Seed-cluster bootstrap intervals quantify repeated simulation uncertainty, not population transfer.",
            "AR(1)-profile BIC is a sensitivity model and does not cover arbitrary colored or heteroscedastic residuals.",
            "Sequential acquisition can induce dependence beyond the declared grouping variables.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Submission grouping and noise robustness audit",
        "",
        "Primary public-task decisions are unchanged by this audit.",
        "",
        "| Task | Independent unit | Count | Split unit |",
        "|---|---|---:|---|",
    ]
    for name, row in payload["public_task_audit"].items():
        lines.append(f"| {name} | {row['independent_unit']} | {row['independent_unit_count']} | {row['split_unit']} |")
    lines.extend([
        "",
        f"Hydraulic ordinary BIC prefers rank {ordinary_rank}; AR(1)-profile BIC prefers rank {ar1_rank}.",
        "This disagreement is retained as sensitivity evidence and does not replace the frozen refusal decision.",
        "",
    ])
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "public_task_audit": payload["public_task_audit"],
        "hydraulic_sensitivity": payload["correlated_noise_model_sensitivity"]["hydraulic"],
    }, indent=2))


if __name__ == "__main__":
    main()
