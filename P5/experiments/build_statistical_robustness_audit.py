"""Combine independent-unit statistics and decision sensitivity for public tasks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "statistical_robustness_audit.json"
SUMMARY = RESULTS / "statistical_robustness_audit.md"


def pva_redecision(full: dict, gain_gate: float, stability_gate: float, separation_gate: float) -> str:
    records = full["rank_records"]
    selected, conflict = 1, False
    for rank in (2, 3):
        lower, current = records[str(rank - 1)], records[str(rank)]
        delta_bic = lower["mean_bic"] - current["mean_bic"]
        gain = (lower["median_prediction_nrmse"] - current["median_prediction_nrmse"]) / max(lower["median_prediction_nrmse"], 1e-15)
        checks = (
            delta_bic >= 10.0,
            gain >= gain_gate,
            current["max_log_rate_std"] <= stability_gate,
            current["minimum_rate_ratio"] >= separation_gate,
            current["all_finite"],
        )
        if checks[0]:
            if all(checks) and selected == rank - 1:
                selected = rank
            else:
                conflict = True
    return "INDETERMINATE" if conflict else f"SUPPORTED_RANK_{selected}"


def main() -> None:
    pva = json.loads((RESULTS / "public_pva_relaxation.json").read_text(encoding="utf-8"))
    gas = json.loads((RESULTS / "public_uci_gas_recovery.json").read_text(encoding="utf-8"))
    factors = json.loads((RESULTS / "stage62_boundary_factor_audit.json").read_text(encoding="utf-8"))
    pva_sensitivity = []
    for gain in (0.02, 0.05, 0.10):
        for stability in (0.30, 0.50, 0.80):
            for separation in (1.10, 1.25, 1.50):
                pva_sensitivity.append({
                    "predictive_gain": gain,
                    "max_log_rate_std": stability,
                    "min_rate_ratio": separation,
                    "decision": pva_redecision(pva["full_task"], gain, stability, separation),
                })
    leave_specimen = {}
    for rank, row in pva["full_task"]["rank_records"].items():
        leave_specimen[rank] = [
            {"held_specimen": fold["held_sample"], "rates": fold["rates"], "median_curve_nrmse": sorted(fold["errors"])[len(fold["errors"]) // 2]}
            for fold in row["folds"]
        ]
    payload = {
        "experiment": "stage63_joint_statistical_robustness_audit",
        "independence_units": {
            "pva": "specimen (3 specimens; cycle-level p-values are not used as primary inference)",
            "uci_gas": "exposure experiment (50 experiments; 16 channels clustered within experiment)",
        },
        "pva_cluster_statistics": pva["paired_statistics"],
        "gas_cluster_statistics": gas["statistics"],
        "pva_leave_one_specimen": leave_specimen,
        "gas_leave_one_batch_groups": gas["evaluation"]["groups"],
        "pva_threshold_sensitivity": pva_sensitivity,
        "gas_threshold_sensitivity": gas["threshold_sensitivity"],
        "pva_multistart": factors["optimizer_start_budget"],
        "gas_multistart": gas["multistart_audit"],
        "warnings": [
            "PVA has only three independent specimens; its bootstrap interval is descriptive and cannot support population-level generalization alone.",
            "Gas-sensor channels are repeated measurements within exposure and are clustered before tests.",
            "Threshold sensitivity is a robustness audit, not post-hoc threshold selection.",
        ],
    }
    payload["decision_counts"] = {
        "pva": dict(Counter(row["decision"] for row in pva_sensitivity)),
        "uci_gas": dict(Counter(row["decision"] for row in gas["threshold_sensitivity"])),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Statistical robustness audit", "",
        f"PVA threshold decisions: `{payload['decision_counts']['pva']}`.", "",
        f"UCI gas threshold decisions: `{payload['decision_counts']['uci_gas']}`.", "",
        "Primary inference uses specimens or exposure experiments, never cycles or sensor channels as independent replicates.", "",
        "The UCI task remains indeterminate across every audited threshold combination because fitted rates coalesce.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
