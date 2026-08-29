"""Independent validation of a frozen optimizer-budget consensus rule.

Stage 50 demonstrated that a single refinement budget can reverse the
scientific decision.  This stage permits a determinate decision only when at
least two preregistered budgets agree and no budget gives the opposite binary
decision.  Conflicting retain/refuse outcomes are exposed as budget-sensitive
abstentions rather than resolved by post-hoc budget selection.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from probe_decomposed_tolerance_transfer import (
    CHANNEL_COUNTS,
    HETEROGENEITY_CONSTRUCTIONS,
    LOG_SPECTRAL_DRIFTS,
)
from probe_extended_refinement_transfer import ADAM_STEPS, load_frozen_stage48
from probe_high_dimensional_shared_spectrum import DEVICE, DTYPE
from probe_noise_scale_optimizer_transfer import PROJECT_NOISE_STDS, PROJECT_REPEATS
from probe_optimizer_budget_stability import (
    LBFGS_BUDGETS,
    decision_class,
    project_record_for_budget,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROJECT_SEED_BASE = 151000
MIN_VOTES = 2
MAX_FALSE_REFUSAL_FRACTION = 1.0 / 48.0
MIN_EXACT_MILD_RETAIN_FRACTION = 0.50
MIN_SEVERE_REFUSAL_FRACTION = 0.75
MIN_CELL_SEVERE_REFUSAL_FRACTION = 2.0 / 3.0
MAX_BUDGET_SENSITIVE_FRACTION = 0.35


def consensus_decision(classes: list[str]) -> tuple[str, str]:
    """Return a conservative consensus class and an auditable reason."""
    counts = Counter(classes)
    if counts["RETAIN"] and counts["REFUSE"]:
        return "INDETERMINATE", "BUDGET_SENSITIVE_BINARY_CONFLICT"
    if counts["RETAIN"] >= MIN_VOTES:
        return "RETAIN", "CONSENSUS_RETAIN"
    if counts["REFUSE"] >= MIN_VOTES:
        return "REFUSE", "CONSENSUS_REFUSE"
    return "INDETERMINATE", "INSUFFICIENT_DETERMINATE_VOTES"


def summarize(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["channels"],
            record["noise_std_diagnostic"],
            record["heterogeneity_construction"],
            record["log_spectral_drift"],
            record["repeat"],
            record["seed"],
        )
        grouped[key].append(record)

    pairs = []
    for key, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: item["lbfgs_steps"])
        classes = [item["decision_class"] for item in ordered]
        consensus, reason = consensus_decision(classes)
        pairs.append(
            {
                "channels": key[0],
                "noise_std": key[1],
                "heterogeneity_construction": key[2],
                "log_spectral_drift": key[3],
                "repeat": key[4],
                "seed": key[5],
                "budgets": [item["lbfgs_steps"] for item in ordered],
                "budget_decisions": [item["decision"] for item in ordered],
                "budget_classes": classes,
                "consensus_class": consensus,
                "consensus_reason": reason,
                "budget_sensitive": reason == "BUDGET_SENSITIVE_BINARY_CONFLICT",
            }
        )

    cell_groups = defaultdict(list)
    for pair in pairs:
        cell_groups[
            (
                pair["channels"],
                pair["noise_std"],
                pair["heterogeneity_construction"],
                pair["log_spectral_drift"],
            )
        ].append(pair)
    cells = []
    for key, items in sorted(cell_groups.items()):
        cells.append(
            {
                "channels": key[0],
                "noise_std": key[1],
                "heterogeneity_construction": key[2],
                "log_spectral_drift": key[3],
                "trials": len(items),
                "retain_fraction": float(
                    np.mean([item["consensus_class"] == "RETAIN" for item in items])
                ),
                "refuse_fraction": float(
                    np.mean([item["consensus_class"] == "REFUSE" for item in items])
                ),
                "indeterminate_fraction": float(
                    np.mean(
                        [item["consensus_class"] == "INDETERMINATE" for item in items]
                    )
                ),
                "budget_sensitive_fraction": float(
                    np.mean([item["budget_sensitive"] for item in items])
                ),
            }
        )

    exact_mild = [
        pair for pair in pairs if pair["log_spectral_drift"] in (0.0, 0.05)
    ]
    severe = [pair for pair in pairs if pair["log_spectral_drift"] == 0.15]
    false_refusal_fraction = float(
        np.mean([pair["consensus_class"] == "REFUSE" for pair in exact_mild])
    )
    exact_mild_retain_fraction = float(
        np.mean([pair["consensus_class"] == "RETAIN" for pair in exact_mild])
    )
    severe_refusal_fraction = float(
        np.mean([pair["consensus_class"] == "REFUSE" for pair in severe])
    )
    budget_sensitive_fraction = float(
        np.mean([pair["budget_sensitive"] for pair in pairs])
    )
    severe_cells = [cell for cell in cells if cell["log_spectral_drift"] == 0.15]
    checks = {
        "complete_independent_matrix": len(pairs) == 72
        and all(len(items) == len(LBFGS_BUDGETS) for items in grouped.values()),
        "false_refusal_fraction_at_most_1_of_48": false_refusal_fraction
        <= MAX_FALSE_REFUSAL_FRACTION,
        "exact_mild_retain_fraction_at_least_0_50": exact_mild_retain_fraction
        >= MIN_EXACT_MILD_RETAIN_FRACTION,
        "severe_refusal_fraction_at_least_0_75": severe_refusal_fraction
        >= MIN_SEVERE_REFUSAL_FRACTION,
        "each_severe_cell_refusal_fraction_at_least_2_of_3": all(
            cell["refuse_fraction"] >= MIN_CELL_SEVERE_REFUSAL_FRACTION
            for cell in severe_cells
        ),
        "budget_sensitive_fraction_at_most_0_35": budget_sensitive_fraction
        <= MAX_BUDGET_SENSITIVE_FRACTION,
        "all_diagnostics_in_calibration_scope": all(
            record["diagnostics_in_calibration_scope"] for record in records
        ),
    }
    return {
        "pairs": pairs,
        "cells": cells,
        "false_refusal_fraction": false_refusal_fraction,
        "exact_mild_retain_fraction": exact_mild_retain_fraction,
        "severe_refusal_fraction": severe_refusal_fraction,
        "budget_sensitive_fraction": budget_sensitive_fraction,
        "indeterminate_fraction": float(
            np.mean([pair["consensus_class"] == "INDETERMINATE" for pair in pairs])
        ),
        "checks": checks,
        "route_pass": all(checks.values()),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "budget_consensus_abstention.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    summary = payload["summary"]
    lines = [
        "# Independent optimizer-budget consensus and abstention audit",
        "",
        f"Device: `{payload['device']}`; route pass: **{summary['route_pass']}**.",
        "",
        f"Exact/mild false refusal: {summary['false_refusal_fraction']:.3f}; "
        f"exact/mild retained: {summary['exact_mild_retain_fraction']:.3f}; "
        f"severe refused: {summary['severe_refusal_fraction']:.3f}; "
        f"budget-sensitive: {summary['budget_sensitive_fraction']:.3f}.",
        "",
        "| Channels | Noise std | Construction | Drift | Retain | Refuse | Indeterminate | Budget-sensitive |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cells"]:
        lines.append(
            f"| {row['channels']} | {row['noise_std']:.1e} | "
            f"{row['heterogeneity_construction']} | "
            f"{row['log_spectral_drift']:.2f} | {row['retain_fraction']:.2f} | "
            f"{row['refuse_fraction']:.2f} | {row['indeterminate_fraction']:.2f} | "
            f"{row['budget_sensitive_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The rule, budgets, thresholds, and exit conditions were frozen before the independent project seeds were evaluated.",
        ]
    )
    (RESULTS / "budget_consensus_abstention.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    noise_calibration, consistency_calibration = load_frozen_stage48()
    records = []
    for channels in CHANNEL_COUNTS:
        for noise_std in PROJECT_NOISE_STDS:
            for construction in HETEROGENEITY_CONSTRUCTIONS:
                for drift in LOG_SPECTRAL_DRIFTS:
                    for repeat in range(PROJECT_REPEATS):
                        for budget in LBFGS_BUDGETS:
                            record = project_record_for_budget(
                                channels,
                                noise_std,
                                construction,
                                repeat,
                                budget,
                                noise_calibration,
                                consistency_calibration,
                                log_spectral_drift=drift,
                                project_seed_base=PROJECT_SEED_BASE,
                            )
                            records.append(record)
                            print(
                                f"channels={channels} noise={noise_std:.1e} "
                                f"construction={construction} drift={drift:.2f} "
                                f"repeat={repeat} budget={budget} "
                                f"decision={record['decision']}",
                                flush=True,
                            )
    summary = summarize(records)
    payload = {
        "experiment": "budget_consensus_abstention",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "protocol": {
            "project_seed_base": PROJECT_SEED_BASE,
            "adam_steps": ADAM_STEPS,
            "lbfgs_budgets": list(LBFGS_BUDGETS),
            "minimum_consensus_votes": MIN_VOTES,
            "conflicting_binary_votes_force_abstention": True,
            "log_spectral_drifts": list(LOG_SPECTRAL_DRIFTS),
            "paired_datasets": 72,
            "total_budget_evaluations": 216,
        },
        "frozen_noise_calibration": noise_calibration,
        "frozen_consistency_calibration": consistency_calibration,
        "records": records,
        "summary": summary,
        "exit_rule": {
            "maximum_false_refusal_fraction": MAX_FALSE_REFUSAL_FRACTION,
            "minimum_exact_mild_retain_fraction": MIN_EXACT_MILD_RETAIN_FRACTION,
            "minimum_severe_refusal_fraction": MIN_SEVERE_REFUSAL_FRACTION,
            "minimum_cell_severe_refusal_fraction": MIN_CELL_SEVERE_REFUSAL_FRACTION,
            "maximum_budget_sensitive_fraction": MAX_BUDGET_SENSITIVE_FRACTION,
            "failure_action": "retain budget sensitivity as an unresolved computational-identifiability boundary",
        },
    }
    write_outputs(payload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
