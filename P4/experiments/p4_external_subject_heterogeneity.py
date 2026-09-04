"""Cluster-aware analysis of the P4 external-subject pilot.

This stage does not create new projects or fault families. It tests whether the
aggregate strategy comparison is stable across interfaces and fault families,
using subject-fault combinations rather than repeated seeds as the unit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p4_external_subject_pilot import (  # noqa: E402
    FAULTS,
    SEEDS,
    STRATEGIES,
    SUBJECTS,
    accepted,
    audit,
)


BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 420_069
STRATEGY_ORDER = tuple(STRATEGIES)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def detection(row: dict, strategy: str) -> bool:
    return row["fault"] != "clean" and not accepted(row, strategy)


def grouped_rates(records: list[dict], field: str) -> dict:
    output = {}
    injected = [row for row in records if row["fault"] != "clean"]
    for value in sorted({row[field] for row in injected}):
        local = [row for row in injected if row[field] == value]
        output[value] = {
            strategy: {
                "trials": len(local),
                "detected": sum(detection(row, strategy) for row in local),
                "detection_rate": float(np.mean([detection(row, strategy) for row in local])),
            }
            for strategy in STRATEGY_ORDER
        }
    return output


def combination_records(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        if row["fault"] != "clean":
            groups[(row["subject"], row["fault"])].append(row)
    return dict(groups)


def cluster_summary(records: list[dict]) -> dict:
    groups = combination_records(records)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    keys = tuple(sorted(groups))
    output = {}
    for strategy in STRATEGY_ORDER:
        rates = np.array([
            np.mean([detection(row, strategy) for row in groups[key]])
            for key in keys
        ])
        draws = rng.integers(0, len(rates), size=(BOOTSTRAP_REPEATS, len(rates)))
        bootstrap = rates[draws].mean(axis=1)
        output[strategy] = {
            "subject_fault_clusters": len(rates),
            "mean_cluster_detection_rate": float(rates.mean()),
            "percentile_95_interval": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "fully_detected_clusters": int(np.sum(rates == 1.0)),
            "undetected_clusters": int(np.sum(rates == 0.0)),
        }
    return output


def paired_increments(records: list[dict]) -> list[dict]:
    groups = combination_records(records)
    output = []
    for earlier, later in zip(STRATEGY_ORDER, STRATEGY_ORDER[1:]):
        earlier_detected = {
            key: all(detection(row, earlier) for row in rows)
            for key, rows in groups.items()
        }
        later_detected = {
            key: all(detection(row, later) for row in rows)
            for key, rows in groups.items()
        }
        gained = [key for key in groups if later_detected[key] and not earlier_detected[key]]
        regressed = [key for key in groups if earlier_detected[key] and not later_detected[key]]
        output.append({
            "earlier": earlier,
            "later": later,
            "newly_fully_detected_clusters": len(gained),
            "regressed_clusters": len(regressed),
            "gained_subject_faults": [list(key) for key in sorted(gained)],
        })
    return output


def leave_one_subject_out(records: list[dict]) -> dict:
    subjects = sorted({row["subject"] for row in records})
    output = {}
    for omitted in subjects:
        local = [row for row in records if row["subject"] != omitted and row["fault"] != "clean"]
        output[omitted] = {
            strategy: float(np.mean([detection(row, strategy) for row in local]))
            for strategy in STRATEGY_ORDER
        }
    return output


def run() -> dict:
    records = [
        audit(subject, seed, fault)
        for subject in SUBJECTS
        for seed in SEEDS
        for fault in (None, *FAULTS)
    ]
    clean = [row for row in records if row["fault"] == "clean"]
    complete_false_rejections = sum(not accepted(row, "execution_evidence_suite") for row in clean)
    return {
        "schema": "P4-External-Subject-Heterogeneity-v1",
        "design": {
            "interfaces": [subject.name for subject in SUBJECTS],
            "fault_families": list(FAULTS),
            "seeds_per_subject_fault": len(SEEDS),
            "primary_unit": "subject-fault combination",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "by_subject": grouped_rates(records, "subject"),
        "by_fault": grouped_rates(records, "fault"),
        "cluster_summary": cluster_summary(records),
        "paired_strategy_increments": paired_increments(records),
        "leave_one_subject_out_detection_rate": leave_one_subject_out(records),
        "clean_control_complete_suite": {
            "trials": len(clean),
            "false_rejections": complete_false_rejections,
            "false_rejection_rate": complete_false_rejections / len(clean),
            "wilson_95_interval": wilson_interval(complete_false_rejections, len(clean)),
        },
        "jss_readiness_gate": {
            "minimum_independent_sut_projects": 3,
            "observed_independent_sut_projects": 1,
            "historical_defect_families_required": True,
            "historical_defect_families_observed": 0,
            "passes": False,
        },
        "claim_boundary": (
            "This is a cluster-aware reanalysis of one PyTorch-vendor pilot. It tests interface and "
            "fault-family heterogeneity but supplies no new independent project or historical defect."
        ),
        "record_storage": "Aggregate-only; source trial decisions are deterministically regenerated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-summary", action="store_true")
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
