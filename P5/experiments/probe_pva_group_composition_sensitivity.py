"""Stage 73 sensitivity of the Stage 72 PVA result to group composition.

The frozen Stage 72 adapter and Stage 69 certificates are replayed on every
six-curve subset of the nine public PVA curves.  The 84 subsets overlap and
therefore form a deterministic sensitivity analysis, not 84 independent
external experiments.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_public_pva_relaxation as pva  # noqa: E402
from probe_external_power_certificate_transfer import (  # noqa: E402
    TARGET_CHANNELS,
    evaluate_group,
    load_certificates,
)


def curve_label(row) -> str:
    return "sample-%d-cycle-%d" % (row.sample, row.cycle)


def enumerate_groups(rows: list) -> list[tuple[str, list]]:
    ordered = sorted(rows, key=lambda row: (row.sample, row.cycle))
    groups = []
    for indices in itertools.combinations(range(len(ordered)), TARGET_CHANNELS):
        selected = [ordered[index] for index in indices]
        group_id = "__".join(curve_label(row) for row in selected)
        groups.append((group_id, selected))
    return groups


def _distribution(values: list[float]) -> dict:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def run() -> dict:
    groups = enumerate_groups(pva.load_curves())
    certificates = load_certificates()
    records = [
        evaluate_group("pva", group_id, rows, certificates)
        for group_id, rows in groups
    ]
    eligible = [row for row in records if row["scope"]["eligible"]]
    improvements = [row["criterion_improvement_rank1_to_rank2"] for row in eligible]
    noise_bins = Counter(str(row["scope"]["declared_noise_bin"]) for row in eligible)
    noise_models = Counter(row["scope"]["declared_noise_model"] for row in eligible)
    decisions = Counter(row["decision"] for row in records)
    all_checks = sum(all(row["rank_two_checks"].values()) for row in eligible)
    return {
        "schema": "P5-PVA-Group-Composition-Sensitivity-v1",
        "design": {
            "available_curves": len(pva.load_curves()),
            "curves_per_group": TARGET_CHANNELS,
            "enumeration": "all_six_curve_subsets_without_replacement",
            "adapter_or_certificate_retuned": False,
        },
        "summary": {
            "groups": len(records),
            "scope_eligible": len(eligible),
            "decisions": dict(sorted(decisions.items())),
            "declared_noise_bins": dict(sorted(noise_bins.items())),
            "declared_noise_models": dict(sorted(noise_models.items())),
            "all_five_rank_two_checks_passed": all_checks,
            "criterion_improvement_rank1_to_rank2": _distribution(improvements) if improvements else None,
        },
        "records": records,
        "claim_boundary": (
            "The 84 groups are strongly overlapping subsets of one nine-curve PVA dataset. "
            "They measure composition sensitivity of a retrospective result and are not "
            "independent replications or prospective confirmation."
        ),
        "record_storage": "Complete subset-level records included.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-summary", action="store_true")
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
