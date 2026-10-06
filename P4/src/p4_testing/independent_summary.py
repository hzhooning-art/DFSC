"""Publication-safe summaries for complete independent baseline logs."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .independent_eval import audit_independent_runs


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def summarize_complete_runs(
    rows: Sequence[Mapping[str, Any]], case_ids: Sequence[str], strategies: Sequence[str],
    preregistration_sha256: str | None = None,
) -> dict[str, Any]:
    audit = audit_independent_runs(rows, case_ids, strategies, preregistration_sha256)
    if not audit["complete"]:
        raise ValueError("independent baseline audit is incomplete")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(row)
    rates = {}
    for strategy in strategies:
        subset = grouped[strategy]
        detected = sum(bool(row["detected"]) for row in subset)
        low, high = wilson_interval(detected, len(subset))
        rates[strategy] = {"detected": detected, "total": len(subset),
                           "rate": detected / len(subset), "wilson_95": [low, high],
                           "mean_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in subset) / len(subset),
                           "mean_subject_invocations": sum(int(row["subject_invocations"]) for row in subset) / len(subset)}
    pairwise = {}
    by_case = {case_id: {str(row["strategy"]): row for row in rows if row["case_id"] == case_id} for case_id in case_ids}
    for left_index, left in enumerate(strategies):
        for right in strategies[left_index + 1:]:
            left_wins = right_wins = ties = 0
            for case_id in case_ids:
                a, b = bool(by_case[case_id][left]["detected"]), bool(by_case[case_id][right]["detected"])
                if a and not b: left_wins += 1
                elif b and not a: right_wins += 1
                else: ties += 1
            pairwise[f"{left}__vs__{right}"] = {"left_wins": left_wins, "right_wins": right_wins, "ties": ties}
    return {"schema": "P4-Independent-Baseline-Summary-v1", "audit": audit, "rates": rates, "pairwise": pairwise}
