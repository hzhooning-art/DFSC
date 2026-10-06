"""Completeness and duplicate checks for fresh baseline execution logs."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from .independent_logs import validate_independent_run


BASELINE_STRATEGIES = ("example", "tolerance", "property", "metamorphic", "differential")


def audit_independent_runs(
    rows: Iterable[Mapping[str, Any]],
    case_ids: Iterable[str],
    strategies: Iterable[str] = BASELINE_STRATEGIES,
    preregistration_sha256: str | None = None,
) -> dict[str, Any]:
    rows = list(rows)
    expected = {(case_id, strategy) for case_id in case_ids for strategy in strategies}
    valid_rows = [row for row in rows if validate_independent_run(row)
                  and (preregistration_sha256 is None or row["preregistration_sha256"] == preregistration_sha256)]
    keys = [(row["case_id"], row["strategy"]) for row in valid_rows]
    counts = Counter(keys)
    observed = set(keys)
    duplicates = sorted([list(key) for key, count in counts.items() if count > 1])
    missing = sorted([list(key) for key in expected - observed])
    unexpected = sorted([list(key) for key in observed - expected])
    environments: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in valid_rows:
        environments[str(row["case_id"])].add((str(row["runner_sha256"]), str(row["environment_sha256"])))
    inconsistent_cases = sorted(case_id for case_id, hashes in environments.items() if len(hashes) != 1)
    complete = (not missing and not duplicates and not unexpected and not inconsistent_cases
                and len(valid_rows) == len(expected))
    return {
        "schema": "P4-Independent-Baseline-Audit-v1",
        "expected_rows": len(expected),
        "valid_rows": len(valid_rows),
        "invalid_rows": sum(1 for row in rows if not validate_independent_run(row)),
        "missing": missing,
        "duplicates": duplicates,
        "unexpected": unexpected,
        "inconsistent_cases": inconsistent_cases,
        "preregistration_sha256": preregistration_sha256,
        "complete": complete,
        "claim_status": "ADMISSIBLE_FOR_COMPARISON" if complete else "BLOCKED_PENDING_COMPLETE_FRESH_RUNS",
    }
