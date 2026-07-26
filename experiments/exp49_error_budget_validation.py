"""Validate certified and empirical components of the dfsc error budget."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dfsc


def main() -> None:
    torch.set_default_dtype(torch.float64)
    output = ROOT / "revision_results"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in (0.35, 0.55, 0.8, 1.0):
        for radius in (0.05, 0.2, 0.5, 1.0, 2.0):
            z = torch.tensor([-radius])
            for terms in (12, 20, 32):
                approximate = dfsc.mittag_leffler_e(alpha, z, terms=terms)
                reference = dfsc.mittag_leffler_e(alpha, z, terms=180)
                actual = float(torch.abs(approximate - reference))
                bound_t = dfsc.alternating_series_remainder_bound(alpha, z, terms=terms)
                bound = None if bound_t is None else float(bound_t.max())
                rows.append(
                    {
                        "alpha": alpha,
                        "abs_z": radius,
                        "terms": terms,
                        "actual_error": actual,
                        "certified_bound": bound,
                        "covered": bound is not None and actual <= bound * (1 + 1e-10) + 1e-15,
                    }
                )
    certified = [row for row in rows if row["certified_bound"] is not None]
    payload = {
        "scope": "negative-real alternating-series certificate; not a global hybrid-evaluator bound",
        "cases": len(rows),
        "certified_cases": len(certified),
        "coverage_fraction": sum(bool(row["covered"]) for row in certified) / len(certified),
        "median_effectivity_for_resolved_errors": sorted(
            row["certified_bound"] / row["actual_error"]
            for row in certified if row["actual_error"] > 1e-14
        )[len([row for row in certified if row["actual_error"] > 1e-14]) // 2],
        "note": "effectivity excludes actual errors below 1e-14, where float64 cancellation dominates the ratio",
        "rows": rows,
    }
    (output / "error_budget_validation.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
