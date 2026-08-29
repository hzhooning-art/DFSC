"""Refine the contract-refusal transition left unresolved by the coarse scan."""

from __future__ import annotations

import json

from probe_memory_rank import DEVICE, DTYPE
from probe_refusal_boundary import RESULTS, boundary_assessment, evaluate, summarize


def main() -> None:
    grids = {
        "signed_residue": (0.0, 0.001, 0.0025, 0.005, 0.0075, 0.010),
        "oscillation": (0.0, 0.010, 0.020, 0.030, 0.040, 0.050),
    }
    records = []
    for family, strengths in grids.items():
        for strength in strengths:
            for repeat in range(3):
                row = evaluate(family, strength, repeat)
                records.append(row)
                print(
                    f"family={family:14s} strength={strength:6.4f} repeat={repeat} "
                    f"decision={row['decision']:15s} rank={row['selected_rank']} "
                    f"rmse={row['validation_rmse']:.3g} lag1={row['validation_residual_lag1']:.3g}",
                    flush=True,
                )

    summary = summarize(records)
    payload = {
        "experiment": "refined_near_boundary_positive_real_contract_refusal",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "repeats_per_setting": 3,
        "violation_grids": grids,
        "records": records,
        "summary": summary,
        "assessment": boundary_assessment(summary),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "refusal_boundary_refined.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Refined near-boundary refusal scan",
        "",
        "| Family | Violation strength | Accept fraction | Median validation RMSE | Median |lag-1| | Selected ranks |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['family']} | {row['strength']:.4g} | {row['accept_fraction']:.2f} | "
            f"{row['median_validation_rmse']:.3g} | {row['median_abs_residual_lag1']:.3g} | "
            f"{row['selected_ranks']} |"
        )
    lines.extend([
        "",
        f"Route pass: **{payload['assessment']['route_pass']}**.",
        "",
        "The transition is empirical and depends on the declared noise, horizon, sampling,",
        "candidate-rank cap, and diagnostic thresholds.",
    ])
    (RESULTS / "refusal_boundary_refined.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["assessment"], indent=2), flush=True)


if __name__ == "__main__":
    main()
