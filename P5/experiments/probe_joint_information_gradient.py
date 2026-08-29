"""Map joint memory-rank selection and refusal across information regimes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probe_joint_rank_refusal_calibration import evaluate
from probe_memory_rank import DEVICE, DTYPE
from probe_refusal_calibration import RESULTS, wilson_interval


REPEATS = 6
REGIMES = (
    {"name": "low_information", "horizon": 8.0, "noise_std": 1.2e-3},
    {"name": "medium_information", "horizon": 14.0, "noise_std": 6.0e-4},
    {"name": "high_information", "horizon": 20.0, "noise_std": 3.0e-4},
)
LEVELS = {
    "signed_residue": {"zero": 0.0, "below": 0.005, "above": 0.0075},
    "oscillation": {"zero": 0.0, "below": 0.03, "above": 0.04},
}


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    keys = sorted({
        (row["family"], row["level"], row["regime"])
        for row in records
    })
    for family, level, regime in keys:
        group = [
            row for row in records
            if (row["family"], row["level"], row["regime"])
            == (family, level, regime)
        ]
        refused = sum(row["decision"] == "REFUSE_CONTRACT" for row in group)
        elevated = sum(row["selected_rank"] > 1 for row in group)
        absorbed = sum(
            row["decision"] == "ACCEPT_CONTRACT" and row["selected_rank"] > 1
            for row in group
        )
        rows.append({
            "family": family,
            "level": level,
            "regime": regime,
            "horizon": group[0]["horizon"],
            "noise_std": group[0]["noise_std"],
            "trials": len(group),
            "refusal_fraction": refused / len(group),
            "refusal_wilson95": wilson_interval(refused, len(group)),
            "elevated_rank_fraction": elevated / len(group),
            "elevated_rank_wilson95": wilson_interval(elevated, len(group)),
            "absorbed_violation_fraction": absorbed / len(group),
            "selected_ranks": [row["selected_rank"] for row in group],
            "median_validation_rmse": float(np.median([
                row["validation_rmse"] for row in group
            ])),
            "median_abs_residual_lag1": float(np.median([
                abs(row["validation_residual_lag1"]) for row in group
            ])),
        })
    return rows


def assess(summary: list[dict]) -> dict:
    lookup = {
        (row["family"], row["level"], row["regime"]): row
        for row in summary
    }
    family_results = {}
    order = [regime["name"] for regime in REGIMES]
    for family in LEVELS:
        zero_rows = [lookup[(family, "zero", regime)] for regime in order]
        below_rows = [lookup[(family, "below", regime)] for regime in order]
        above_rows = [lookup[(family, "above", regime)] for regime in order]
        family_results[family] = {
            "zero_false_refusal_at_most_0.20": all(
                row["refusal_fraction"] <= 0.20 for row in zero_rows
            ),
            "zero_false_elevated_rank_at_most_0.20": all(
                row["elevated_rank_fraction"] <= 0.20 for row in zero_rows
            ),
            "above_refusal_monotone_with_information": all(
                left["refusal_fraction"] <= right["refusal_fraction"]
                for left, right in zip(above_rows, above_rows[1:])
            ),
            "above_not_below_within_each_regime": all(
                above["refusal_fraction"] >= below["refusal_fraction"]
                for above, below in zip(above_rows, below_rows)
            ),
            "high_information_above_refusal_at_least_0.80": (
                above_rows[-1]["refusal_fraction"] >= 0.80
            ),
            "accepted_elevated_rank_absorption_at_most_0.20": all(
                row["absorbed_violation_fraction"] <= 0.20
                for row in below_rows + above_rows
            ),
            "rates": {
                regime: {
                    "zero": lookup[(family, "zero", regime)]["refusal_fraction"],
                    "below": lookup[(family, "below", regime)]["refusal_fraction"],
                    "above": lookup[(family, "above", regime)]["refusal_fraction"],
                }
                for regime in order
            },
        }
    return {
        "families": family_results,
        "route_pass": all(
            all(value for key, value in result.items() if key != "rates")
            for result in family_results.values()
        ),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS / "joint_information_gradient.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Joint information-gradient calibration",
        "",
        "Candidate ranks 1--3 compete at every operating point.",
        "",
        "| Family | Level | Regime | H | Noise | Refusal (Wilson 95%) | Elevated rank |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lo, hi = row["refusal_wilson95"]
        lines.append(
            f"| {row['family']} | {row['level']} | {row['regime']} | "
            f"{row['horizon']:.0f} | {row['noise_std']:.1e} | "
            f"{row['refusal_fraction']:.2f} [{lo:.2f}, {hi:.2f}] | "
            f"{row['elevated_rank_fraction']:.2f} |"
        )
    lines.extend([
        "",
        f"Route pass: **{payload['assessment']['route_pass']}**.",
        "",
        "The regimes vary horizon and noise together and therefore establish an",
        "information-gradient result, not separate causal effects of either variable.",
    ])
    (RESULTS / "joint_information_gradient.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    records = []
    for regime in REGIMES:
        for family, levels in LEVELS.items():
            for level, strength in levels.items():
                for repeat in range(REPEATS):
                    row = evaluate(
                        family,
                        level,
                        strength,
                        repeat + 100,
                        horizon=regime["horizon"],
                        noise_std=regime["noise_std"],
                    )
                    row["regime"] = regime["name"]
                    records.append(row)
                    print(
                        f"regime={regime['name']:18s} family={family:14s} "
                        f"level={level:5s} repeat={repeat} "
                        f"decision={row['decision']:15s} rank={row['selected_rank']} "
                        f"lag1={row['validation_residual_lag1']:.3g}",
                        flush=True,
                    )
    summary = summarize(records)
    payload = {
        "experiment": "joint_rank_refusal_information_gradient",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "candidate_ranks": [1, 2, 3],
        "starts_per_rank": 2,
        "repeats_per_setting": REPEATS,
        "regimes": REGIMES,
        "generator_labels_used_only_for_scoring": LEVELS,
        "records": records,
        "summary": summary,
        "assessment": assess(summary),
    }
    write_outputs(payload)
    print(json.dumps(payload["assessment"], indent=2), flush=True)


if __name__ == "__main__":
    main()
