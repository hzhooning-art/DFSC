"""Jointly calibrate rank selection and near-boundary contract refusal."""

from __future__ import annotations

import json

import numpy as np

from probe_memory_rank import DEVICE, DTYPE, fit_rank
from probe_out_of_class_refusal import mean_lag1, prediction_from_fit
from probe_refusal_calibration import RESULTS, generate, wilson_interval


REPEATS = 10
HORIZON = 14.0
NOISE_STD = 6.0e-4


def evaluate(
    family: str,
    level: str,
    strength: float,
    repeat: int,
    horizon: float = HORIZON,
    noise_std: float = NOISE_STD,
) -> dict:
    seed = (
        23000
        + repeat
        + int(round(strength * 100000))
        + int(round(horizon * 10))
        + int(round(noise_std * 1.0e7))
        + sum(map(ord, family))
    )
    times, observations, train_idx, val_idx = generate(
        family, strength, horizon, noise_std, seed
    )
    fits = []
    for rank in (1, 2, 3):
        candidates = [
            fit_rank(
                times,
                observations,
                train_idx,
                val_idx,
                rank=rank,
                seed=seed * 100 + start,
                adam_steps=210,
                lbfgs_steps=60,
            )
            for start in range(2)
        ]
        fits.append(min(candidates, key=lambda item: item.bic))

    winner = min(fits, key=lambda item: item.bic)
    residual = prediction_from_fit(times, winner)[val_idx] - observations[val_idx]
    lag1 = mean_lag1(residual)
    rank2 = next(fit for fit in fits if fit.rank == 2)
    rank3 = next(fit for fit in fits if fit.rank == 3)
    cap_gain = rank2.bic - rank3.bic

    checks = {
        "prediction": winner.val_rmse <= max(4.0 * noise_std, 3.0e-3),
        "condition": winner.jacobian_condition <= 1.0e8,
        "residual": abs(lag1) <= 0.55,
    }
    rank_cap = winner.rank == 3 and cap_gain >= 6.0
    accepted = all(checks.values()) and not rank_cap
    return {
        "family": family,
        "level": level,
        "strength": strength,
        "repeat": repeat,
        "horizon": horizon,
        "noise_std": noise_std,
        "decision": "ACCEPT_CONTRACT" if accepted else "REFUSE_CONTRACT",
        "selected_rank": winner.rank,
        "validation_rmse": winner.val_rmse,
        "jacobian_condition": winner.jacobian_condition,
        "validation_residual_lag1": lag1,
        "rank3_vs_rank2_bic_gain": cap_gain,
        "failed_gates": [name for name, passed in checks.items() if not passed]
        + (["rank_cap"] if rank_cap else []),
        "candidate_bic": {str(fit.rank): fit.bic for fit in fits},
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    keys = sorted({(row["family"], row["level"], row["strength"]) for row in records})
    for family, level, strength in keys:
        group = [
            row for row in records
            if (row["family"], row["level"], row["strength"])
            == (family, level, strength)
        ]
        refusals = sum(row["decision"] == "REFUSE_CONTRACT" for row in group)
        elevated = sum(row["selected_rank"] > 1 for row in group)
        refusal_ci = wilson_interval(refusals, len(group))
        elevated_ci = wilson_interval(elevated, len(group))
        rows.append({
            "family": family,
            "level": level,
            "strength": strength,
            "trials": len(group),
            "refusal_fraction": refusals / len(group),
            "refusal_wilson95": refusal_ci,
            "elevated_rank_fraction": elevated / len(group),
            "elevated_rank_wilson95": elevated_ci,
            "selected_ranks": [row["selected_rank"] for row in group],
            "median_validation_rmse": float(np.median([row["validation_rmse"] for row in group])),
            "median_abs_residual_lag1": float(np.median([
                abs(row["validation_residual_lag1"]) for row in group
            ])),
            "failed_gate_counts": {
                gate: sum(gate in row["failed_gates"] for row in group)
                for gate in ("prediction", "condition", "residual", "rank_cap")
            },
        })
    return rows


def assess(summary: list[dict]) -> dict:
    lookup = {(row["family"], row["level"]): row for row in summary}
    families = {}
    for family in ("signed_residue", "oscillation"):
        zero = lookup[(family, "zero")]
        below = lookup[(family, "below")]
        above = lookup[(family, "above")]
        families[family] = {
            "zero_false_refusal_at_most_0.20": zero["refusal_fraction"] <= 0.20,
            "zero_false_elevated_rank_at_most_0.20": zero["elevated_rank_fraction"] <= 0.20,
            "above_refusal_not_less_than_below": (
                above["refusal_fraction"] >= below["refusal_fraction"]
            ),
            "rates": {
                "zero_refusal": zero["refusal_fraction"],
                "zero_elevated_rank": zero["elevated_rank_fraction"],
                "below_refusal": below["refusal_fraction"],
                "above_refusal": above["refusal_fraction"],
            },
        }
    return {
        "families": families,
        "route_pass": all(all(
            value for key, value in family_result.items() if key != "rates"
        ) for family_result in families.values()),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "joint_rank_refusal_calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Joint rank-selection and refusal calibration",
        "",
        f"Horizon: `{HORIZON}`; noise standard deviation: `{NOISE_STD}`.",
        "",
        "| Family | Level | Trials | Refusal fraction | Elevated-rank fraction | Selected ranks |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['family']} | {row['level']} | {row['trials']} | "
            f"{row['refusal_fraction']:.2f} | {row['elevated_rank_fraction']:.2f} | "
            f"{row['selected_ranks']} |"
        )
    lines.extend([
        "",
        f"Route pass: **{payload['assessment']['route_pass']}**.",
        "",
        "Ten repeats per setting narrow the route uncertainty but do not provide a final",
        "uniform error guarantee over noise, horizon, and candidate-rank choices.",
    ])
    (RESULTS / "joint_rank_refusal_calibration.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    levels = {
        "signed_residue": {"zero": 0.0, "below": 0.005, "above": 0.0075},
        "oscillation": {"zero": 0.0, "below": 0.03, "above": 0.04},
    }
    records = []
    for family, family_levels in levels.items():
        for level, strength in family_levels.items():
            for repeat in range(REPEATS):
                row = evaluate(family, level, strength, repeat)
                records.append(row)
                print(
                    f"family={family:14s} level={level:5s} repeat={repeat:2d} "
                    f"decision={row['decision']:15s} rank={row['selected_rank']} "
                    f"rmse={row['validation_rmse']:.3g} "
                    f"lag1={row['validation_residual_lag1']:.3g}",
                    flush=True,
                )
    summary = summarize(records)
    payload = {
        "experiment": "joint_rank_selection_and_contract_refusal_calibration",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "horizon": HORIZON,
        "noise_std": NOISE_STD,
        "candidate_ranks": [1, 2, 3],
        "starts_per_rank": 2,
        "repeats_per_setting": REPEATS,
        "generator_labels_used_only_for_scoring": levels,
        "records": records,
        "summary": summary,
        "assessment": assess(summary),
    }
    write_outputs(payload)
    print(json.dumps(payload["assessment"], indent=2), flush=True)


if __name__ == "__main__":
    main()
