"""Map the empirical refusal boundary around the positive-real memory contract.

The generator labels are never passed to the decision rule.  They are used only
after fitting to score whether acceptance changes sensibly as the violation grows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, FitResult, fit_rank, lifted_response
from probe_out_of_class_refusal import mean_lag1, oscillatory_response, prediction_from_fit


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPEATS = 3


def generate_observations(family: str, strength: float, seed: int):
    rng = np.random.default_rng(seed)
    channels = 6
    noise_std = 6.0e-4
    times = torch.linspace(0.0, 14.0, 97, dtype=DTYPE, device=DEVICE)
    scale = torch.linspace(0.72, 1.28, channels, dtype=DTYPE, device=DEVICE)

    if family == "signed_residue":
        # strength is the magnitude of a residue whose sign violates the contract.
        weights = torch.stack([0.46 * scale, -strength / scale], dim=1)
        clean = lifted_response(
            times,
            weights,
            torch.tensor([0.22, 1.35], dtype=DTYPE, device=DEVICE),
        )
    elif family == "oscillation":
        # frequency=0 is an in-class rank-one positive-real control.
        clean = oscillatory_response(times, 0.38 * scale, decay=0.24, frequency=strength)
    else:
        raise ValueError(f"Unknown family: {family}")

    observations = clean + noise_std * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )
    train_pool = np.arange(1, 70)
    train_np = np.sort(rng.choice(train_pool, size=48, replace=False))
    val_np = np.arange(70, times.numel())
    return (
        times,
        observations,
        torch.tensor(train_np, dtype=torch.long, device=DEVICE),
        torch.tensor(val_np, dtype=torch.long, device=DEVICE),
        noise_std,
    )


def evaluate(family: str, strength: float, repeat: int) -> dict:
    seed = 12000 + repeat + int(round(strength * 10000)) + sum(map(ord, family))
    times, observations, train_idx, val_idx, noise_std = generate_observations(
        family, strength, seed
    )
    fits: list[FitResult] = []
    for rank in (1, 2, 3, 4):
        candidates = [
            fit_rank(
                times,
                observations,
                train_idx,
                val_idx,
                rank,
                seed * 100 + local_seed,
                adam_steps=220,
                lbfgs_steps=65,
            )
            for local_seed in range(2)
        ]
        fits.append(min(candidates, key=lambda item: item.bic))

    winner = min(fits, key=lambda item: item.bic)
    residual = prediction_from_fit(times, winner)[val_idx] - observations[val_idx]
    lag1 = mean_lag1(residual)
    rank3 = next(fit for fit in fits if fit.rank == 3)
    rank4 = next(fit for fit in fits if fit.rank == 4)
    cap_gain = rank3.bic - rank4.bic

    checks = {
        "prediction_ok": winner.val_rmse <= max(4.0 * noise_std, 3.0e-3),
        "condition_ok": winner.jacobian_condition <= 1.0e8,
        "residual_ok": abs(lag1) <= 0.55,
    }
    cap_saturated = winner.rank == 4 and cap_gain >= 6.0
    accepted = all(checks.values()) and not cap_saturated
    failed_gates = [name for name, passed in checks.items() if not passed]
    if cap_saturated:
        failed_gates.append("rank_cap")
    return {
        "family": family,
        "strength": strength,
        "repeat": repeat,
        "decision": "ACCEPT_CONTRACT" if accepted else "REFUSE_CONTRACT",
        "selected_rank": winner.rank,
        "validation_rmse": winner.val_rmse,
        "jacobian_condition": winner.jacobian_condition,
        "validation_residual_lag1": lag1,
        "failed_gates": failed_gates,
        "candidate_bic": {str(fit.rank): fit.bic for fit in fits},
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    settings = sorted({(row["family"], row["strength"]) for row in records})
    for family, strength in settings:
        group = [
            row for row in records
            if row["family"] == family and row["strength"] == strength
        ]
        rows.append(
            {
                "family": family,
                "strength": strength,
                "accept_fraction": float(np.mean([
                    row["decision"] == "ACCEPT_CONTRACT" for row in group
                ])),
                "median_validation_rmse": float(np.median([
                    row["validation_rmse"] for row in group
                ])),
                "median_abs_residual_lag1": float(np.median([
                    abs(row["validation_residual_lag1"]) for row in group
                ])),
                "selected_ranks": [row["selected_rank"] for row in group],
                "failed_gate_counts": {
                    gate: sum(gate in row["failed_gates"] for row in group)
                    for gate in ("prediction_ok", "condition_ok", "residual_ok", "rank_cap")
                },
            }
        )
    return rows


def boundary_assessment(summary: list[dict]) -> dict:
    assessment = {}
    for family in ("signed_residue", "oscillation"):
        rows = sorted(
            [row for row in summary if row["family"] == family],
            key=lambda row: row["strength"],
        )
        fractions = [row["accept_fraction"] for row in rows]
        assessment[family] = {
            "zero_control_accepted": fractions[0] >= 2.0 / 3.0,
            "strongest_violation_refused": fractions[-1] <= 1.0 / 3.0,
            "acceptance_nonincreasing": all(
                fractions[index + 1] <= fractions[index] + 1.0 / 3.0
                for index in range(len(fractions) - 1)
            ),
            "first_majority_refusal_strength": next(
                (row["strength"] for row in rows if row["accept_fraction"] <= 1.0 / 3.0),
                None,
            ),
        }
    route_pass = all(
        item["zero_control_accepted"]
        and item["strongest_violation_refused"]
        and item["acceptance_nonincreasing"]
        for item in assessment.values()
    )
    return {"families": assessment, "route_pass": route_pass}


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "refusal_boundary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Near-boundary contract-refusal probe",
        "",
        "Labels define the post-hoc scoring axis only; the decision rule uses fit diagnostics.",
        "",
        "| Family | Violation strength | Accept fraction | Median validation RMSE | Median |lag-1| | Selected ranks |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['family']} | {row['strength']:.4g} | {row['accept_fraction']:.2f} | "
            f"{row['median_validation_rmse']:.3g} | {row['median_abs_residual_lag1']:.3g} | "
            f"{row['selected_ranks']} |"
        )
    lines.extend([
        "",
        f"Route pass: **{payload['assessment']['route_pass']}**.",
        "",
        "This is an empirical boundary scan with three repeats per setting, not a calibrated",
        "type-I/type-II error guarantee.",
    ])
    (RESULTS / "refusal_boundary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    grids = {
        "signed_residue": (0.0, 0.01, 0.025, 0.05, 0.08, 0.13),
        "oscillation": (0.0, 0.05, 0.10, 0.20, 0.40, 0.80),
    }
    records = []
    for family, strengths in grids.items():
        for strength in strengths:
            for repeat in range(REPEATS):
                record = evaluate(family, strength, repeat)
                records.append(record)
                print(
                    f"family={family:14s} strength={strength:5.3f} repeat={repeat} "
                    f"decision={record['decision']:15s} rank={record['selected_rank']} "
                    f"rmse={record['validation_rmse']:.3g} "
                    f"lag1={record['validation_residual_lag1']:.3g}"
                )
    summary = summarize(records)
    payload = {
        "experiment": "near_boundary_positive_real_contract_refusal",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "repeats_per_setting": REPEATS,
        "violation_grids": grids,
        "records": records,
        "summary": summary,
        "assessment": boundary_assessment(summary),
    }
    write_outputs(payload)
    print(json.dumps(payload["assessment"], indent=2))


if __name__ == "__main__":
    main()
