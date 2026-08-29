"""Resolve the frozen nested sharing gate inside the 0.05--0.075 drift band."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probe_nested_group_boundary_map import wilson_interval
from probe_nested_group_sharing_gate import evaluate


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DRIFTS = (0.05000, 0.05625, 0.06250, 0.06875, 0.07500)
NOISE_CORRELATIONS = (0.0, 0.60)
REPEATS = 6
REPEAT_OFFSET = 100


def crossing_bracket(rows: list[dict]) -> dict | None:
    """Bracket and interpolate the first empirical 50% refusal crossing."""
    ordered = sorted(rows, key=lambda row: row["log_spectral_drift"])
    for left, right in zip(ordered, ordered[1:]):
        p0 = left["refuse_fraction"]
        p1 = right["refuse_fraction"]
        if p0 < 0.5 <= p1:
            d0 = left["log_spectral_drift"]
            d1 = right["log_spectral_drift"]
            estimate = d1 if p1 == p0 else d0 + (0.5 - p0) * (d1 - d0) / (p1 - p0)
            return {"lower": d0, "upper": d1, "linear_estimate": float(estimate)}
    if ordered and ordered[0]["refuse_fraction"] == 0.5:
        drift = ordered[0]["log_spectral_drift"]
        return {"lower": drift, "upper": drift, "linear_estimate": drift}
    return None


def summarize(records: list[dict]) -> dict:
    regime_rows = []
    pooled_rows = []
    for drift in DRIFTS:
        pooled = [record for record in records if record["log_spectral_drift"] == drift]
        pooled_refused = sum(record["decision"] == "REFUSE_SHARED_MECHANISM" for record in pooled)
        pooled_ci = wilson_interval(pooled_refused, len(pooled))
        pooled_rows.append(
            {
                "log_spectral_drift": drift,
                "trials": len(pooled),
                "refused": pooled_refused,
                "refuse_fraction": pooled_refused / len(pooled),
                "refuse_fraction_ci95": list(pooled_ci),
                "median_shared_val_rmse": float(np.median([r["shared_val_rmse"] for r in pooled])),
                "median_group_bic_support": float(np.median([r["group_bic_support"] for r in pooled])),
            }
        )
        for rho in NOISE_CORRELATIONS:
            group = [record for record in pooled if record["noise_correlation"] == rho]
            refused = sum(record["decision"] == "REFUSE_SHARED_MECHANISM" for record in group)
            ci = wilson_interval(refused, len(group))
            regime_rows.append(
                {
                    "log_spectral_drift": drift,
                    "noise_correlation": rho,
                    "trials": len(group),
                    "refused": refused,
                    "refuse_fraction": refused / len(group),
                    "refuse_fraction_ci95": list(ci),
                    "median_shared_val_rmse": float(np.median([r["shared_val_rmse"] for r in group])),
                    "median_group_bic_support": float(np.median([r["group_bic_support"] for r in group])),
                }
            )

    differences = []
    for drift in DRIFTS:
        fractions = [
            row["refuse_fraction"]
            for row in regime_rows
            if row["log_spectral_drift"] == drift
        ]
        differences.append(abs(fractions[0] - fractions[1]))
    bracket = crossing_bracket(pooled_rows)
    pooled_fractions = [row["refuse_fraction"] for row in pooled_rows]
    checks = {
        "complete_matrix": len(records) == len(DRIFTS) * len(NOISE_CORRELATIONS) * REPEATS,
        "empirical_refusal_is_monotone": all(
            left <= right for left, right in zip(pooled_fractions, pooled_fractions[1:])
        ),
        "lower_endpoint_retained": pooled_fractions[0] <= 0.25,
        "upper_endpoint_reaches_majority_refusal": pooled_fractions[-1] >= 0.50,
        "fifty_percent_crossing_bracketed": bracket is not None,
        "noise_regime_difference_bounded": max(differences) <= 0.50,
    }
    return {
        "pooled_rows": pooled_rows,
        "noise_regime_rows": regime_rows,
        "crossing_bracket": bracket,
        "maximum_noise_regime_difference": max(differences),
        "checks": checks,
        "dense_boundary_pass": bool(all(checks.values())),
        "frozen_rule": {
            "lower_endpoint_refusal_max": 0.25,
            "upper_endpoint_refusal_min": 0.50,
            "maximum_noise_regime_difference": 0.50,
            "empirical_monotonicity_required": True,
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dense_nested_group_boundary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Dense nested sharing-gate boundary",
        "",
        f"Device: `{payload['device']}`; dense-boundary pass: **{payload['summary']['dense_boundary_pass']}**.",
        "",
        "| Log drift | Refused | Refusal fraction (95% Wilson CI) | Shared RMSE | BIC support |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["pooled_rows"]:
        low, high = row["refuse_fraction_ci95"]
        lines.append(
            f"| {row['log_spectral_drift']:.5f} | {row['refused']}/{row['trials']} | "
            f"{row['refuse_fraction']:.2f} [{low:.2f}, {high:.2f}] | "
            f"{row['median_shared_val_rmse']:.4g} | {row['median_group_bic_support']:.3g} |"
        )
    bracket = payload["summary"]["crossing_bracket"]
    if bracket is None:
        lines.extend(["", "The tested grid did not bracket a 50% refusal crossing."])
    else:
        lines.extend(
            [
                "",
                f"Empirical 50% crossing bracket: [{bracket['lower']:.5f}, {bracket['upper']:.5f}]; "
                f"piecewise-linear estimate: {bracket['linear_estimate']:.5f}.",
            ]
        )
    lines.extend(["", "The inherited gate and thresholds were not modified for this experiment."])
    (RESULTS / "dense_nested_group_boundary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = []
    for drift in DRIFTS:
        for rho in NOISE_CORRELATIONS:
            for local_repeat in range(REPEATS):
                repeat = REPEAT_OFFSET + local_repeat
                record = evaluate(drift, rho, repeat)
                record["local_repeat"] = local_repeat
                records.append(record)
                print(
                    f"drift={drift:.5f} rho={rho:.2f} repeat={local_repeat} "
                    f"decision={record['decision']} shared_rmse={record['shared_val_rmse']:.4g} "
                    f"support={record['group_bic_support']:.3g}",
                    flush=True,
                )
    summary = summarize(records)
    payload = {
        "experiment": "dense_nested_group_boundary",
        "device": "cuda",
        "protocol": {
            "log_spectral_drifts": list(DRIFTS),
            "noise_correlations": list(NOISE_CORRELATIONS),
            "repeats_per_cell": REPEATS,
            "repeat_offset": REPEAT_OFFSET,
            "inherited_decision_rule": "probe_nested_group_sharing_gate.py",
        },
        "records": records,
        "summary": summary,
    }
    write_outputs(payload)
    print(json.dumps({"dense_boundary_pass": summary["dense_boundary_pass"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
