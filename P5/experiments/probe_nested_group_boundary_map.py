"""Map the refusal boundary of the frozen nested grouped-sharing gate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from probe_nested_group_sharing_gate import evaluate


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BOUNDARY_DRIFTS = (0.075, 0.100, 0.125)
NOISE_CORRELATIONS = (0.0, 0.60)
REPEATS = 5


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def interpolate_boundary(rows: list[dict], noise_correlation: float) -> float | None:
    """Linearly interpolate the first 50% refusal crossing."""
    selected = sorted(
        (row for row in rows if row["noise_correlation"] == noise_correlation),
        key=lambda row: row["log_spectral_drift"],
    )
    for left, right in zip(selected, selected[1:]):
        p0 = left["refuse_fraction"]
        p1 = right["refuse_fraction"]
        if p0 == 0.5:
            return float(left["log_spectral_drift"])
        if p0 <= 0.5 <= p1 and p1 > p0:
            d0 = left["log_spectral_drift"]
            d1 = right["log_spectral_drift"]
            return float(d0 + (0.5 - p0) * (d1 - d0) / (p1 - p0))
    if selected and selected[-1]["refuse_fraction"] == 0.5:
        return float(selected[-1]["log_spectral_drift"])
    return None


def summarize(records: list[dict]) -> dict:
    rows = []
    for drift in BOUNDARY_DRIFTS:
        for rho in NOISE_CORRELATIONS:
            group = [
                record
                for record in records
                if record["log_spectral_drift"] == drift
                and record["noise_correlation"] == rho
            ]
            refused = sum(record["decision"] == "REFUSE_SHARED_MECHANISM" for record in group)
            interval = wilson_interval(refused, len(group))
            rows.append(
                {
                    "log_spectral_drift": drift,
                    "noise_correlation": rho,
                    "trials": len(group),
                    "refused": refused,
                    "refuse_fraction": refused / len(group),
                    "refuse_fraction_ci95": list(interval),
                    "median_group_bic_support": float(np.median([r["group_bic_support"] for r in group])),
                    "median_shared_val_rmse": float(np.median([r["shared_val_rmse"] for r in group])),
                    "median_validation_ratio": float(
                        np.median([r["shared_to_grouped_val_ratio"] for r in group])
                    ),
                }
            )

    by_rho = {
        rho: [row for row in rows if row["noise_correlation"] == rho]
        for rho in NOISE_CORRELATIONS
    }
    monotone = {
        str(rho): all(
            left["refuse_fraction"] <= right["refuse_fraction"]
            for left, right in zip(by_rho[rho], by_rho[rho][1:])
        )
        for rho in NOISE_CORRELATIONS
    }
    correlation_differences = [
        abs(
            next(r["refuse_fraction"] for r in rows if r["log_spectral_drift"] == drift and r["noise_correlation"] == 0.0)
            - next(r["refuse_fraction"] for r in rows if r["log_spectral_drift"] == drift and r["noise_correlation"] == 0.60)
        )
        for drift in BOUNDARY_DRIFTS
    ]
    boundaries = {
        str(rho): interpolate_boundary(rows, rho) for rho in NOISE_CORRELATIONS
    }
    low_mean = float(np.mean([r["refuse_fraction"] for r in rows if r["log_spectral_drift"] == 0.075]))
    high_mean = float(np.mean([r["refuse_fraction"] for r in rows if r["log_spectral_drift"] == 0.125]))
    checks = {
        "complete_matrix": len(records) == len(BOUNDARY_DRIFTS) * len(NOISE_CORRELATIONS) * REPEATS,
        "monotone_by_noise_regime": all(monotone.values()),
        "transition_bracketed": low_mean <= 0.40 and high_mean >= 0.60,
        "noise_sensitivity_bounded": max(correlation_differences) <= 0.40,
    }
    return {
        "rows": rows,
        "boundary_estimates": boundaries,
        "maximum_noise_regime_difference": max(correlation_differences),
        "checks": checks,
        "boundary_map_pass": bool(all(checks.values())),
        "frozen_interpretation_rule": {
            "low_drift_mean_refusal_max": 0.40,
            "high_drift_mean_refusal_min": 0.60,
            "maximum_noise_regime_difference": 0.40,
            "monotonicity_required": True,
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "nested_group_boundary_map.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Nested grouped-sharing refusal boundary",
        "",
        f"Device: `{payload['device']}`; boundary-map pass: **{payload['summary']['boundary_map_pass']}**.",
        "",
        "| Log drift | Noise corr. | Refused | Refusal fraction (95% Wilson CI) | BIC support | Shared RMSE | Val. ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["rows"]:
        lo, hi = row["refuse_fraction_ci95"]
        lines.append(
            f"| {row['log_spectral_drift']:.3f} | {row['noise_correlation']:.2f} | "
            f"{row['refused']}/{row['trials']} | {row['refuse_fraction']:.2f} [{lo:.2f}, {hi:.2f}] | "
            f"{row['median_group_bic_support']:.3g} | {row['median_shared_val_rmse']:.3g} | "
            f"{row['median_validation_ratio']:.3g} |"
        )
    lines.extend(
        [
            "",
            "The thresholds and decision rule are inherited unchanged from the preceding nested-gate experiment.",
        ]
    )
    (RESULTS / "nested_group_boundary_map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = []
    for drift in BOUNDARY_DRIFTS:
        for rho in NOISE_CORRELATIONS:
            for repeat in range(REPEATS):
                record = evaluate(drift, rho, repeat)
                records.append(record)
                print(
                    f"drift={drift:.3f} rho={rho:.2f} repeat={repeat} "
                    f"decision={record['decision']} support={record['group_bic_support']:.3g} "
                    f"shared_rmse={record['shared_val_rmse']:.3g}",
                    flush=True,
                )
    summary = summarize(records)
    payload = {
        "experiment": "nested_group_boundary_map",
        "device": records[0].get("device", "cuda") if records else "unknown",
        "protocol": {
            "log_spectral_drifts": list(BOUNDARY_DRIFTS),
            "noise_correlations": list(NOISE_CORRELATIONS),
            "repeats": REPEATS,
            "inherited_decision_rule": "probe_nested_group_sharing_gate.py",
        },
        "records": records,
        "summary": summary,
    }
    write_outputs(payload)
    print(json.dumps({"boundary_map_pass": summary["boundary_map_pass"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
