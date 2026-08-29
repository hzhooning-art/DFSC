"""Confirm whether shared channels improve short-horizon memory-rank recovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probe_identifiability_boundary import build_observation
from probe_memory_rank import DEVICE, DTYPE, fit_rank


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def evaluate(ratio: float, channels: int, repeat: int) -> dict:
    horizon = 4.0
    noise_std = 8.0e-4
    seed = 7000 + channels * 100 + int(ratio * 10) + repeat
    times, observations, train_idx, val_idx, true_rates = build_observation(
        ratio, horizon, channels, noise_std, seed
    )
    best = {}
    for rank in (1, 2):
        candidates = [
            fit_rank(
                times,
                observations,
                train_idx,
                val_idx,
                rank,
                seed * 100 + local_seed,
                adam_steps=280,
                lbfgs_steps=80,
            )
            for local_seed in range(2)
        ]
        best[rank] = min(candidates, key=lambda item: item.bic)

    rank1, rank2 = best[1], best[2]
    bic_support = rank1.bic - rank2.bic
    rate_error = float(
        np.linalg.norm(np.log(np.asarray(rank2.rates)) - np.log(np.asarray(true_rates)))
        / np.sqrt(2.0)
    )
    resolved = (
        bic_support >= 6.0
        and rank2.jacobian_condition <= 1.0e8
        and rank2.val_rmse <= max(4.0 * noise_std, 3.0e-3)
        and rate_error <= 0.35
    )
    return {
        "ratio": ratio,
        "channels": channels,
        "repeat": repeat,
        "resolved": resolved,
        "bic_support_rank2": bic_support,
        "rank2_condition": rank2.jacobian_condition,
        "rank2_val_rmse": rank2.val_rmse,
        "relative_log_rate_error": rate_error,
        "estimated_rates": rank2.rates,
    }


def main() -> None:
    records = []
    for ratio in (2.0, 4.0, 8.0):
        for channels in (1, 4, 12):
            for repeat in range(3):
                record = evaluate(ratio, channels, repeat)
                records.append(record)
                print(
                    f"ratio={ratio:3.0f} channels={channels:2d} repeat={repeat} "
                    f"resolved={record['resolved']} rate_error={record['relative_log_rate_error']:.3g}"
                )

    summary = []
    for ratio in (2.0, 4.0, 8.0):
        for channels in (1, 4, 12):
            group = [r for r in records if r["ratio"] == ratio and r["channels"] == channels]
            summary.append(
                {
                    "ratio": ratio,
                    "channels": channels,
                    "trials": len(group),
                    "resolved_fraction": sum(r["resolved"] for r in group) / len(group),
                    "median_bic_support": float(np.median([r["bic_support_rank2"] for r in group])),
                    "median_rate_error": float(np.median([r["relative_log_rate_error"] for r in group])),
                    "median_validation_rmse": float(np.median([r["rank2_val_rmse"] for r in group])),
                }
            )

    scalar = {row["ratio"]: row["resolved_fraction"] for row in summary if row["channels"] == 1}
    shared = {row["ratio"]: row["resolved_fraction"] for row in summary if row["channels"] == 12}
    support_gain = {str(ratio): shared[ratio] - scalar[ratio] for ratio in scalar}
    route_pass = support_gain["4.0"] >= 1.0 / 3.0 and shared[8.0] >= 2.0 / 3.0

    payload = {
        "experiment": "short_horizon_shared_spectrum_gain",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "horizon": 4.0,
        "noise_std": 8.0e-4,
        "records": records,
        "summary": summary,
        "support_gain_12_vs_1": support_gain,
        "route_pass": route_pass,
        "route_pass_rule": "gain at ratio 4 >= 1/3 and 12-channel success at ratio 8 >= 2/3",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "shared_spectrum_gain.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Shared-spectrum short-horizon confirmation",
        "",
        "| Rate ratio | Channels | Resolved fraction | Median BIC support | Median rate error | Median validation RMSE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['ratio']:.0f} | {row['channels']} | {row['resolved_fraction']:.2f} | "
            f"{row['median_bic_support']:.3g} | {row['median_rate_error']:.3g} | "
            f"{row['median_validation_rmse']:.3g} |"
        )
    lines.extend(["", f"Route pass: **{route_pass}**.", ""])
    (RESULTS / "shared_spectrum_gain.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"route_pass={route_pass}")


if __name__ == "__main__":
    main()

