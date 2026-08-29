"""Map a first identifiability boundary for shared latent memory rank.

This route-selection experiment varies pole separation, observation horizon, and
channel count. It asks whether a true rank-two memory realization is supported by
the observations; it does not claim global identifiability.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank, lifted_response


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def build_observation(
    ratio: float,
    horizon: float,
    channels: int,
    noise_std: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    rng = np.random.default_rng(seed)
    slow_rate = 0.25
    rates = [slow_rate, slow_rate * ratio]
    times = torch.linspace(0.0, horizon, 65, dtype=DTYPE, device=DEVICE)

    amplitude = np.linspace(0.28, 0.78, channels)[:, None]
    channel_tilt = np.linspace(0.80, 1.20, channels)[:, None]
    weights_np = np.concatenate(
        [0.58 * amplitude / channel_tilt, 0.42 * amplitude * channel_tilt], axis=1
    )
    weights = torch.tensor(weights_np, dtype=DTYPE, device=DEVICE)
    clean = lifted_response(
        times,
        weights,
        torch.tensor(rates, dtype=DTYPE, device=DEVICE),
    )
    observations = clean + noise_std * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )

    split = 48
    train_pool = np.arange(1, split)
    train_np = np.sort(rng.choice(train_pool, size=34, replace=False))
    val_np = np.arange(split, times.numel())
    train_idx = torch.tensor(train_np, dtype=torch.long, device=DEVICE)
    val_idx = torch.tensor(val_np, dtype=torch.long, device=DEVICE)
    return times, observations, train_idx, val_idx, rates


def evaluate_setting(ratio: float, horizon: float, channels: int, seed: int) -> dict:
    noise_std = 8.0e-4
    times, observations, train_idx, val_idx, true_rates = build_observation(
        ratio, horizon, channels, noise_std, seed
    )
    fits = []
    for rank in (1, 2):
        candidates = [
            fit_rank(
                times,
                observations,
                train_idx,
                val_idx,
                rank,
                seed * 100 + local_seed,
                adam_steps=190,
                lbfgs_steps=55,
            )
            for local_seed in range(2)
        ]
        fits.append(min(candidates, key=lambda item: item.bic))

    rank1, rank2 = fits
    bic_support = rank1.bic - rank2.bic
    recovered = np.asarray(rank2.rates)
    relative_rate_error = float(
        np.linalg.norm(np.log(recovered) - np.log(np.asarray(true_rates))) / np.sqrt(2.0)
    )
    prediction_ok = rank2.val_rmse <= max(4.0 * noise_std, 3.0e-3)
    condition_ok = rank2.jacobian_condition <= 1.0e8
    evidence_ok = bic_support >= 6.0
    rate_ok = relative_rate_error <= 0.35

    if evidence_ok and condition_ok and prediction_ok and rate_ok:
        status = "RANK_2_RESOLVED"
    elif rank1.bic + 6.0 < rank2.bic:
        status = "EFFECTIVE_RANK_1_ONLY"
    else:
        status = "INSUFFICIENT_EVIDENCE"

    return {
        "ratio": ratio,
        "horizon": horizon,
        "channels": channels,
        "seed": seed,
        "status": status,
        "bic_support_rank2": bic_support,
        "rank2_condition": rank2.jacobian_condition,
        "rank2_val_rmse": rank2.val_rmse,
        "relative_log_rate_error": relative_rate_error,
        "estimated_rates": rank2.rates,
        "rank1_bic": rank1.bic,
        "rank2_bic": rank2.bic,
    }


def summarize(records: list[dict]) -> list[dict]:
    summary = []
    keys = sorted({(r["ratio"], r["horizon"], r["channels"]) for r in records})
    for ratio, horizon, channels in keys:
        group = [
            r for r in records
            if r["ratio"] == ratio and r["horizon"] == horizon and r["channels"] == channels
        ]
        resolved = sum(r["status"] == "RANK_2_RESOLVED" for r in group)
        summary.append(
            {
                "ratio": ratio,
                "horizon": horizon,
                "channels": channels,
                "trials": len(group),
                "resolved_fraction": resolved / len(group),
                "median_bic_support": float(np.median([r["bic_support_rank2"] for r in group])),
                "median_rate_error": float(np.median([r["relative_log_rate_error"] for r in group])),
                "statuses": {status: sum(r["status"] == status for r in group) for status in (
                    "RANK_2_RESOLVED", "EFFECTIVE_RANK_1_ONLY", "INSUFFICIENT_EVIDENCE"
                )},
            }
        )
    return summary


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "identifiability_boundary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Initial identifiability boundary",
        "",
        f"Device: `{payload['device']}`; noise standard deviation: `{payload['noise_std']}`.",
        "",
        "Each cell reports the fraction of two repeated trials in which rank two was",
        "resolved with adequate BIC support, conditioning, extrapolation error, and pole recovery.",
        "",
    ]
    for channels in (1, 8):
        lines.extend(
            [
                f"## Channels = {channels}",
                "",
                "| Horizon | Rate ratio 1.10 | 1.35 | 2.00 | 4.00 | 8.00 |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for horizon in (4.0, 8.0, 12.0):
            values = []
            for ratio in (1.10, 1.35, 2.00, 4.00, 8.00):
                row = next(
                    item for item in payload["summary"]
                    if item["channels"] == channels
                    and item["horizon"] == horizon
                    and item["ratio"] == ratio
                )
                values.append(f"{row['resolved_fraction']:.2f}")
            lines.append(f"| {horizon:.0f} | " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(
        [
            "The map is a low-budget feasibility screen, not a calibrated statistical",
            "coverage result. A publishable boundary requires many more seeds and explicit",
            "false-discovery control.",
        ]
    )
    (RESULTS / "identifiability_boundary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = []
    for channels in (1, 8):
        for horizon in (4.0, 8.0, 12.0):
            for ratio in (1.10, 1.35, 2.00, 4.00, 8.00):
                for repeat in range(2):
                    seed = 1000 + channels * 100 + int(horizon) * 10 + int(ratio * 10) + repeat
                    result = evaluate_setting(ratio, horizon, channels, seed)
                    records.append(result)
                    print(
                        f"channels={channels:2d} horizon={horizon:4.1f} ratio={ratio:4.2f} "
                        f"repeat={repeat} status={result['status']}"
                    )
    payload = {
        "experiment": "shared_memory_rank_identifiability_boundary",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "noise_std": 8.0e-4,
        "records": records,
        "summary": summarize(records),
    }
    write_outputs(payload)


if __name__ == "__main__":
    main()

