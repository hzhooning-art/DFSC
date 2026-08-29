"""Falsification audit for approximate spectral sharing under correlated noise.

The experiment asks whether an observable subgroup-spectrum diagnostic can
reject a globally shared mechanism after channel spectra become heterogeneous.
It maps a controlled boundary; it does not establish a universal field test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from probe_high_dimensional_shared_spectrum import (
    DTYPE,
    DEVICE,
    fit_candidate,
    independent_lifted_response,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHANNELS = 64
BLOCKS = 4
LOG_SPECTRAL_DRIFTS = (0.0, 0.05, 0.15)
NOISE_CORRELATIONS = (0.0, 0.60)
REPEATS = 3
TRUE_CENTRAL_RATES = np.asarray([0.25, 1.0])
NOISE_STD = 8.0e-4
HORIZON = 4.0
SUBGROUP_DISPERSION_LIMIT = 0.10


def build_block_heterogeneous_observation(
    log_spectral_drift: float,
    noise_correlation: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Generate four channel blocks with oppositely shifted pole locations."""
    if not 0.0 <= noise_correlation < 1.0:
        raise ValueError("noise_correlation must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    times = torch.linspace(0.0, HORIZON, 65, dtype=DTYPE, device=DEVICE)

    block_labels = np.repeat(np.arange(BLOCKS), CHANNELS // BLOCKS)
    block_scores = np.asarray([-1.5, -0.5, 0.5, 1.5])
    score = block_scores[block_labels]
    log_rates = np.log(TRUE_CENTRAL_RATES)[None, :] + log_spectral_drift * np.column_stack(
        [score, -score]
    )
    true_rates = np.exp(log_rates)

    amplitude = np.linspace(0.28, 0.78, CHANNELS)[:, None]
    channel_tilt = np.linspace(0.80, 1.20, CHANNELS)[:, None]
    weights_np = np.concatenate(
        [0.58 * amplitude / channel_tilt, 0.42 * amplitude * channel_tilt], axis=1
    )
    clean = independent_lifted_response(
        times,
        torch.tensor(weights_np, dtype=DTYPE, device=DEVICE),
        torch.tensor(true_rates, dtype=DTYPE, device=DEVICE),
    )

    common = rng.standard_normal((times.numel(), 1))
    independent = rng.standard_normal((times.numel(), CHANNELS))
    noise = math.sqrt(noise_correlation) * common + math.sqrt(1.0 - noise_correlation) * independent
    observations = clean + NOISE_STD * torch.tensor(noise, dtype=DTYPE, device=DEVICE)

    split = 48
    train_np = np.sort(rng.choice(np.arange(1, split), size=34, replace=False))
    train_idx = torch.tensor(train_np, dtype=torch.long, device=DEVICE)
    val_idx = torch.tensor(np.arange(split, times.numel()), dtype=torch.long, device=DEVICE)
    return times, observations, train_idx, val_idx, true_rates, block_labels


def log_spectrum_dispersion(rate_matrix: np.ndarray) -> float:
    """RMS log-distance of subgroup spectra from their geometric centre."""
    values = np.log(np.asarray(rate_matrix, dtype=float))
    centre = values.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean((values - centre) ** 2)))


def evaluate(log_spectral_drift: float, noise_correlation: float, repeat: int) -> dict:
    seed = 40000 + int(1000 * log_spectral_drift) + int(100 * noise_correlation) + repeat
    times, observations, train_idx, val_idx, true_rates, block_labels = (
        build_block_heterogeneous_observation(log_spectral_drift, noise_correlation, seed)
    )

    shared_fits = {}
    for rank in (1, 2):
        candidates = [
            fit_candidate(
                times, observations, train_idx, val_idx, rank, True, seed * 10 + rank * 2 + start
            )
            for start in range(2)
        ]
        shared_fits[rank] = min(candidates, key=lambda item: item.bic)
    shared_rank1, shared_rank2 = shared_fits[1], shared_fits[2]

    subgroup_rates = []
    subgroup_val_rmse = []
    for block in range(BLOCKS):
        mask_np = block_labels == block
        block_observations = observations[:, torch.tensor(mask_np, device=DEVICE)]
        fit = fit_candidate(
            times,
            block_observations,
            train_idx,
            val_idx,
            2,
            True,
            seed * 100 + block,
            adam_steps=280,
            lbfgs_steps=80,
        )
        subgroup_rates.append(fit.rates)
        subgroup_val_rmse.append(fit.val_rmse)

    independent_rank2 = fit_candidate(
        times, observations, train_idx, val_idx, 2, False, seed * 10 + 9
    )
    subgroup_rates_np = np.asarray(subgroup_rates)
    observed_dispersion = log_spectrum_dispersion(subgroup_rates_np)
    true_dispersion = log_spectrum_dispersion(
        np.asarray([true_rates[block_labels == block][0] for block in range(BLOCKS)])
    )
    central_rate_error = float(
        np.linalg.norm(
            np.log(np.asarray(shared_rank2.rates)) - np.log(TRUE_CENTRAL_RATES)
        )
        / math.sqrt(2.0)
    )
    channel_mechanism_distortion = float(
        np.sqrt(
            np.mean(
                (
                    np.log(np.asarray(shared_rank2.rates))[None, :]
                    - np.log(true_rates)
                )
                ** 2
            )
        )
    )
    bic_support = shared_rank1.bic - shared_rank2.bic
    old_gate_pass = bool(
        bic_support >= 6.0 and shared_rank2.val_rmse <= 3.0e-3 and central_rate_error <= 0.35
    )
    refusal = bool(observed_dispersion > SUBGROUP_DISPERSION_LIMIT)
    decision = "REFUSE_SHARED_MECHANISM" if refusal else "ACCEPT_SHARED_MECHANISM"

    return {
        "log_spectral_drift": log_spectral_drift,
        "noise_correlation": noise_correlation,
        "repeat": repeat,
        "seed": seed,
        "decision": decision,
        "old_gate_pass": old_gate_pass,
        "shared_bic_support": bic_support,
        "shared_val_rmse": shared_rank2.val_rmse,
        "independent_val_rmse": independent_rank2.val_rmse,
        "shared_to_independent_val_ratio": shared_rank2.val_rmse / independent_rank2.val_rmse,
        "central_rate_error": central_rate_error,
        "channel_mechanism_distortion": channel_mechanism_distortion,
        "true_subgroup_dispersion": true_dispersion,
        "observed_subgroup_dispersion": observed_dispersion,
        "shared_estimated_rates": shared_rank2.rates,
        "subgroup_estimated_rates": subgroup_rates,
        "median_subgroup_val_rmse": float(np.median(subgroup_val_rmse)),
    }


def summarize(records: list[dict]) -> dict:
    rows = []
    for drift in sorted({record["log_spectral_drift"] for record in records}):
        for correlation in sorted({record["noise_correlation"] for record in records}):
            group = [
                record
                for record in records
                if record["log_spectral_drift"] == drift
                and record["noise_correlation"] == correlation
            ]
            rows.append(
                {
                    "log_spectral_drift": drift,
                    "noise_correlation": correlation,
                    "trials": len(group),
                    "accept_fraction": float(
                        np.mean([r["decision"] == "ACCEPT_SHARED_MECHANISM" for r in group])
                    ),
                    "old_gate_pass_fraction": float(np.mean([r["old_gate_pass"] for r in group])),
                    "median_observed_dispersion": float(
                        np.median([r["observed_subgroup_dispersion"] for r in group])
                    ),
                    "median_true_dispersion": float(
                        np.median([r["true_subgroup_dispersion"] for r in group])
                    ),
                    "median_mechanism_distortion": float(
                        np.median([r["channel_mechanism_distortion"] for r in group])
                    ),
                    "median_validation_ratio": float(
                        np.median([r["shared_to_independent_val_ratio"] for r in group])
                    ),
                }
            )

    by_cell = {(r["log_spectral_drift"], r["noise_correlation"]): r for r in rows}
    mild_ok = all(
        by_cell[(drift, rho)]["accept_fraction"] >= 2.0 / 3.0
        for drift in (0.0, 0.05)
        for rho in NOISE_CORRELATIONS
    )
    severe_refused = all(
        by_cell[(0.15, rho)]["accept_fraction"] <= 1.0 / 3.0
        for rho in NOISE_CORRELATIONS
    )
    old_gate_blind_spot = all(
        by_cell[(0.15, rho)]["old_gate_pass_fraction"] >= 2.0 / 3.0
        for rho in NOISE_CORRELATIONS
    )
    return {
        "rows": rows,
        "route_pass": bool(mild_ok and severe_refused),
        "checks": {
            "mild_sharing_retained": mild_ok,
            "severe_heterogeneity_refused": severe_refused,
            "old_gate_blind_spot_observed": old_gate_blind_spot,
        },
        "frozen_rule": {
            "mild_drift_max": 0.05,
            "mild_accept_fraction_min": 2.0 / 3.0,
            "severe_drift": 0.15,
            "severe_accept_fraction_max": 1.0 / 3.0,
            "subgroup_dispersion_limit": SUBGROUP_DISPERSION_LIMIT,
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "approximate_sharing_refusal_boundary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Approximate-sharing refusal boundary",
        "",
        f"Device: `{payload['device']}`; route pass: **{payload['summary']['route_pass']}**.",
        "",
        "| Log drift | Noise corr. | Accepted | Old gate pass | Observed dispersion | True dispersion | Mechanism distortion | Val. ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["rows"]:
        lines.append(
            f"| {row['log_spectral_drift']:.2f} | {row['noise_correlation']:.2f} | "
            f"{row['accept_fraction']:.2f} | {row['old_gate_pass_fraction']:.2f} | "
            f"{row['median_observed_dispersion']:.3g} | {row['median_true_dispersion']:.3g} | "
            f"{row['median_mechanism_distortion']:.3g} | {row['median_validation_ratio']:.3g} |"
        )
    lines.extend(
        [
            "",
            "The refusal rule uses only subgroup estimates and is frozen before the run.",
            "True dispersion and mechanism distortion are diagnostic evaluation metrics.",
        ]
    )
    (RESULTS / "approximate_sharing_refusal_boundary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    records = []
    for drift in LOG_SPECTRAL_DRIFTS:
        for correlation in NOISE_CORRELATIONS:
            for repeat in range(REPEATS):
                record = evaluate(drift, correlation, repeat)
                records.append(record)
                print(
                    f"drift={drift:.2f} rho={correlation:.2f} repeat={repeat} "
                    f"decision={record['decision']} "
                    f"dispersion={record['observed_subgroup_dispersion']:.3g} "
                    f"old_gate={record['old_gate_pass']}",
                    flush=True,
                )
    summary = summarize(records)
    payload = {
        "experiment": "approximate_sharing_refusal_boundary",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "protocol": {
            "channels": CHANNELS,
            "blocks": BLOCKS,
            "log_spectral_drifts": list(LOG_SPECTRAL_DRIFTS),
            "noise_correlations": list(NOISE_CORRELATIONS),
            "noise_std": NOISE_STD,
            "horizon": HORIZON,
            "repeats": REPEATS,
            "subgroup_dispersion_limit": SUBGROUP_DISPERSION_LIMIT,
        },
        "records": records,
        "summary": summary,
    }
    write_outputs(payload)
    print(json.dumps({"route_pass": summary["route_pass"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
