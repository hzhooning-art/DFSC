"""Falsify the frozen decomposed tolerance gate under controlled transfer."""

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
from probe_nested_group_sharing_gate import fit_grouped_candidate
from probe_noise_aware_sharing_gate import (
    BIC_EVIDENCE_LIMIT,
    RELATIVE_DEGRADATION_LIMIT,
    classify_with_limit,
    second_difference_correlation_proxy,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHANNEL_COUNTS = (32, 128)
NOISE_STDS = (4.0e-4, 1.6e-3)
HETEROGENEITY_CONSTRUCTIONS = ("antisymmetric", "curved")
LOG_SPECTRAL_DRIFTS = (0.0, 0.05, 0.15)
NOISE_CORRELATION = 0.30
REPEATS = 3
BLOCKS = 4
HORIZON = 4.0
TRUE_CENTRAL_RATES = np.asarray([0.25, 1.0])

# Frozen verbatim from Stage 45. The transfer experiment never refits them.
NOISE_ENVELOPE = {
    "intercept": 0.0012527224790298695,
    "slope": 0.00012626852978749425,
    "one_sided_residual": 0.0009613423183086616,
}
MODEL_ALLOWANCE = 0.0013357341563319772
PROXY_SCOPE = (0.08970932263544007, 0.9154926050327769)


def block_score_matrix(construction: str) -> np.ndarray:
    base = np.asarray([-1.5, -0.5, 0.5, 1.5])
    if construction == "antisymmetric":
        scores = np.column_stack([base, -base])
    elif construction == "curved":
        curved = base**2 - np.mean(base**2)
        scores = np.column_stack([base, curved])
        target_rms = float(np.sqrt(np.mean(np.column_stack([base, -base]) ** 2)))
        scores *= target_rms / float(np.sqrt(np.mean(scores**2)))
    else:
        raise ValueError(f"unknown heterogeneity construction: {construction}")
    return scores


def build_transfer_observation(
    channels: int,
    noise_std: float,
    construction: str,
    log_spectral_drift: float,
    seed: int,
):
    if channels % BLOCKS:
        raise ValueError("channels must be divisible by four")
    rng = np.random.default_rng(seed)
    times = torch.linspace(0.0, HORIZON, 65, dtype=DTYPE, device=DEVICE)
    block_labels = np.repeat(np.arange(BLOCKS), channels // BLOCKS)
    scores = block_score_matrix(construction)[block_labels]
    true_rates = np.exp(
        np.log(TRUE_CENTRAL_RATES)[None, :] + log_spectral_drift * scores
    )

    amplitude = np.linspace(0.28, 0.78, channels)[:, None]
    channel_tilt = np.linspace(0.80, 1.20, channels)[:, None]
    weights = np.concatenate(
        [0.58 * amplitude / channel_tilt, 0.42 * amplitude * channel_tilt], axis=1
    )
    clean = independent_lifted_response(
        times,
        torch.tensor(weights, dtype=DTYPE, device=DEVICE),
        torch.tensor(true_rates, dtype=DTYPE, device=DEVICE),
    )
    common = rng.standard_normal((times.numel(), 1))
    independent = rng.standard_normal((times.numel(), channels))
    noise = (
        math.sqrt(NOISE_CORRELATION) * common
        + math.sqrt(1.0 - NOISE_CORRELATION) * independent
    )
    observations = clean + noise_std * torch.tensor(noise, dtype=DTYPE, device=DEVICE)
    split = 48
    train = np.sort(rng.choice(np.arange(1, split), size=34, replace=False))
    train_idx = torch.tensor(train, dtype=torch.long, device=DEVICE)
    val_idx = torch.tensor(np.arange(split, times.numel()), dtype=torch.long, device=DEVICE)
    return times, observations, train_idx, val_idx, true_rates, block_labels


def frozen_total_tolerance(proxy: float) -> float:
    noise = max(
        1.0e-6,
        NOISE_ENVELOPE["intercept"]
        + NOISE_ENVELOPE["slope"] * proxy
        + NOISE_ENVELOPE["one_sided_residual"],
    )
    return noise + MODEL_ALLOWANCE


def best_shared_fit(times, observations, train_idx, val_idx, seed: int):
    return min(
        [
            fit_candidate(
                times, observations, train_idx, val_idx, 2, True, seed * 10 + start
            )
            for start in range(2)
        ],
        key=lambda item: item.bic,
    )


def evaluate(
    channels: int,
    noise_std: float,
    construction: str,
    drift: float,
    repeat: int,
) -> dict:
    construction_code = HETEROGENEITY_CONSTRUCTIONS.index(construction)
    seed = (
        61000
        + channels * 10
        + int(noise_std * 1.0e6)
        + construction_code * 1000
        + int(drift * 1000)
        + repeat
    )
    times, observations, train_idx, val_idx, _, block_labels = build_transfer_observation(
        channels, noise_std, construction, drift, seed
    )
    shared = best_shared_fit(times, observations, train_idx, val_idx, seed)
    labels = torch.tensor(block_labels, dtype=torch.long, device=DEVICE)
    grouped = min(
        [
            fit_grouped_candidate(
                times, observations, train_idx, val_idx, labels, seed * 10 + 5 + start
            )
            for start in range(2)
        ],
        key=lambda item: item.bic,
    )
    support = shared.bic - grouped.bic
    proxy = second_difference_correlation_proxy(observations)
    tolerance = frozen_total_tolerance(proxy)
    decision = classify_with_limit(
        support, shared.val_rmse, grouped.val_rmse, tolerance
    )
    return {
        "channels": channels,
        "noise_std": noise_std,
        "heterogeneity_construction": construction,
        "log_spectral_drift": drift,
        "noise_correlation_diagnostic": NOISE_CORRELATION,
        "repeat": repeat,
        "seed": seed,
        "correlation_proxy": proxy,
        "proxy_in_frozen_scope": PROXY_SCOPE[0] <= proxy <= PROXY_SCOPE[1],
        "frozen_total_tolerance": tolerance,
        "decision": decision,
        "group_bic_support": support,
        "shared_val_rmse": shared.val_rmse,
        "grouped_val_rmse": grouped.val_rmse,
        "shared_to_grouped_val_ratio": shared.val_rmse / max(grouped.val_rmse, 1.0e-15),
    }


def summarize(records: list[dict]) -> dict:
    rows = []
    for channels in CHANNEL_COUNTS:
        for noise_std in NOISE_STDS:
            for construction in HETEROGENEITY_CONSTRUCTIONS:
                for drift in LOG_SPECTRAL_DRIFTS:
                    group = [
                        r
                        for r in records
                        if r["channels"] == channels
                        and r["noise_std"] == noise_std
                        and r["heterogeneity_construction"] == construction
                        and r["log_spectral_drift"] == drift
                    ]
                    rows.append(
                        {
                            "channels": channels,
                            "noise_std": noise_std,
                            "heterogeneity_construction": construction,
                            "log_spectral_drift": drift,
                            "trials": len(group),
                            "refuse_fraction": float(
                                np.mean(
                                    [
                                        r["decision"] == "REFUSE_SHARED_MECHANISM"
                                        for r in group
                                    ]
                                )
                            ),
                            "median_proxy": float(
                                np.median([r["correlation_proxy"] for r in group])
                            ),
                            "median_shared_val_rmse": float(
                                np.median([r["shared_val_rmse"] for r in group])
                            ),
                            "median_group_bic_support": float(
                                np.median([r["group_bic_support"] for r in group])
                            ),
                            "in_scope_fraction": float(
                                np.mean([r["proxy_in_frozen_scope"] for r in group])
                            ),
                        }
                    )
    mild_retained = all(
        row["refuse_fraction"] <= 1.0 / 3.0
        for row in rows
        if row["log_spectral_drift"] in (0.0, 0.05)
    )
    severe_refused = all(
        row["refuse_fraction"] >= 2.0 / 3.0
        for row in rows
        if row["log_spectral_drift"] == 0.15
    )
    checks = {
        "complete_transfer_matrix": len(records)
        == len(CHANNEL_COUNTS)
        * len(NOISE_STDS)
        * len(HETEROGENEITY_CONSTRUCTIONS)
        * len(LOG_SPECTRAL_DRIFTS)
        * REPEATS,
        "exact_and_mild_retained_in_every_transfer_cell": mild_retained,
        "severe_refused_in_every_transfer_cell": severe_refused,
        "all_proxies_in_frozen_scope": all(
            record["proxy_in_frozen_scope"] for record in records
        ),
    }
    return {
        "rows": rows,
        "checks": checks,
        "route_pass": bool(all(checks.values())),
        "frozen_rule": {
            "mild_refuse_fraction_max": 1.0 / 3.0,
            "severe_refuse_fraction_min": 2.0 / 3.0,
            "out_of_proxy_scope_action": "route failure",
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "decomposed_tolerance_transfer.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Frozen decomposed-tolerance transfer audit",
        "",
        f"Device: `{payload['device']}`; route pass: **{payload['summary']['route_pass']}**.",
        "",
        "| Channels | Noise std | Construction | Drift | Refused | Proxy | Shared RMSE | Group BIC support | In scope |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["rows"]:
        lines.append(
            f"| {row['channels']} | {row['noise_std']:.1e} | {row['heterogeneity_construction']} | "
            f"{row['log_spectral_drift']:.2f} | {row['refuse_fraction']:.2f} | "
            f"{row['median_proxy']:.3f} | {row['median_shared_val_rmse']:.4g} | "
            f"{row['median_group_bic_support']:.3g} | {row['in_scope_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "All tolerance coefficients and the proxy scope are copied from Stage 45; no transfer record is used for recalibration.",
        ]
    )
    (RESULTS / "decomposed_tolerance_transfer.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    records = []
    for channels in CHANNEL_COUNTS:
        for noise_std in NOISE_STDS:
            for construction in HETEROGENEITY_CONSTRUCTIONS:
                for drift in LOG_SPECTRAL_DRIFTS:
                    for repeat in range(REPEATS):
                        record = evaluate(
                            channels, noise_std, construction, drift, repeat
                        )
                        records.append(record)
                        print(
                            f"channels={channels} noise={noise_std:.1e} construction={construction} "
                            f"drift={drift:.2f} repeat={repeat} decision={record['decision']} "
                            f"proxy={record['correlation_proxy']:.3f} rmse={record['shared_val_rmse']:.4g}",
                            flush=True,
                        )
    summary = summarize(records)
    payload = {
        "experiment": "decomposed_tolerance_transfer",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "protocol": {
            "channel_counts": list(CHANNEL_COUNTS),
            "noise_stds": list(NOISE_STDS),
            "heterogeneity_constructions": list(HETEROGENEITY_CONSTRUCTIONS),
            "log_spectral_drifts": list(LOG_SPECTRAL_DRIFTS),
            "noise_correlation": NOISE_CORRELATION,
            "repeats": REPEATS,
        },
        "frozen_stage45_budget": {
            "noise_envelope": NOISE_ENVELOPE,
            "model_allowance": MODEL_ALLOWANCE,
            "proxy_scope": list(PROXY_SCOPE),
        },
        "records": records,
        "summary": summary,
    }
    write_outputs(payload)
    print(json.dumps({"route_pass": summary["route_pass"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
