"""Nested global-versus-grouped spectrum gate under correlated noise."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from probe_approximate_sharing_refusal_boundary import (
    BLOCKS,
    LOG_SPECTRAL_DRIFTS,
    NOISE_CORRELATIONS,
    REPEATS,
    build_block_heterogeneous_observation,
)
from probe_high_dimensional_shared_spectrum import DTYPE, DEVICE, fit_candidate, independent_lifted_response


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BIC_EVIDENCE_LIMIT = 6.0
VALIDATION_RMSE_LIMIT = 3.0e-3
RELATIVE_DEGRADATION_LIMIT = 1.50


@dataclass
class GroupedFit:
    bic: float
    val_rmse: float
    rates: list[list[float]]


def grouped_response(
    times: torch.Tensor,
    weights: torch.Tensor,
    group_rates: torch.Tensor,
    group_labels: torch.Tensor,
) -> torch.Tensor:
    """Evaluate channel responses using one spectrum per declared group."""
    return independent_lifted_response(times, weights, group_rates[group_labels])


def fit_grouped_candidate(
    times: torch.Tensor,
    observations: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    group_labels: torch.Tensor,
    seed: int,
    adam_steps: int = 280,
    lbfgs_steps: int = 80,
) -> GroupedFit:
    torch.manual_seed(seed)
    channels = observations.shape[1]
    rank = 2
    base = torch.linspace(math.log(0.18), math.log(1.4), rank, dtype=DTYPE, device=DEVICE)
    raw_rates = torch.nn.Parameter(base.expand(BLOCKS, rank).clone())
    raw_rates.data.add_(0.12 * torch.randn_like(raw_rates))
    raw_weights = torch.nn.Parameter(
        math.log(0.40) + 0.18 * torch.randn((channels, rank), dtype=DTYPE, device=DEVICE)
    )

    def unpack() -> tuple[torch.Tensor, torch.Tensor]:
        rates = torch.sort(torch.exp(raw_rates).clamp(1.0e-3, 20.0), dim=-1).values
        weights = torch.exp(raw_weights).clamp(1.0e-6, 20.0)
        return rates, weights

    optimizer = torch.optim.Adam([raw_rates, raw_weights], lr=0.035)
    best_loss = float("inf")
    best_state = None
    for _ in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        rates, weights = unpack()
        prediction = grouped_response(times, weights, rates, group_labels)
        loss = (prediction[train_idx] - observations[train_idx]).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_rates, raw_weights], 10.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = (raw_rates.detach().clone(), raw_weights.detach().clone())
    if best_state is None:
        raise RuntimeError("grouped optimizer did not produce a finite state")
    raw_rates.data.copy_(best_state[0])
    raw_weights.data.copy_(best_state[1])

    refiner = torch.optim.LBFGS(
        [raw_rates, raw_weights],
        lr=0.8,
        max_iter=lbfgs_steps,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        refiner.zero_grad(set_to_none=True)
        rates, weights = unpack()
        prediction = grouped_response(times, weights, rates, group_labels)
        loss = (prediction[train_idx] - observations[train_idx]).square().mean()
        loss.backward()
        return loss

    refiner.step(closure)
    rates, weights = unpack()
    prediction = grouped_response(times, weights, rates, group_labels)
    train_residual = prediction[train_idx] - observations[train_idx]
    val_residual = prediction[val_idx] - observations[val_idx]
    n = train_residual.numel()
    parameters = BLOCKS * rank + channels * rank
    rss = train_residual.square().sum().clamp_min(1.0e-30)
    bic = n * torch.log(rss / n) + parameters * math.log(n)
    return GroupedFit(
        bic=float(bic.detach().cpu()),
        val_rmse=float(torch.sqrt(val_residual.square().mean()).detach().cpu()),
        rates=rates.detach().cpu().tolist(),
    )


def classify(group_support: float, shared_val_rmse: float, grouped_val_rmse: float) -> str:
    degradation = shared_val_rmse / max(grouped_val_rmse, 1.0e-15)
    heterogeneity_detected = group_support >= BIC_EVIDENCE_LIMIT
    materially_inadequate = (
        shared_val_rmse > VALIDATION_RMSE_LIMIT
        or (heterogeneity_detected and degradation > RELATIVE_DEGRADATION_LIMIT)
    )
    if materially_inadequate:
        return "REFUSE_SHARED_MECHANISM"
    if heterogeneity_detected:
        return "ACCEPT_WITH_SCOPE_LIMITS"
    return "ACCEPT_SHARED_MECHANISM"


def evaluate(log_spectral_drift: float, noise_correlation: float, repeat: int) -> dict:
    seed = 41000 + int(1000 * log_spectral_drift) + int(100 * noise_correlation) + repeat
    times, observations, train_idx, val_idx, _, block_labels_np = (
        build_block_heterogeneous_observation(log_spectral_drift, noise_correlation, seed)
    )
    shared_candidates = [
        fit_candidate(times, observations, train_idx, val_idx, 2, True, seed * 10 + start)
        for start in range(2)
    ]
    shared = min(shared_candidates, key=lambda item: item.bic)
    labels = torch.tensor(block_labels_np, dtype=torch.long, device=DEVICE)
    grouped_candidates = [
        fit_grouped_candidate(times, observations, train_idx, val_idx, labels, seed * 10 + 5 + start)
        for start in range(2)
    ]
    grouped = min(grouped_candidates, key=lambda item: item.bic)
    support = shared.bic - grouped.bic
    degradation = shared.val_rmse / max(grouped.val_rmse, 1.0e-15)
    decision = classify(support, shared.val_rmse, grouped.val_rmse)
    return {
        "log_spectral_drift": log_spectral_drift,
        "noise_correlation": noise_correlation,
        "repeat": repeat,
        "seed": seed,
        "decision": decision,
        "group_bic_support": support,
        "shared_val_rmse": shared.val_rmse,
        "grouped_val_rmse": grouped.val_rmse,
        "shared_to_grouped_val_ratio": degradation,
        "shared_rates": shared.rates,
        "grouped_rates": grouped.rates,
    }


def summarize(records: list[dict]) -> dict:
    rows = []
    for drift in sorted({r["log_spectral_drift"] for r in records}):
        for rho in sorted({r["noise_correlation"] for r in records}):
            group = [r for r in records if r["log_spectral_drift"] == drift and r["noise_correlation"] == rho]
            rows.append(
                {
                    "log_spectral_drift": drift,
                    "noise_correlation": rho,
                    "trials": len(group),
                    "refuse_fraction": float(np.mean([r["decision"] == "REFUSE_SHARED_MECHANISM" for r in group])),
                    "scope_limited_fraction": float(np.mean([r["decision"] == "ACCEPT_WITH_SCOPE_LIMITS" for r in group])),
                    "median_group_bic_support": float(np.median([r["group_bic_support"] for r in group])),
                    "median_shared_val_rmse": float(np.median([r["shared_val_rmse"] for r in group])),
                    "median_validation_ratio": float(np.median([r["shared_to_grouped_val_ratio"] for r in group])),
                }
            )
    cells = {(r["log_spectral_drift"], r["noise_correlation"]): r for r in rows}
    mild_retained = all(
        cells[(d, rho)]["refuse_fraction"] <= 1.0 / 3.0
        for d in (0.0, 0.05)
        for rho in NOISE_CORRELATIONS
    )
    severe_refused = all(
        cells[(0.15, rho)]["refuse_fraction"] >= 2.0 / 3.0 for rho in NOISE_CORRELATIONS
    )
    return {
        "rows": rows,
        "route_pass": bool(mild_retained and severe_refused),
        "checks": {"mild_sharing_retained": mild_retained, "severe_heterogeneity_refused": severe_refused},
        "frozen_rule": {
            "bic_evidence_limit": BIC_EVIDENCE_LIMIT,
            "validation_rmse_limit": VALIDATION_RMSE_LIMIT,
            "relative_degradation_limit": RELATIVE_DEGRADATION_LIMIT,
            "mild_refuse_fraction_max": 1.0 / 3.0,
            "severe_refuse_fraction_min": 2.0 / 3.0,
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "nested_group_sharing_gate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Nested grouped-sharing gate",
        "",
        f"Device: `{payload['device']}`; route pass: **{payload['summary']['route_pass']}**.",
        "",
        "| Log drift | Noise corr. | Refused | Scope-limited | Group BIC support | Shared RMSE | Val. ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["rows"]:
        lines.append(
            f"| {row['log_spectral_drift']:.2f} | {row['noise_correlation']:.2f} | "
            f"{row['refuse_fraction']:.2f} | {row['scope_limited_fraction']:.2f} | "
            f"{row['median_group_bic_support']:.3g} | {row['median_shared_val_rmse']:.3g} | "
            f"{row['median_validation_ratio']:.3g} |"
        )
    lines.extend(["", "BIC detects heterogeneity; refusal additionally requires material held-out degradation."])
    (RESULTS / "nested_group_sharing_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = []
    for drift in LOG_SPECTRAL_DRIFTS:
        for rho in NOISE_CORRELATIONS:
            for repeat in range(REPEATS):
                record = evaluate(drift, rho, repeat)
                records.append(record)
                print(
                    f"drift={drift:.2f} rho={rho:.2f} repeat={repeat} "
                    f"decision={record['decision']} support={record['group_bic_support']:.3g} "
                    f"ratio={record['shared_to_grouped_val_ratio']:.3g}", flush=True
                )
    summary = summarize(records)
    payload = {
        "experiment": "nested_group_sharing_gate",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "protocol": {
            "log_spectral_drifts": list(LOG_SPECTRAL_DRIFTS),
            "noise_correlations": list(NOISE_CORRELATIONS),
            "repeats": REPEATS,
            "blocks": BLOCKS,
        },
        "records": records,
        "summary": summary,
    }
    write_outputs(payload)
    print(json.dumps({"route_pass": summary["route_pass"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
