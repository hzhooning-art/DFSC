"""High-dimensional scaling audit for shared latent memory spectra.

The experiment compares a rank-two model with pole locations shared across all
channels against a vectorized control whose pole locations are fitted separately
for every channel.  It is a controlled synthetic feasibility test, not evidence
that a scientific field necessarily has a shared memory spectrum.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from probe_identifiability_boundary import build_observation
from probe_memory_rank import DEVICE, DTYPE, lifted_response


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHANNEL_COUNTS = (1, 16, 64, 256)
REPEATS = 3
TRUE_RATES = np.asarray([0.25, 1.0])
NOISE_STD = 8.0e-4
HORIZON = 4.0
RATE_RATIO = 4.0


@dataclass
class CandidateFit:
    rank: int
    shared: bool
    train_rmse: float
    val_rmse: float
    bic: float
    rates: list
    weights: list
    channel_bic: list[float]
    channel_val_rmse: list[float]
    elapsed_seconds: float
    peak_memory_bytes: int


def independent_lifted_response(
    times: torch.Tensor,
    weights: torch.Tensor,
    rates: torch.Tensor,
) -> torch.Tensor:
    """Evaluate channels with channel-specific positive pole locations."""
    channels, rank = weights.shape
    if rates.shape != (channels, rank):
        raise ValueError("rates must have shape (channels, rank)")
    matrices = torch.zeros((channels, rank + 1, rank + 1), dtype=DTYPE, device=DEVICE)
    matrices[:, 0, 1:] = -weights
    matrices[:, 1:, 0] = 1.0
    diagonal = torch.arange(rank, device=DEVICE)
    matrices[:, diagonal + 1, diagonal + 1] = -rates
    propagators = torch.matrix_exp(times[:, None, None, None] * matrices[None, :, :, :])
    initial = torch.zeros((channels, rank + 1), dtype=DTYPE, device=DEVICE)
    initial[:, 0] = 1.0
    return torch.einsum("tcij,cj->tci", propagators, initial)[:, :, 0]


def _positive_parameters(
    raw_rates: torch.Tensor,
    raw_weights: torch.Tensor,
    shared: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rates = torch.exp(raw_rates).clamp(1.0e-3, 20.0)
    rates = torch.sort(rates, dim=-1).values
    if shared:
        rates = rates.reshape(-1)
    weights = torch.exp(raw_weights).clamp(1.0e-6, 20.0)
    return rates, weights


def _predict(
    times: torch.Tensor,
    raw_rates: torch.Tensor,
    raw_weights: torch.Tensor,
    shared: bool,
) -> torch.Tensor:
    rates, weights = _positive_parameters(raw_rates, raw_weights, shared)
    if shared:
        return lifted_response(times, weights, rates)
    return independent_lifted_response(times, weights, rates)


def fit_candidate(
    times: torch.Tensor,
    observations: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    rank: int,
    shared: bool,
    seed: int,
    adam_steps: int = 280,
    lbfgs_steps: int = 80,
) -> CandidateFit:
    """Fit one shared or channel-independent candidate without a full Jacobian."""
    torch.manual_seed(seed)
    channels = observations.shape[1]
    rate_shape = (rank,) if shared else (channels, rank)
    base_rates = torch.linspace(math.log(0.18), math.log(1.4), rank, dtype=DTYPE, device=DEVICE)
    raw_rates = torch.nn.Parameter(base_rates.expand(rate_shape).clone())
    raw_rates.data.add_(0.12 * torch.randn_like(raw_rates))
    raw_weights = torch.nn.Parameter(
        math.log(0.40) + 0.18 * torch.randn((channels, rank), dtype=DTYPE, device=DEVICE)
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    optimizer = torch.optim.Adam([raw_rates, raw_weights], lr=0.035)
    best_loss = float("inf")
    best_state = None
    for _ in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(times, raw_rates, raw_weights, shared)
        loss = (prediction[train_idx] - observations[train_idx]).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_rates, raw_weights], 10.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = (raw_rates.detach().clone(), raw_weights.detach().clone())

    if best_state is None:
        raise RuntimeError("optimizer did not produce a finite state")
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
        prediction = _predict(times, raw_rates, raw_weights, shared)
        loss = (prediction[train_idx] - observations[train_idx]).square().mean()
        loss.backward()
        return loss

    refiner.step(closure)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    prediction = _predict(times, raw_rates, raw_weights, shared)
    train_residual = prediction[train_idx] - observations[train_idx]
    val_residual = prediction[val_idx] - observations[val_idx]
    n = train_residual.numel()
    parameters = rank + channels * rank if shared else 2 * channels * rank
    rss = train_residual.square().sum().clamp_min(1.0e-30)
    bic = n * torch.log(rss / n) + parameters * math.log(n)
    channel_n = train_residual.shape[0]
    channel_rss = train_residual.square().sum(dim=0).clamp_min(1.0e-30)
    channel_parameters = rank + rank
    channel_bic = channel_n * torch.log(channel_rss / channel_n) + channel_parameters * math.log(
        channel_n
    )
    channel_val_rmse = torch.sqrt(val_residual.square().mean(dim=0))
    rates, weights = _positive_parameters(raw_rates, raw_weights, shared)
    peak_memory = torch.cuda.max_memory_allocated() if DEVICE.type == "cuda" else 0
    return CandidateFit(
        rank=rank,
        shared=shared,
        train_rmse=float(torch.sqrt(train_residual.square().mean()).detach().cpu()),
        val_rmse=float(torch.sqrt(val_residual.square().mean()).detach().cpu()),
        bic=float(bic.detach().cpu()),
        rates=rates.detach().cpu().tolist(),
        weights=weights.detach().cpu().tolist(),
        channel_bic=channel_bic.detach().cpu().tolist(),
        channel_val_rmse=channel_val_rmse.detach().cpu().tolist(),
        elapsed_seconds=elapsed,
        peak_memory_bytes=int(peak_memory),
    )


def _best_fit(
    times: torch.Tensor,
    observations: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    rank: int,
    shared: bool,
    seed: int,
) -> CandidateFit:
    candidates = [
        fit_candidate(times, observations, train_idx, val_idx, rank, shared, seed * 10 + start)
        for start in range(2)
    ]
    return min(candidates, key=lambda item: item.bic)


def evaluate(channels: int, repeat: int) -> dict:
    seed = 39000 + channels * 10 + repeat
    times, observations, train_idx, val_idx, _ = build_observation(
        RATE_RATIO, HORIZON, channels, NOISE_STD, seed
    )
    shared_rank1 = _best_fit(times, observations, train_idx, val_idx, 1, True, seed + 1)
    shared_rank2 = _best_fit(times, observations, train_idx, val_idx, 2, True, seed + 2)
    shared_support = shared_rank1.bic - shared_rank2.bic
    shared_rates = np.asarray(shared_rank2.rates)
    shared_rate_error = float(
        np.linalg.norm(np.log(shared_rates) - np.log(TRUE_RATES)) / math.sqrt(2.0)
    )
    shared_resolved = bool(
        shared_support >= 6.0
        and shared_rank2.val_rmse <= 3.0e-3
        and shared_rate_error <= 0.35
    )

    independent_rank1 = _best_fit(times, observations, train_idx, val_idx, 1, False, seed + 3)
    independent_rank2 = _best_fit(times, observations, train_idx, val_idx, 2, False, seed + 4)
    rate2 = np.asarray(independent_rank2.rates)
    channel_rate_error = np.linalg.norm(
        np.log(rate2) - np.log(TRUE_RATES)[None, :], axis=1
    ) / math.sqrt(2.0)

    # The vectorized independent fit has one aggregate BIC.  Per-channel recovery
    # is evaluated from held-out accuracy and pole error; aggregate BIC records
    # whether the extra independent poles are supported as a model class.
    independent_support = independent_rank1.bic - independent_rank2.bic
    independent_channel_support = np.asarray(independent_rank1.channel_bic) - np.asarray(
        independent_rank2.channel_bic
    )
    independent_channel_val_rmse = np.asarray(independent_rank2.channel_val_rmse)
    independent_channel_resolved = (
        (channel_rate_error <= 0.35)
        & (independent_channel_val_rmse <= 3.0e-3)
        & (independent_channel_support >= 6.0)
    )

    return {
        "channels": channels,
        "repeat": repeat,
        "seed": seed,
        "shared_resolved": shared_resolved,
        "shared_bic_support": shared_support,
        "shared_rate_error": shared_rate_error,
        "shared_val_rmse": shared_rank2.val_rmse,
        "shared_estimated_rates": shared_rank2.rates,
        "shared_elapsed_seconds": shared_rank1.elapsed_seconds + shared_rank2.elapsed_seconds,
        "independent_bic_support": independent_support,
        "independent_resolved_fraction": float(np.mean(independent_channel_resolved)),
        "independent_median_rate_error": float(np.median(channel_rate_error)),
        "independent_median_channel_bic_support": float(np.median(independent_channel_support)),
        "independent_median_channel_val_rmse": float(np.median(independent_channel_val_rmse)),
        "independent_val_rmse": independent_rank2.val_rmse,
        "independent_elapsed_seconds": independent_rank1.elapsed_seconds + independent_rank2.elapsed_seconds,
        "peak_memory_bytes": max(
            shared_rank1.peak_memory_bytes,
            shared_rank2.peak_memory_bytes,
            independent_rank1.peak_memory_bytes,
            independent_rank2.peak_memory_bytes,
        ),
    }


def summarize(records: list[dict]) -> dict:
    rows = []
    for channels in sorted({record["channels"] for record in records}):
        group = [record for record in records if record["channels"] == channels]
        rows.append(
            {
                "channels": channels,
                "trials": len(group),
                "shared_resolved_fraction": float(np.mean([r["shared_resolved"] for r in group])),
                "median_shared_rate_error": float(np.median([r["shared_rate_error"] for r in group])),
                "median_shared_validation_rmse": float(np.median([r["shared_val_rmse"] for r in group])),
                "median_shared_bic_support": float(np.median([r["shared_bic_support"] for r in group])),
                "median_independent_resolved_fraction": float(
                    np.median([r["independent_resolved_fraction"] for r in group])
                ),
                "median_independent_rate_error": float(
                    np.median([r["independent_median_rate_error"] for r in group])
                ),
                "median_shared_seconds": float(np.median([r["shared_elapsed_seconds"] for r in group])),
                "median_independent_seconds": float(
                    np.median([r["independent_elapsed_seconds"] for r in group])
                ),
                "median_peak_memory_bytes": int(np.median([r["peak_memory_bytes"] for r in group])),
            }
        )

    by_channels = {row["channels"]: row for row in rows}
    required = (16, 64, 256)
    resolution_ok = all(by_channels[c]["shared_resolved_fraction"] >= 2.0 / 3.0 for c in required)
    terminal_accuracy_ok = by_channels[256]["median_shared_rate_error"] <= 0.10
    advantage_ok = all(
        by_channels[c]["shared_resolved_fraction"]
        - by_channels[c]["median_independent_resolved_fraction"] >= 0.25
        for c in (64, 256)
    )
    route_pass = bool(resolution_ok and terminal_accuracy_ok and advantage_ok)
    passing = [
        c for c in required
        if by_channels[c]["shared_resolved_fraction"] >= 2.0 / 3.0
        and by_channels[c]["median_shared_rate_error"] <= 0.10
    ]
    return {
        "rows": rows,
        "route_pass": route_pass,
        "minimum_passing_channels": min(passing) if passing else None,
        "checks": {
            "shared_resolution": resolution_ok,
            "terminal_rate_accuracy": terminal_accuracy_ok,
            "shared_over_independent_advantage": advantage_ok,
        },
        "frozen_rule": {
            "required_channel_counts": list(required),
            "shared_resolution_min": 2.0 / 3.0,
            "channel_256_median_rate_error_max": 0.10,
            "advantage_min_at_64_and_256": 0.25,
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "high_dimensional_shared_spectrum.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# High-dimensional shared-spectrum audit",
        "",
        f"Device: `{payload['device']}`; route pass: **{payload['summary']['route_pass']}**.",
        "",
        "| Channels | Shared resolved | Shared rate error | Independent resolved | Shared time (s) | Independent time (s) | Peak memory (MiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["rows"]:
        lines.append(
            f"| {row['channels']} | {row['shared_resolved_fraction']:.2f} | "
            f"{row['median_shared_rate_error']:.3g} | "
            f"{row['median_independent_resolved_fraction']:.2f} | "
            f"{row['median_shared_seconds']:.3g} | {row['median_independent_seconds']:.3g} | "
            f"{row['median_peak_memory_bytes'] / 2**20:.2f} |"
        )
    lines.extend(
        [
            "",
            "The experiment fixes the generator, horizon, noise level, optimizer budget,",
            "and decision rule before inspecting outcomes.  It is a synthetic scaling",
            "audit; the true-rate error is an evaluation metric unavailable in deployment.",
        ]
    )
    (RESULTS / "high_dimensional_shared_spectrum.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    records = []
    for channels in CHANNEL_COUNTS:
        for repeat in range(REPEATS):
            record = evaluate(channels, repeat)
            records.append(record)
            print(
                f"channels={channels:3d} repeat={repeat} "
                f"shared={record['shared_resolved']} "
                f"shared_error={record['shared_rate_error']:.3g} "
                f"independent_fraction={record['independent_resolved_fraction']:.3g}",
                flush=True,
            )
    summary = summarize(records)
    payload = {
        "experiment": "high_dimensional_shared_memory_spectrum",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "protocol": {
            "channels": list(CHANNEL_COUNTS),
            "repeats": REPEATS,
            "horizon": HORIZON,
            "noise_std": NOISE_STD,
            "rate_ratio": RATE_RATIO,
            "optimizer_starts": 2,
            "full_parameter_jacobian": False,
        },
        "records": records,
        "summary": summary,
    }
    write_outputs(payload)
    print(json.dumps({"route_pass": summary["route_pass"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
