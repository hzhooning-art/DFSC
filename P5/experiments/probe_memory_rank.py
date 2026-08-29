"""Feasibility probe for identifiable latent memory-rank discovery.

The experiment fits positive pole-residue memory realizations to scalar and
multi-channel relaxation trajectories. It is intentionally small: its purpose is to
test whether the proposed P5 question survives basic falsification, not to provide
publication evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class FitResult:
    rank: int
    seed: int
    train_rmse: float
    val_rmse: float
    bic: float
    jacobian_condition: float
    rates: list[float]
    weights: list[list[float]]


def lifted_response(times: torch.Tensor, weights: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
    """Return x(t) for independent channels sharing positive pole locations."""
    channels, rank = weights.shape
    matrices = torch.zeros((channels, rank + 1, rank + 1), dtype=DTYPE, device=DEVICE)
    matrices[:, 0, 1:] = -weights
    matrices[:, 1:, 0] = 1.0
    diagonal = torch.arange(rank, device=DEVICE)
    matrices[:, diagonal + 1, diagonal + 1] = -rates[None, :]

    scaled = times[:, None, None, None] * matrices[None, :, :, :]
    propagators = torch.matrix_exp(scaled)
    initial = torch.zeros((channels, rank + 1), dtype=DTYPE, device=DEVICE)
    initial[:, 0] = 1.0
    states = torch.einsum("tcij,cj->tci", propagators, initial)
    return states[:, :, 0]


def unpack(raw_rates: torch.Tensor, raw_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rates = torch.sort(torch.exp(raw_rates).clamp(1.0e-3, 20.0)).values
    weights = torch.exp(raw_weights).clamp(1.0e-6, 20.0)
    return rates, weights


def fit_rank(
    times: torch.Tensor,
    observations: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    rank: int,
    seed: int,
    adam_steps: int = 320,
    lbfgs_steps: int = 90,
) -> FitResult:
    torch.manual_seed(seed)
    channels = observations.shape[1]
    raw_rates = torch.nn.Parameter(torch.linspace(math.log(0.12), math.log(1.8), rank, device=DEVICE, dtype=DTYPE))
    raw_rates.data.add_(0.20 * torch.randn_like(raw_rates))
    raw_weights = torch.nn.Parameter(
        math.log(0.45) + 0.25 * torch.randn((channels, rank), device=DEVICE, dtype=DTYPE)
    )
    optimizer = torch.optim.Adam([raw_rates, raw_weights], lr=0.035)

    best_loss = float("inf")
    best_state: tuple[torch.Tensor, torch.Tensor] | None = None
    for _ in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        rates, weights = unpack(raw_rates, raw_weights)
        prediction = lifted_response(times, weights, rates)
        residual = prediction[train_idx] - observations[train_idx]
        loss = residual.square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_rates, raw_weights], 10.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = (raw_rates.detach().clone(), raw_weights.detach().clone())

    assert best_state is not None
    raw_rates.data.copy_(best_state[0])
    raw_weights.data.copy_(best_state[1])

    # A deterministic second stage is important here: incomplete optimization can
    # look exactly like an additional memory mode and corrupt order selection.
    refiner = torch.optim.LBFGS(
        [raw_rates, raw_weights],
        lr=0.8,
        max_iter=lbfgs_steps,
        tolerance_grad=1.0e-11,
        tolerance_change=1.0e-13,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        refiner.zero_grad(set_to_none=True)
        local_rates, local_weights = unpack(raw_rates, raw_weights)
        local_prediction = lifted_response(times, local_weights, local_rates)
        local_loss = (local_prediction[train_idx] - observations[train_idx]).square().mean()
        local_loss.backward()
        return local_loss

    refiner.step(closure)
    rates, weights = unpack(raw_rates, raw_weights)
    prediction = lifted_response(times, weights, rates)
    train_residual = prediction[train_idx] - observations[train_idx]
    val_residual = prediction[val_idx] - observations[val_idx]
    train_rmse = torch.sqrt(train_residual.square().mean())
    val_rmse = torch.sqrt(val_residual.square().mean())
    n = train_residual.numel()
    parameters = rank + channels * rank
    rss = torch.sum(train_residual.square()).clamp_min(1.0e-30)
    bic = n * torch.log(rss / n) + parameters * math.log(n)

    # Local identifiability diagnostic on noise-free model outputs.
    flat = torch.cat([raw_rates.detach(), raw_weights.detach().reshape(-1)]).requires_grad_(True)

    def predict_from_flat(vector: torch.Tensor) -> torch.Tensor:
        rr = vector[:rank]
        rw = vector[rank:].reshape(channels, rank)
        local_rates, local_weights = unpack(rr, rw)
        return lifted_response(times[train_idx], local_weights, local_rates).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(predict_from_flat, flat, vectorize=True)
    singular = torch.linalg.svdvals(jacobian)
    condition = singular[0] / singular[-1].clamp_min(1.0e-15)

    return FitResult(
        rank=rank,
        seed=seed,
        train_rmse=float(train_rmse.detach().cpu()),
        val_rmse=float(val_rmse.detach().cpu()),
        bic=float(bic.detach().cpu()),
        jacobian_condition=float(condition.detach().cpu()),
        rates=[float(v) for v in rates.detach().cpu()],
        weights=[[float(v) for v in row] for row in weights.detach().cpu()],
    )


def make_case(name: str, rates: list[float], channels: int, noise: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    times = torch.linspace(0.0, 12.0, 97, dtype=DTYPE, device=DEVICE)
    rank = len(rates)
    base = np.linspace(0.30, 0.75, channels)[:, None]
    modulation = np.linspace(0.85, 1.15, rank)[None, :]
    weights_np = base * modulation / rank
    weights = torch.tensor(weights_np, dtype=DTYPE, device=DEVICE)
    rates_tensor = torch.tensor(rates, dtype=DTYPE, device=DEVICE)
    clean = lifted_response(times, weights, rates_tensor)
    noisy = clean + noise * torch.tensor(rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE)

    all_idx = np.arange(times.numel())
    train_np = np.sort(rng.choice(all_idx[1:76], size=42, replace=False))
    val_np = np.arange(76, times.numel())
    train_idx = torch.tensor(train_np, dtype=torch.long, device=DEVICE)
    val_idx = torch.tensor(val_np, dtype=torch.long, device=DEVICE)

    fits: list[FitResult] = []
    for candidate_rank in (1, 2, 3):
        candidates = [
            fit_rank(times, noisy, train_idx, val_idx, candidate_rank, seed * 100 + fit_seed)
            for fit_seed in range(3)
        ]
        fits.append(min(candidates, key=lambda item: item.bic))

    ordered = sorted(fits, key=lambda item: item.bic)
    winner, runner_up = ordered[:2]
    bic_gap = runner_up.bic - winner.bic
    accepted = bic_gap >= 6.0 and winner.jacobian_condition <= 1.0e8 and winner.val_rmse <= max(4 * noise, 2.5e-3)
    decision = f"RANK_{winner.rank}" if accepted else "INSUFFICIENT_EVIDENCE"

    return {
        "name": name,
        "true_rank": rank,
        "channels": channels,
        "noise_std": noise,
        "true_rates": rates,
        "decision": decision,
        "selected_rank_before_refusal": winner.rank,
        "bic_gap": bic_gap,
        "winner_condition": winner.jacobian_condition,
        "winner_val_rmse": winner.val_rmse,
        "fits": [asdict(item) for item in fits],
    }


def write_report(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "memory_rank_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Memory-rank feasibility probe",
        "",
        f"Device: `{payload['device']}`",
        "",
        "| Case | True rank | Channels | Decision | BIC gap | Jacobian condition | Validation RMSE |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['name']} | {case['true_rank']} | {case['channels']} | "
            f"{case['decision']} | {case['bic_gap']:.3g} | "
            f"{case['winner_condition']:.3g} | {case['winner_val_rmse']:.3g} |"
        )
    lines.extend(
        [
            "",
            "This probe is a route-selection test. It does not establish statistical coverage,",
            "global identifiability, or scientific validity on real data.",
        ]
    )
    (RESULTS / "memory_rank_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cases = [
        make_case("scalar_rank1", [0.35], channels=1, noise=5.0e-4, seed=11),
        make_case("scalar_rank2_separated", [0.16, 1.55], channels=1, noise=5.0e-4, seed=12),
        make_case("scalar_rank2_near_coincident", [0.40, 0.48], channels=1, noise=5.0e-4, seed=13),
        make_case("field_shared_rank2", [0.16, 1.55], channels=12, noise=1.0e-3, seed=14),
    ]
    payload = {
        "experiment": "minimal_latent_memory_rank_probe",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "decision_rule": {
            "bic_gap_min": 6.0,
            "jacobian_condition_max": 1.0e8,
            "validation_rmse_max": "max(4 * noise_std, 2.5e-3)",
        },
        "cases": cases,
    }
    write_report(payload)
    print(json.dumps({"device": str(DEVICE), "decisions": {c["name"]: c["decision"] for c in cases}}, indent=2))


if __name__ == "__main__":
    main()
