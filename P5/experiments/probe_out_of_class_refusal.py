"""Test refusal for memory mechanisms outside the positive-real contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, FitResult, fit_rank, lifted_response


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def oscillatory_response(
    times: torch.Tensor,
    amplitudes: torch.Tensor,
    decay: float,
    frequency: float,
) -> torch.Tensor:
    """Response for K(t)=a exp(-decay*t) cos(frequency*t)."""
    channels = amplitudes.numel()
    matrices = torch.zeros((channels, 3, 3), dtype=DTYPE, device=DEVICE)
    matrices[:, 0, 1] = -amplitudes
    matrices[:, 1, 0] = 1.0
    matrices[:, 1, 1] = -decay
    matrices[:, 1, 2] = -frequency
    matrices[:, 2, 1] = frequency
    matrices[:, 2, 2] = -decay
    propagators = torch.matrix_exp(times[:, None, None, None] * matrices[None, :, :, :])
    initial = torch.zeros((channels, 3), dtype=DTYPE, device=DEVICE)
    initial[:, 0] = 1.0
    states = torch.einsum("tcij,cj->tci", propagators, initial)
    return states[:, :, 0]


def generate_case(kind: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    rng = np.random.default_rng(seed)
    channels = 6
    noise_std = 6.0e-4
    times = torch.linspace(0.0, 14.0, 97, dtype=DTYPE, device=DEVICE)
    scale = torch.linspace(0.72, 1.28, channels, dtype=DTYPE, device=DEVICE)

    if kind == "positive_rank1":
        weights = (0.34 * scale)[:, None]
        clean = lifted_response(times, weights, torch.tensor([0.30], dtype=DTYPE, device=DEVICE))
    elif kind == "positive_rank2":
        weights = torch.stack([0.28 * scale, 0.16 / scale], dim=1)
        clean = lifted_response(times, weights, torch.tensor([0.18, 1.45], dtype=DTYPE, device=DEVICE))
    elif kind == "signed_rank2":
        weights = torch.stack([0.46 * scale, -0.13 / scale], dim=1)
        clean = lifted_response(times, weights, torch.tensor([0.22, 1.35], dtype=DTYPE, device=DEVICE))
    elif kind == "oscillatory_kernel":
        clean = oscillatory_response(times, 0.38 * scale, decay=0.24, frequency=1.25)
    else:
        raise ValueError(f"Unknown case: {kind}")

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


def prediction_from_fit(times: torch.Tensor, fit: FitResult) -> torch.Tensor:
    weights = torch.tensor(fit.weights, dtype=DTYPE, device=DEVICE)
    rates = torch.tensor(fit.rates, dtype=DTYPE, device=DEVICE)
    return lifted_response(times, weights, rates)


def mean_lag1(residual: torch.Tensor) -> float:
    values = []
    centered = residual - residual.mean(dim=0, keepdim=True)
    for channel in range(centered.shape[1]):
        x = centered[:-1, channel]
        y = centered[1:, channel]
        denominator = torch.sqrt(torch.sum(x.square()) * torch.sum(y.square())).clamp_min(1.0e-20)
        values.append(float((torch.sum(x * y) / denominator).detach().cpu()))
    return float(np.mean(values))


def evaluate(kind: str, repeat: int) -> dict:
    seed = 9000 + repeat + sum(ord(char) for char in kind)
    times, observations, train_idx, val_idx, noise_std = generate_case(kind, seed)
    fits = []
    for rank in (1, 2, 3, 4):
        candidates = [
            fit_rank(
                times,
                observations,
                train_idx,
                val_idx,
                rank,
                seed * 100 + local_seed,
                adam_steps=260,
                lbfgs_steps=75,
            )
            for local_seed in range(2)
        ]
        fits.append(min(candidates, key=lambda item: item.bic))

    ordered = sorted(fits, key=lambda item: item.bic)
    winner = ordered[0]
    prediction = prediction_from_fit(times, winner)
    validation_residual = prediction[val_idx] - observations[val_idx]
    lag1 = mean_lag1(validation_residual)
    cap_gain = next(f.bic for f in fits if f.rank == 3) - next(f.bic for f in fits if f.rank == 4)

    prediction_ok = winner.val_rmse <= max(4.0 * noise_std, 3.0e-3)
    condition_ok = winner.jacobian_condition <= 1.0e8
    residual_ok = abs(lag1) <= 0.55
    cap_saturated = winner.rank == 4 and cap_gain >= 6.0
    accepted = prediction_ok and condition_ok and residual_ok and not cap_saturated

    return {
        "case": kind,
        "repeat": repeat,
        "decision": "ACCEPT_CONTRACT" if accepted else "REFUSE_CONTRACT",
        "selected_rank": winner.rank,
        "validation_rmse": winner.val_rmse,
        "jacobian_condition": winner.jacobian_condition,
        "validation_residual_lag1": lag1,
        "rank4_vs_rank3_bic_gain": cap_gain,
        "cap_saturated": cap_saturated,
        "checks": {
            "prediction_ok": prediction_ok,
            "condition_ok": condition_ok,
            "residual_ok": residual_ok,
        },
        "candidate_bic": {str(f.rank): f.bic for f in fits},
    }


def main() -> None:
    labels = {
        "positive_rank1": "in_class",
        "positive_rank2": "in_class",
        "signed_rank2": "out_of_class",
        "oscillatory_kernel": "out_of_class",
    }
    records = []
    for kind in labels:
        for repeat in range(3):
            record = evaluate(kind, repeat)
            records.append(record)
            print(
                f"case={kind:18s} repeat={repeat} decision={record['decision']} "
                f"rank={record['selected_rank']} rmse={record['validation_rmse']:.3g} "
                f"lag1={record['validation_residual_lag1']:.3g}"
            )

    summary = []
    for kind, scope in labels.items():
        group = [record for record in records if record["case"] == kind]
        accept_fraction = sum(r["decision"] == "ACCEPT_CONTRACT" for r in group) / len(group)
        summary.append(
            {
                "case": kind,
                "scope": scope,
                "accept_fraction": accept_fraction,
                "refuse_fraction": 1.0 - accept_fraction,
                "selected_ranks": [r["selected_rank"] for r in group],
            }
        )

    in_class_ok = all(row["accept_fraction"] >= 2.0 / 3.0 for row in summary if row["scope"] == "in_class")
    out_class_ok = all(row["refuse_fraction"] >= 2.0 / 3.0 for row in summary if row["scope"] == "out_of_class")
    route_pass = in_class_ok and out_class_ok
    payload = {
        "experiment": "positive_real_contract_out_of_class_refusal",
        "device": str(DEVICE),
        "labels_used_only_for_scoring": labels,
        "records": records,
        "summary": summary,
        "route_pass": route_pass,
        "route_pass_rule": "accept >=2/3 for each in-class case and refuse >=2/3 for each out-of-class case",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "out_of_class_refusal.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Positive-real contract refusal probe",
        "",
        "| Case | Scope | Accept fraction | Refuse fraction | Selected ranks |",
        "|---|---|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['case']} | {row['scope']} | {row['accept_fraction']:.2f} | "
            f"{row['refuse_fraction']:.2f} | {row['selected_ranks']} |"
        )
    lines.extend(["", f"Route pass: **{route_pass}**.", ""])
    (RESULTS / "out_of_class_refusal.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"route_pass={route_pass}")


if __name__ == "__main__":
    main()

