"""P4 mechanism-selection probe under partial temporal observation."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from distributed_order_reference_benchmark import fit_distributed_order  # noqa: E402
from strong_baseline_benchmark import fit_constrained, fit_fractional  # noqa: E402


def rel(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y)).detach())


def apply_kernel(kernel: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    n = u.numel()
    idx = torch.arange(n, device=u.device)
    lag = idx[:, None] - idx[None, :]
    return dt * ((lag >= 0).to(u.dtype) * kernel[lag.clamp_min(0)]).tril() @ u


def normalize(x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    return x / (x.sum() * dt).clamp_min(1e-12)


def target(tau: torch.Tensor, kind: str, dt: torch.Tensor) -> torch.Tensor:
    if kind == "tempered_power_law":
        raw = tau.pow(-0.62) * torch.exp(-0.045 * tau)
    elif kind == "two_timescale":
        raw = 0.75 * 0.12 * torch.exp(-0.12 * tau) + 0.25 * 2.5 * torch.exp(-2.5 * tau)
    else:
        raise ValueError(kind)
    return normalize(raw, dt)


def one(kind: str, seed: int, device: torch.device) -> dict[str, object]:
    torch.manual_seed(seed)
    n = 192
    t = torch.linspace(0.0, 32.0, n, device=device)
    dt = t[1] - t[0]
    tau = t + 0.5 * dt
    truth = target(tau, kind, dt)
    obs = (truth + 0.025 * truth.max() * torch.randn_like(truth)).clamp_min(0.0)
    train = (t <= 14.0) & ((torch.arange(n, device=device) % 2) == 0)
    valid = (t > 14.0) & (t <= 20.0) & ((torch.arange(n, device=device) % 2) == 0)
    candidates = {
        "exponential_mixture": fit_constrained(tau, obs, train | valid, dt),
        "distributed_order": fit_distributed_order(t, tau, obs, train | valid, dt),
        "fractional_tempered": fit_fractional(tau, obs, train | valid, dt),
    }
    # Validation is computed only on the later observed prefix, not on the test horizon.
    validation_scores = {name: float(((kernel[valid] - obs[valid]) ** 2).mean().detach()) for name, kernel in candidates.items()}
    selected = min(validation_scores, key=validation_scores.get)
    u = torch.sin(0.45 * t) + 0.3 * torch.cos(0.17 * t) + torch.exp(-0.5 * ((t - 3.0) / 0.3) ** 2)
    reference = apply_kernel(truth, u, dt)
    test = t > 20.0
    test_scores = {name: rel(apply_kernel(kernel, u, dt)[test], reference[test]) for name, kernel in candidates.items()}
    best = min(test_scores, key=test_scores.get)
    return {
        "kind": kind,
        "seed": seed,
        "selected": selected,
        "oracle_test_best": best,
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "selected_test_error": test_scores[selected],
        "oracle_test_error": test_scores[best],
        "selection_regret": test_scores[selected] - test_scores[best],
        "correct_family_selection": selected == best,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [one(kind, seed, device) for kind in ("tempered_power_law", "two_timescale") for seed in (3, 7, 11)]
    summary = {}
    for kind in ("tempered_power_law", "two_timescale"):
        group = [r for r in rows if r["kind"] == kind]
        regrets = torch.tensor([float(r["selection_regret"]) for r in group])
        summary[kind] = {
            "selection_accuracy": sum(bool(r["correct_family_selection"]) for r in group) / len(group),
            "selected_test_error_mean": float(torch.tensor([float(r["selected_test_error"]) for r in group]).mean()),
            "oracle_test_error_mean": float(torch.tensor([float(r["oracle_test_error"]) for r in group]).mean()),
            "selection_regret_mean": float(regrets.mean()),
            "selection_regret_std": float(regrets.std(unbiased=True)),
        }
    out = Path(__file__).parents[1] / "results" / "p4_mechanism_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
