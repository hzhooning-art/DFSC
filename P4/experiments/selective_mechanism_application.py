"""Application probe: abstain when finite observations do not identify a kernel family."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from mechanism_selection import apply_kernel, fit_constrained, fit_fractional, fit_distributed_order, target  # noqa: E402


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
    scores = {name: float(((kernel[valid] - obs[valid]) ** 2).mean().detach()) for name, kernel in candidates.items()}
    ordered = sorted(scores, key=scores.get)
    margin = (scores[ordered[1]] - scores[ordered[0]]) / max(scores[ordered[1]], 1e-12)
    u = torch.sin(0.45 * t) + 0.3 * torch.cos(0.17 * t) + torch.exp(-0.5 * ((t - 3.0) / 0.3) ** 2)
    reference = apply_kernel(truth, u, dt)
    test = t > 20.0
    test_errors = {name: float((torch.linalg.vector_norm(apply_kernel(kernel, u, dt)[test] - reference[test]) / torch.linalg.vector_norm(reference[test])).detach()) for name, kernel in candidates.items()}
    oracle = min(test_errors.values())
    return {"kind": kind, "seed": seed, "selected": ordered[0], "margin": margin, "selected_error": test_errors[ordered[0]], "oracle_error": oracle, "regret": test_errors[ordered[0]] - oracle, "scores": scores}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [one(kind, seed, device) for kind in ("tempered_power_law", "two_timescale") for seed in range(3, 9)]
    thresholds = (0.0, 0.0005, 0.001, 0.002, 0.003)
    summary = {}
    for kind in ("tempered_power_law", "two_timescale"):
        group = [r for r in rows if r["kind"] == kind]
        summary[kind] = {}
        for threshold in thresholds:
            accepted = [r for r in group if float(r["margin"]) >= threshold]
            if not accepted:
                summary[kind][str(threshold)] = {"coverage": 0.0}
                continue
            summary[kind][str(threshold)] = {
                "coverage": len(accepted) / len(group),
                "accepted_error_mean": float(torch.tensor([float(r["selected_error"]) for r in accepted]).mean()),
                "accepted_regret_mean": float(torch.tensor([float(r["regret"]) for r in accepted]).mean()),
                "accepted_regret_std": float(torch.tensor([float(r["regret"]) for r in accepted]).std(unbiased=True)) if len(accepted) > 1 else 0.0,
            }
    out = Path(__file__).parents[1] / "results" / "p4_selective_mechanism_application.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
