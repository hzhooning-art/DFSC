"""P4 robustness sweep with a frozen selective-mechanism threshold."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from mechanism_selection import apply_kernel, fit_constrained, fit_fractional, fit_distributed_order, target  # noqa: E402


def one(kind: str, seed: int, noise: float, observed_end: float, device: torch.device) -> dict[str, object]:
    torch.manual_seed(seed)
    n = 192
    t = torch.linspace(0.0, 32.0, n, device=device)
    dt = t[1] - t[0]
    tau = t + 0.5 * dt
    truth = target(tau, kind, dt)
    obs = (truth + noise * truth.max() * torch.randn_like(truth)).clamp_min(0.0)
    train = (t <= observed_end * 0.7) & ((torch.arange(n, device=device) % 2) == 0)
    valid = (t > observed_end * 0.7) & (t <= observed_end) & ((torch.arange(n, device=device) % 2) == 0)
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
    test = t > observed_end
    errors = {name: float((torch.linalg.vector_norm(apply_kernel(kernel, u, dt)[test] - reference[test]) / torch.linalg.vector_norm(reference[test])).detach()) for name, kernel in candidates.items()}
    oracle = min(errors.values())
    return {"kind": kind, "seed": seed, "noise": noise, "observed_end": observed_end, "margin": margin, "accepted": margin >= 0.005, "selected_error": errors[ordered[0]], "oracle_error": oracle, "regret": errors[ordered[0]] - oracle}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [one(kind, seed, noise, end, device) for kind in ("tempered_power_law", "two_timescale") for noise in (0.0, 0.05) for end in (10.0, 14.0) for seed in (31, 37, 41)]
    summary = {}
    for kind in ("tempered_power_law", "two_timescale"):
        for noise in (0.0, 0.05):
            for end in (10.0, 14.0):
                group = [r for r in rows if r["kind"] == kind and r["noise"] == noise and r["observed_end"] == end]
                accepted = [r for r in group if r["accepted"]]
                key = f"{kind}|noise={noise}|observed_end={end}"
                summary[key] = {
                    "count": len(group),
                    "coverage": len(accepted) / len(group),
                    "accepted_regret_mean": float(torch.tensor([float(r["regret"]) for r in accepted]).mean()) if accepted else None,
                    "unconditional_regret_mean": float(torch.tensor([float(r["regret"]) for r in group]).mean()),
                    "accepted_error_mean": float(torch.tensor([float(r["selected_error"]) for r in accepted]).mean()) if accepted else None,
                }
    out = Path(__file__).parents[1] / "results" / "p4_robustness_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"frozen_threshold": 0.005, "device": str(device), "rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"frozen_threshold": 0.005, "device": str(device), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
