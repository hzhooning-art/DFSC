"""Formal P4 validation: independent calibration, test, and bootstrap intervals."""

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
    val = {name: float(((kernel[valid] - obs[valid]) ** 2).mean().detach()) for name, kernel in candidates.items()}
    ordered = sorted(val, key=val.get)
    margin = (val[ordered[1]] - val[ordered[0]]) / max(val[ordered[1]], 1e-12)
    u = torch.sin(0.45 * t) + 0.3 * torch.cos(0.17 * t) + torch.exp(-0.5 * ((t - 3.0) / 0.3) ** 2)
    reference = apply_kernel(truth, u, dt)
    test = t > 20.0
    errors = {name: float((torch.linalg.vector_norm(apply_kernel(kernel, u, dt)[test] - reference[test]) / torch.linalg.vector_norm(reference[test])).detach()) for name, kernel in candidates.items()}
    oracle = min(errors.values())
    return {"kind": kind, "seed": seed, "margin": margin, "selected": ordered[0], "selected_error": errors[ordered[0]], "oracle_error": oracle, "regret": errors[ordered[0]] - oracle}


def bootstrap_mean(values: list[float], seed: int = 91, draws: int = 2000) -> tuple[float, float, float]:
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor(values)
    idx = torch.randint(0, len(values), (draws, len(values)), generator=g)
    means = x[idx].mean(1)
    q = torch.quantile(means, torch.tensor((0.025, 0.975)))
    return float(x.mean()), float(q[0]), float(q[1])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibration = [one(kind, seed, device) for kind in ("tempered_power_law", "two_timescale") for seed in range(100, 106)]
    test_rows = [one(kind, seed, device) for kind in ("tempered_power_law", "two_timescale") for seed in range(200, 208)]
    summary = {}
    for kind in ("tempered_power_law", "two_timescale"):
        cal = [r for r in calibration if r["kind"] == kind]
        test = [r for r in test_rows if r["kind"] == kind]
        # Calibrate a 50% coverage target using only calibration margins.
        threshold = float(torch.quantile(torch.tensor([float(r["margin"]) for r in cal]), 0.5))
        accepted = [r for r in test if float(r["margin"]) >= threshold]
        regrets = [float(r["regret"]) for r in accepted]
        errors = [float(r["selected_error"]) for r in accepted]
        summary[kind] = {
            "calibration_count": len(cal),
            "test_count": len(test),
            "calibrated_threshold": threshold,
            "test_coverage": len(accepted) / len(test),
            "accepted_count": len(accepted),
            "accepted_regret_bootstrap_95": bootstrap_mean(regrets) if regrets else None,
            "accepted_error_bootstrap_95": bootstrap_mean(errors) if errors else None,
            "unconditional_regret_bootstrap_95": bootstrap_mean([float(r["regret"]) for r in test]),
        }
    out = Path(__file__).parents[1] / "results" / "p4_formal_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "calibration": calibration, "test": test_rows, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
