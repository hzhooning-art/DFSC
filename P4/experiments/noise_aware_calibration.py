"""P4 noise-aware threshold calibration with independent test realizations."""

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
    return {"kind": kind, "seed": seed, "noise": noise, "observed_end": observed_end, "margin": margin, "selected_error": errors[ordered[0]], "oracle_error": oracle, "regret": errors[ordered[0]] - oracle}


def summarize(calibration: list[dict[str, object]], test: list[dict[str, object]], target_coverage: float = 0.5) -> dict[str, object]:
    margins = torch.tensor([float(row["margin"]) for row in calibration])
    threshold = float(torch.quantile(margins, 1.0 - target_coverage))
    fixed = [row for row in test if float(row["margin"]) >= 0.005]
    calibrated = [row for row in test if float(row["margin"]) >= threshold]
    def stats(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "count": len(rows),
            "coverage": len(rows) / len(test),
            "accepted_regret_mean": float(torch.tensor([float(row["regret"]) for row in rows]).mean()) if rows else None,
            "accepted_error_mean": float(torch.tensor([float(row["selected_error"]) for row in rows]).mean()) if rows else None,
        }
    return {"calibrated_threshold": threshold, "fixed_threshold": 0.005, "fixed": stats(fixed), "noise_aware": stats(calibrated), "calibration_count": len(calibration), "test_count": len(test)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {}
    for kind in ("tempered_power_law", "two_timescale"):
        for noise in (0.0, 0.05):
            for end in (10.0, 14.0):
                calibration = [one(kind, seed, noise, end, device) for seed in range(101, 104)]
                test = [one(kind, seed, noise, end, device) for seed in range(201, 205)]
                rows[f"{kind}|noise={noise}|observed_end={end}"] = summarize(calibration, test)
    out = Path(__file__).parents[1] / "results" / "p4_noise_aware_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"device": str(device), "target_coverage": 0.5, "summary": rows}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
