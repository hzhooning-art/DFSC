"""Bootstrap confidence for P4 mechanism selection and abstention."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from mechanism_selection import apply_kernel, fit_constrained, fit_fractional, fit_distributed_order, target  # noqa: E402


def one(kind: str, seed: int, noise: float, observed_end: float, device: torch.device, draws: int = 256) -> dict[str, object]:
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
    names = list(candidates)
    residuals = torch.stack([(kernel[valid] - obs[valid]) ** 2 for kernel in candidates.values()])
    generator = torch.Generator(device=device).manual_seed(seed + 7000)
    indices = torch.randint(0, residuals.shape[1], (draws, residuals.shape[1]), generator=generator, device=device)
    bootstrap_scores = residuals[:, indices].mean(2).transpose(0, 1)
    winners = bootstrap_scores.argmin(1)
    probabilities = torch.bincount(winners, minlength=len(names)).float() / draws
    confidence = float(probabilities.max())
    selected_index = int(probabilities.argmax())
    selected = names[selected_index]
    entropy = float((-(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum()).detach())
    point_scores = residuals.mean(1)
    ordered = point_scores.argsort()
    raw_margin = float(((point_scores[ordered[1]] - point_scores[ordered[0]]) / point_scores[ordered[1]].clamp_min(1e-12)).detach())
    u = torch.sin(0.45 * t) + 0.3 * torch.cos(0.17 * t) + torch.exp(-0.5 * ((t - 3.0) / 0.3) ** 2)
    reference = apply_kernel(truth, u, dt)
    test = t > observed_end
    errors = {name: float((torch.linalg.vector_norm(apply_kernel(kernel, u, dt)[test] - reference[test]) / torch.linalg.vector_norm(reference[test])).detach()) for name, kernel in candidates.items()}
    oracle = min(errors.values())
    return {"kind": kind, "seed": seed, "noise": noise, "observed_end": observed_end, "selected": selected, "confidence": confidence, "entropy": entropy, "raw_margin": raw_margin, "selected_error": errors[selected], "oracle_error": oracle, "regret": errors[selected] - oracle}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [one(kind, seed, noise, end, device) for kind in ("tempered_power_law", "two_timescale") for noise in (0.0, 0.05) for end in (10.0, 14.0) for seed in (51, 59)]
    summary = {}
    for kind in ("tempered_power_law", "two_timescale"):
        for noise in (0.0, 0.05):
            for end in (10.0, 14.0):
                group = [row for row in rows if row["kind"] == kind and row["noise"] == noise and row["observed_end"] == end]
                for confidence_threshold in (0.75, 0.9):
                    accepted = [row for row in group if float(row["confidence"]) >= confidence_threshold]
                    key = f"{kind}|noise={noise}|observed_end={end}|confidence>={confidence_threshold}"
                    summary[key] = {
                        "count": len(group),
                        "coverage": len(accepted) / len(group),
                        "accepted_regret_mean": float(torch.tensor([float(row["regret"]) for row in accepted]).mean()) if accepted else None,
                        "accepted_error_mean": float(torch.tensor([float(row["selected_error"]) for row in accepted]).mean()) if accepted else None,
                        "mean_confidence": float(torch.tensor([float(row["confidence"]) for row in group]).mean()),
                        "mean_entropy": float(torch.tensor([float(row["entropy"]) for row in group]).mean()),
                    }
    out = Path(__file__).parents[1] / "results" / "p4_bootstrap_selection_confidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"device": str(device), "bootstrap_draws": 256, "rows": rows, "summary": summary}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
