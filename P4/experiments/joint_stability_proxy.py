"""P4 joint stability proxy: bootstrap confidence plus temporal agreement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from mechanism_selection import apply_kernel, fit_constrained, fit_fractional, fit_distributed_order, target  # noqa: E402


def bootstrap_probabilities(residuals: torch.Tensor, seed: int, draws: int = 128) -> torch.Tensor:
    generator = torch.Generator(device=residuals.device).manual_seed(seed)
    indices = torch.randint(0, residuals.shape[1], (draws, residuals.shape[1]), generator=generator, device=residuals.device)
    scores = residuals[:, indices].mean(2).transpose(0, 1)
    winners = scores.argmin(1)
    return torch.bincount(winners, minlength=residuals.shape[0]).float() / draws


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
    names = list(candidates)
    valid_indices = torch.where(valid)[0]
    midpoint = valid_indices[len(valid_indices) // 2]
    first = valid & (t <= t[midpoint])
    second = valid & (t > t[midpoint])
    residuals_first = torch.stack([(kernel[first] - obs[first]) ** 2 for kernel in candidates.values()])
    residuals_second = torch.stack([(kernel[second] - obs[second]) ** 2 for kernel in candidates.values()])
    p_first = bootstrap_probabilities(residuals_first, seed + 1000)
    p_second = bootstrap_probabilities(residuals_second, seed + 2000)
    full_scores = torch.stack([((kernel[valid] - obs[valid]) ** 2).mean() for kernel in candidates.values()])
    selected_index = int(full_scores.argmin())
    selected = names[selected_index]
    confidence = float(torch.minimum(p_first.max(), p_second.max()))
    agreement = bool(int(p_first.argmax()) == int(p_second.argmax()))
    u = torch.sin(0.45 * t) + 0.3 * torch.cos(0.17 * t) + torch.exp(-0.5 * ((t - 3.0) / 0.3) ** 2)
    reference = apply_kernel(truth, u, dt)
    test = t > observed_end
    errors = {name: float((torch.linalg.vector_norm(apply_kernel(kernel, u, dt)[test] - reference[test]) / torch.linalg.vector_norm(reference[test])).detach()) for name, kernel in candidates.items()}
    oracle = min(errors.values())
    return {"kind": kind, "seed": seed, "noise": noise, "observed_end": observed_end, "selected": selected, "confidence": confidence, "agreement": agreement, "selected_error": errors[selected], "oracle_error": oracle, "regret": errors[selected] - oracle}


def summarize(group: list[dict[str, object]], threshold: float, require_agreement: bool) -> dict[str, object]:
    accepted = [row for row in group if float(row["confidence"]) >= threshold and (not require_agreement or bool(row["agreement"]))]
    return {"count": len(group), "coverage": len(accepted) / len(group), "accepted_count": len(accepted), "accepted_regret_mean": float(torch.tensor([float(row["regret"]) for row in accepted]).mean()) if accepted else None, "accepted_error_mean": float(torch.tensor([float(row["selected_error"]) for row in accepted]).mean()) if accepted else None, "agreement_rate": sum(bool(row["agreement"]) for row in group) / len(group)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [one(kind, seed, noise, end, device) for kind in ("tempered_power_law", "two_timescale") for noise in (0.0, 0.05) for end in (10.0, 14.0) for seed in (61, 67)]
    summary = {}
    for kind in ("tempered_power_law", "two_timescale"):
        for noise in (0.0, 0.05):
            for end in (10.0, 14.0):
                group = [row for row in rows if row["kind"] == kind and row["noise"] == noise and row["observed_end"] == end]
                for threshold in (0.75, 0.9):
                    for require_agreement in (False, True):
                        key = f"{kind}|noise={noise}|observed_end={end}|confidence>={threshold}|agreement={require_agreement}"
                        summary[key] = summarize(group, threshold, require_agreement)
    out = Path(__file__).parents[1] / "results" / "p4_joint_stability_proxy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"device": str(device), "bootstrap_draws": 128, "rows": rows, "summary": summary}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
