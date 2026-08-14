"""Next P4 gate: parameter-conditioned family with nontrivial forcing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from parameter_family_feasibility import (  # noqa: E402
    HybridResidual,
    ParameterizedMLSL,
    PureConditionalMLP,
    batch,
    relative_error,
)
sys.path.insert(0, str(ROOT / "P1" / "paper1_mlsl"))
from dfsc.factory import MLSLConfig, build_dirichlet_mlsl_1d  # noqa: E402


def forcing_target(base, sample):
    u0, t, alpha, beta, lam = sample
    x = torch.linspace(0.0, 1.0, u0.shape[-1], dtype=u0.dtype, device=u0.device)
    spatial = torch.cos(torch.pi * x)[None, :]
    time_forcing = 0.045 * torch.sin(2.5 * t[:, None]) * spatial
    state_forcing = 0.025 * torch.exp(-0.7 * t[:, None]) * torch.tanh(u0**2)
    return base + time_forcing + state_forcing


def fit(seed, train, bases, targets, tests, test_bases, test_targets):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pure = PureConditionalMLP(32).double().to(train[0].device)
    hybrid = HybridResidual(32).double().to(train[0].device)
    op1 = torch.optim.Adam(pure.parameters(), lr=2e-3)
    op2 = torch.optim.Adam(hybrid.parameters(), lr=2e-3)
    for _ in range(150):
        lp = torch.mean((pure(*train) - targets) ** 2)
        op1.zero_grad(); lp.backward(); op1.step()
        lh = torch.mean((hybrid(bases, *train) - targets) ** 2)
        op2.zero_grad(); lh.backward(); op2.step()
    with torch.no_grad():
        pure_id = pure(*tests[0]); pure_ood = pure(*tests[1])
        hyb_id = hybrid(test_bases[0], *tests[0]); hyb_ood = hybrid(test_bases[1], *tests[1])
    return {
        "pure_id": relative_error(pure_id, test_targets[0]),
        "pure_ood": relative_error(pure_ood, test_targets[1]),
        "hybrid_id": relative_error(hyb_id, test_targets[0]),
        "hybrid_ood": relative_error(hyb_ood, test_targets[1]),
    }


def stat(rows, key):
    values = torch.tensor([r[key] for r in rows], dtype=torch.float64)
    return {"mean": float(values.mean()), "std": float(values.std(unbiased=True)), "values": [float(v) for v in values]}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MLSLConfig(terms=40, beta=2.0, dtype=torch.float64, device=device)
    _, layer = build_dirichlet_mlsl_1d(num_points=32, num_modes=12, config=cfg)
    mlsl = ParameterizedMLSL(layer).to(device)
    torch.manual_seed(20260810)
    train = batch(24, 32, device)
    test_id = batch(12, 32, device)
    test_ood = batch(12, 32, device, ood=True)
    with torch.no_grad():
        base_train = mlsl(*train)
        base_id = mlsl(*test_id)
        base_ood = mlsl(*test_ood)
        target_train = forcing_target(base_train, train)
        target_id = forcing_target(base_id, test_id)
        target_ood = forcing_target(base_ood, test_ood)
    rows = [fit(s, train, base_train, target_train, (test_id, test_ood), (base_id, base_ood), (target_id, target_ood)) for s in [11, 23, 37, 51, 73]]
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "forcing": "0.045 sin(2.5 t) cos(pi x) + 0.025 exp(-0.7 t) tanh(u0^2)",
        "seeds": [11, 23, 37, 51, 73],
        "steps_per_seed": 150,
        "pure_conditional_mlp": {k: stat(rows, k) for k in ["pure_id", "pure_ood"]},
        "hybrid_residual": {k: stat(rows, k) for k in ["hybrid_id", "hybrid_ood"]},
        "hybrid_ood_improvement_ratio": stat(rows, "pure_ood")["mean"] / stat(rows, "hybrid_ood")["mean"],
        "interpretation": "forcing feasibility gate; controlled synthetic forcing, not external physical validation",
    }
    out = ROOT / "P4" / "results" / "p4_parameter_family_forcing.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
