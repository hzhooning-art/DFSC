"""Multi-seed statistical check for the revised P4 feasibility route."""

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
sys.path.insert(0, str(ROOT / "P4" / "diagnostics"))
from mechanism_audit import audit_candidates  # noqa: E402

sys.path.insert(0, str(ROOT / "P1" / "paper1_mlsl"))
from dfsc.factory import MLSLConfig, build_dirichlet_mlsl_1d  # noqa: E402


def train_one(seed, train, base_train, target_train, test_id, base_id, target_id, test_ood, base_ood, target_ood):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pure = PureConditionalMLP(32).double().to(train[0].device)
    hybrid = HybridResidual(32).double().to(train[0].device)
    opt_pure = torch.optim.Adam(pure.parameters(), lr=2e-3)
    opt_hybrid = torch.optim.Adam(hybrid.parameters(), lr=2e-3)
    for _ in range(120):
        pred = pure(*train)
        loss = torch.mean((pred - target_train) ** 2)
        opt_pure.zero_grad(); loss.backward(); opt_pure.step()
        pred_h = hybrid(base_train, *train)
        loss_h = torch.mean((pred_h - target_train) ** 2)
        opt_hybrid.zero_grad(); loss_h.backward(); opt_hybrid.step()
    with torch.no_grad():
        pure_id = pure(*test_id)
        pure_ood = pure(*test_ood)
        hyb_id = hybrid(base_id, *test_id)
        hyb_ood = hybrid(base_ood, *test_ood)
    return {
        "pure_id": relative_error(pure_id, target_id),
        "pure_ood": relative_error(pure_ood, target_ood),
        "hybrid_id": relative_error(hyb_id, target_id),
        "hybrid_ood": relative_error(hyb_ood, target_ood),
    }


def summary(values, key):
    x = torch.tensor([v[key] for v in values], dtype=torch.float64)
    return {"mean": float(x.mean()), "std": float(x.std(unbiased=True)), "values": [float(v) for v in x]}


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
        target_train = base_train + 0.06 * torch.exp(-0.5 * train[1][:, None]) * torch.tanh(train[0])
        target_id = base_id + 0.06 * torch.exp(-0.5 * test_id[1][:, None]) * torch.tanh(test_id[0])
        target_ood = base_ood + 0.06 * torch.exp(-0.5 * test_ood[1][:, None]) * torch.tanh(test_ood[0])
    rows = [train_one(s, train, base_train, target_train, test_id, base_id, target_id, test_ood, base_ood, target_ood) for s in [11, 23, 37, 51, 73]]
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": [11, 23, 37, 51, 73],
        "steps_per_seed": 120,
        "pure_conditional_mlp": {k: summary(rows, k) for k in ["pure_id", "pure_ood"]},
        "hybrid_residual": {k: summary(rows, k) for k in ["hybrid_id", "hybrid_ood"]},
        "hybrid_ood_mean_improvement_ratio": float(
            summary(rows, "pure_ood")["mean"] / summary(rows, "hybrid_ood")["mean"]
        ),
        "p2_style_selection_audit": audit_candidates(
            {
                "pure_conditional_mlp": summary(rows, "pure_ood")["mean"],
                "hybrid_residual": summary(rows, "hybrid_ood")["mean"],
            },
            early_winner="hybrid_residual",
            late_winner="hybrid_residual",
            minimum_confidence=0.75,
        ).__dict__,
        "interpretation": "statistical feasibility check; controlled synthetic family only",
    }
    out = ROOT / "P4" / "results" / "p4_parameter_family_multiseed.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
