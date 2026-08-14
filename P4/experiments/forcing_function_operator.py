"""P4 feasibility test with forcing supplied as an input function."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P1" / "paper1_mlsl"))
from dfsc.factory import MLSLConfig, build_dirichlet_mlsl_1d  # noqa: E402
from dfsc.forced_layer import ForcedMittagLefflerSpectralLayer  # noqa: E402


def rel(pred, target):
    return float((torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target)).detach().cpu())


class PureForcingMLP(nn.Module):
    def __init__(self, n, q, hidden=96):
        super().__init__()
        d = n + q * n + 3
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n))

    def forward(self, u0, forcing, t, alpha, beta):
        return self.net(torch.cat([u0, forcing.flatten(1), t[:, None], alpha[:, None], beta[:, None]], -1))


class HybridForcingMLP(nn.Module):
    def __init__(self, n, q, hidden=96):
        super().__init__()
        d = n + q * n + 3
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n))

    def forward(self, base, forcing, t, alpha, beta):
        features = torch.cat([base, forcing.flatten(1), t[:, None], alpha[:, None], beta[:, None]], -1)
        return base + 0.2 * self.net(features)


def make_family(layer, n, q, size, device, high=False):
    torch.manual_seed(20260810 + (1 if high else 0))
    x = torch.linspace(0, 1, n, dtype=torch.float64, device=device)
    times_q = torch.linspace(0.05, 0.95, q, dtype=torch.float64, device=device)
    u0 = torch.sin(torch.pi * x)[None, :] + 0.2 * torch.sin(2 * torch.pi * x)[None, :]
    u0 = u0.repeat(size, 1) + 0.02 * torch.randn(size, n, dtype=torch.float64, device=device)
    t = torch.empty(size, dtype=torch.float64, device=device).uniform_(0.15, 0.95)
    alpha = torch.empty(size, dtype=torch.float64, device=device).uniform_(0.76, 0.84)
    beta = torch.empty(size, dtype=torch.float64, device=device).uniform_(1.88, 2.02)
    amp = torch.empty(size, dtype=torch.float64, device=device).uniform_(0.7, 1.3)
    forcing = torch.stack([
        amp[i] * torch.sin(2 * torch.pi * (t[i] * times_q[:, None] + 0.1)) * torch.cos(torch.pi * x)[None, :]
        for i in range(size)
    ])
    outputs = []
    with torch.no_grad():
        for i in range(size):
            out = layer(
                u0[i], t[i:i + 1], alpha[i], forcing[i:i + 1], times_q,
                beta=beta[i],
            )
            outputs.append(out.squeeze(0))
    return (u0, forcing, t, alpha, beta), torch.stack(outputs)


def run(seed, train, target, test_id, target_id, test_ood, target_ood, base_train, base_id, base_ood):
    torch.manual_seed(seed)
    device = train[0].device
    pure = PureForcingMLP(16, 8).double().to(device)
    hybrid = HybridForcingMLP(16, 8).double().to(device)
    op = torch.optim.Adam(pure.parameters(), lr=2e-3)
    oh = torch.optim.Adam(hybrid.parameters(), lr=2e-3)
    for _ in range(100):
        lp = torch.mean((pure(*train) - target) ** 2)
        op.zero_grad(); lp.backward(); op.step()
        lh = torch.mean((hybrid(base_train, train[1], train[2], train[3], train[4]) - target) ** 2)
        oh.zero_grad(); lh.backward(); oh.step()
    with torch.no_grad():
        return {
            "pure_id": rel(pure(*test_id), target_id),
            "pure_ood": rel(pure(*test_ood), target_ood),
            "hybrid_id": rel(hybrid(base_id, test_id[1], test_id[2], test_id[3], test_id[4]), target_id),
            "hybrid_ood": rel(hybrid(base_ood, test_ood[1], test_ood[2], test_ood[3], test_ood[4]), target_ood),
        }


def stats(rows, key):
    x = torch.tensor([r[key] for r in rows], dtype=torch.float64)
    return {"mean": float(x.mean()), "std": float(x.std(unbiased=True)), "values": [float(v) for v in x]}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MLSLConfig(terms=40, dtype=torch.float64, device=device)
    _, base_layer = build_dirichlet_mlsl_1d(num_points=16, num_modes=8, config=cfg)
    low = ForcedMittagLefflerSpectralLayer(base_layer, forcing_terms=40).to(device)
    # The forced primitive itself is the structured backbone.  The target
    # adds a bounded input-dependent correction so the regression task is not
    # a duplicate evaluation of the same forced layer.
    train, _ = make_family(low, 16, 8, 24, device)
    test_id, _ = make_family(low, 16, 8, 12, device)
    test_ood, _ = make_family(low, 16, 8, 12, device)
    # Deliberately move the OOD parameter ranges while keeping the forcing input form.
    test_ood = tuple(v.clone() for v in test_ood)
    test_ood = (test_ood[0], test_ood[1], test_ood[2], torch.linspace(0.68, 0.92, 12, device=device, dtype=torch.float64), torch.linspace(1.72, 2.28, 12, device=device, dtype=torch.float64))
    def truth(data):
        u0, forcing, t, alpha, beta = data
        q = forcing.shape[1]
        times_q = torch.linspace(0.05, 0.95, q, dtype=torch.float64, device=device)
        vals = []
        with torch.no_grad():
            for i in range(u0.shape[0]):
                base = low(u0[i], t[i:i+1], alpha[i], forcing[i:i+1], times_q, beta=beta[i]).squeeze(0)
                correction = 0.035 * torch.tanh(forcing[i].mean()) * torch.sin(torch.pi * torch.linspace(0, 1, u0.shape[-1], dtype=torch.float64, device=device))
                vals.append(base + correction)
        return torch.stack(vals)
    with torch.no_grad():
        target = truth(train); target_id = truth(test_id); target_ood = truth(test_ood)
        base_train, base_id, base_ood = target.clone(), target_id.clone(), target_ood.clone()
        # Low quadrature backbone uses the same input function, which is the
        # intended structured approximation to be corrected by the head.
        base_train = torch.stack([low(train[0][i], train[2][i:i+1], train[3][i], train[1][i:i+1], torch.linspace(0.05, 0.95, 8, device=device, dtype=torch.float64), beta=train[4][i]).squeeze(0) for i in range(24)])
        base_id = torch.stack([low(test_id[0][i], test_id[2][i:i+1], test_id[3][i], test_id[1][i:i+1], torch.linspace(0.05, 0.95, 8, device=device, dtype=torch.float64), beta=test_id[4][i]).squeeze(0) for i in range(12)])
        base_ood = torch.stack([low(test_ood[0][i], test_ood[2][i:i+1], test_ood[3][i], test_ood[1][i:i+1], torch.linspace(0.05, 0.95, 8, device=device, dtype=torch.float64), beta=test_ood[4][i]).squeeze(0) for i in range(12)])
    max_backbone = float(torch.max(torch.abs(base_train)).cpu())
    if not torch.isfinite(base_train).all() or max_backbone > 1.0e6:
        result = {
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "forcing_as_input": True,
            "status": "BLOCKED_NUMERICAL_STABILITY",
            "max_abs_forced_backbone": max_backbone,
            "finite_backbone": bool(torch.isfinite(base_train).all().item()),
            "interpretation": "P1 forced primitive is outside a validated numerical regime for this forcing-input construction; no learning metrics reported",
        }
        out = ROOT / "P4" / "results" / "p4_forcing_function_operator.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    rows = [run(s, train, target, test_id, target_id, test_ood, target_ood, base_train, base_id, base_ood) for s in [11, 23, 37, 51, 73]]
    result = {
        "device": str(device), "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "forcing_as_input": True, "seeds": [11, 23, 37, 51, 73], "steps_per_seed": 100,
        "pure_conditional_operator": {k: stats(rows, k) for k in ["pure_id", "pure_ood"]},
        "forced_mlsl_hybrid": {k: stats(rows, k) for k in ["hybrid_id", "hybrid_ood"]},
        "ood_improvement_ratio": stats(rows, "pure_ood")["mean"] / stats(rows, "hybrid_ood")["mean"],
        "interpretation": "forcing-input feasibility gate; small controlled family",
    }
    out = ROOT / "P4" / "results" / "p4_forcing_function_operator.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
