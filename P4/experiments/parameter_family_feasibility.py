"""Basic feasibility test for the revised P4 parameter-conditioned family.

This is a gate experiment, not a paper-grade benchmark.  It reuses the P1
Mittag-Leffler evaluator and spectral basis, then compares:

* MLSL-only: the known fractional propagator;
* a pure conditional MLP;
* a hybrid model whose residual is learned on top of the propagator.

The target family contains a small, explicitly controlled nonlinear residual
so that the hybrid model has a meaningful task.  The script reports ID/OOD
errors, parameter gradients, and a high-term evaluator discrepancy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
P1_SRC = ROOT / "P1" / "paper1_mlsl"
sys.path.insert(0, str(P1_SRC))

from dfsc.factory import MLSLConfig, build_dirichlet_mlsl_1d  # noqa: E402
from dfsc.mittag_leffler import mittag_leffler_e  # noqa: E402


def seed_all(seed: int = 20260810) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ParameterizedMLSL(nn.Module):
    """P1 spectral layer with a differentiable modal-rate scale lambda."""

    def __init__(self, layer: nn.Module, terms: int = 80) -> None:
        super().__init__()
        self.layer = layer
        self.terms = terms

    def forward(
        self,
        u0: torch.Tensor,
        times: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        lam: torch.Tensor,
    ) -> torch.Tensor:
        phi = self.layer.eigenvectors.to(dtype=u0.dtype, device=u0.device)
        proj = self.layer.projection_vectors.to(dtype=u0.dtype, device=u0.device)
        eig = self.layer.eigenvalues.to(dtype=u0.dtype, device=u0.device)
        coeff = u0 @ proj
        mu = eig.pow(beta[:, None] / 2.0) * lam[:, None]
        safe_t = times.clamp_min(torch.finfo(times.dtype).tiny)
        if times.ndim == 1 and times.shape[0] == u0.shape[0]:
            t_power = safe_t.pow(alpha)
            z = -mu * t_power[:, None]
            per_sample_time = True
        else:
            z = -mu[:, None, :] * safe_t[None, :, None].pow(alpha[:, None, None])
            per_sample_time = False
        # P1's hybrid branch currently routes masks with scalar alpha.  Keep
        # the batch dimension explicit here; this is a P4 feasibility probe,
        # while preserving the exact P1 evaluator rather than replacing it.
        if per_sample_time:
            kernel = torch.stack([
                mittag_leffler_e(alpha[i], z[i], terms=self.terms,
                                 custom_backward=False, method="hybrid")
                for i in range(u0.shape[0])
            ])
        else:
            kernel = torch.stack([
                mittag_leffler_e(alpha[i], z[i], terms=self.terms,
                                 custom_backward=False, method="hybrid")
                for i in range(u0.shape[0])
            ])
        if per_sample_time:
            return torch.matmul(coeff * kernel, phi.T)
        return torch.matmul(coeff[:, None, :] * kernel, phi.T)


class PureConditionalMLP(nn.Module):
    def __init__(self, n: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n + 4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n),
        )

    def forward(self, u0, t, alpha, beta, lam):
        return self.net(torch.cat([u0, t[:, None], alpha[:, None], beta[:, None], lam[:, None]], dim=-1))


class HybridResidual(nn.Module):
    def __init__(self, n: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n + 4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n),
        )

    def forward(self, base, u0, t, alpha, beta, lam):
        features = torch.cat([base, t[:, None], alpha[:, None], beta[:, None], lam[:, None]], dim=-1)
        return base + 0.15 * self.net(features)


def exact_family(mlsl, u0, t, alpha, beta, lam):
    base = mlsl(u0, t, alpha, beta, lam)
    # A deliberately small model discrepancy: the known propagator remains
    # dominant, while the residual head has a nontrivial but bounded task.
    correction = 0.06 * torch.exp(-0.5 * t[:, None]) * torch.tanh(u0)
    return base + correction


def batch(n_samples, n, device, *, ood=False):
    if ood:
        alpha = torch.tensor([0.70, 0.90], dtype=torch.float64, device=device).repeat(n_samples // 2)
        beta = torch.tensor([1.70, 2.30], dtype=torch.float64, device=device).repeat(n_samples // 2)
        lam = torch.tensor([0.65, 1.35], dtype=torch.float64, device=device).repeat(n_samples // 2)
    else:
        alpha = torch.empty(n_samples, dtype=torch.float64, device=device).uniform_(0.76, 0.84)
        beta = torch.empty(n_samples, dtype=torch.float64, device=device).uniform_(1.85, 2.05)
        lam = torch.empty(n_samples, dtype=torch.float64, device=device).uniform_(0.85, 1.15)
    t = torch.empty(n_samples, dtype=torch.float64, device=device).uniform_(0.08, 0.95)
    x = torch.linspace(0.0, 1.0, n, dtype=torch.float64, device=device)
    u0 = torch.sin(torch.pi * x)[None, :] + 0.25 * torch.sin(2.0 * torch.pi * x)[None, :]
    u0 = u0.expand(n_samples, -1) + 0.03 * torch.randn(n_samples, n, dtype=torch.float64, device=device)
    return u0, t, alpha, beta, lam


def relative_error(pred, target):
    return float((torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target)).detach().cpu())


def main() -> None:
    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MLSLConfig(terms=40, beta=2.0, dtype=torch.float64, device=device)
    _, layer = build_dirichlet_mlsl_1d(num_points=32, num_modes=12, config=cfg)
    layer = layer.to(device)
    mlsl = ParameterizedMLSL(layer).to(device)

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

    pure = PureConditionalMLP(32).double().to(device)
    hybrid = HybridResidual(32).double().to(device)
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
        pure_id, pure_ood = pure(*test_id), pure(*test_ood)
        hyb_id = hybrid(base_id, *test_id)
        hyb_ood = hybrid(base_ood, *test_ood)

    alpha = torch.tensor([0.81], dtype=torch.float64, device=device, requires_grad=True)
    beta = torch.tensor([1.97], dtype=torch.float64, device=device, requires_grad=True)
    lam = torch.tensor([1.03], dtype=torch.float64, device=device, requires_grad=True)
    probe_u0, probe_t, _, _, _ = batch(1, 32, device)
    probe = mlsl(probe_u0, probe_t, alpha, beta, lam).sum()
    ga, gb, gl = torch.autograd.grad(probe, (alpha, beta, lam))

    ref_layer = ParameterizedMLSL(layer, terms=80)
    ref = ref_layer(*test_id)
    coarse = mlsl(*test_id)
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_steps": 120,
        "id_relative_l2": {
            "mlsl_only": relative_error(base_id, target_id),
            "pure_conditional_mlp": relative_error(pure_id, target_id),
            "hybrid_residual": relative_error(hyb_id, target_id),
        },
        "ood_relative_l2": {
            "mlsl_only": relative_error(base_ood, target_ood),
            "pure_conditional_mlp": relative_error(pure_ood, target_ood),
            "hybrid_residual": relative_error(hyb_ood, target_ood),
        },
        "gradient_abs": {"alpha": float(ga.abs().item()), "beta": float(gb.abs().item()), "lambda": float(gl.abs().item())},
        "gradient_finite": bool(torch.isfinite(torch.stack([ga, gb, gl])).all().item()),
        "evaluator_terms_40_vs_80_relative_l2": relative_error(coarse, ref),
        "interpretation": "basic feasibility gate; not evidence of broad P4 generalization",
    }
    out = ROOT / "P4" / "results" / "p4_parameter_family_feasibility.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
