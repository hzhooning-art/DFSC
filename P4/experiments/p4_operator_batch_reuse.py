"""Operator-style batched reuse test for the MLSL primitive."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from p4_spectral_compression_feasibility import ml_family_batch  # noqa: E402


class PureOperatorMLP(torch.nn.Module):
    def __init__(self, modes, queries):
        super().__init__()
        self.modes, self.queries = modes, queries
        self.net = torch.nn.Sequential(torch.nn.Linear(modes + 2, 128), torch.nn.GELU(), torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, modes * queries))

    def forward(self, u0, alpha, rate):
        x = torch.cat((u0, alpha[:, None], rate[:, None]), dim=1)
        return self.net(x).reshape(-1, self.queries, self.modes)


def ml_operator(u0, alpha, rate, times, eigenvalues, terms=100):
    b, q, m = u0.shape[0], times.shape[0], eigenvalues.shape[0]
    aa = alpha[:, None, None].expand(b, q, m).reshape(-1)
    rr = rate[:, None, None].expand(b, q, m).reshape(-1)
    tt = times[None, :, None].expand(b, q, m).reshape(-1)
    lam = eigenvalues[None, None, :].expand(b, q, m).reshape(-1)
    coeff = ml_family_batch(aa, rr * lam, tt, terms=terms).reshape(b, q, m)
    return u0[:, None, :] * coeff


def make_data(device, n, modes, times, alpha_range, rate_range, seed):
    torch.manual_seed(seed)
    u0 = torch.randn(n, modes, dtype=torch.float64, device=device)
    alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * torch.rand(n, dtype=torch.float64, device=device)
    rate = rate_range[0] + (rate_range[1] - rate_range[0]) * torch.rand(n, dtype=torch.float64, device=device)
    eig = torch.linspace(0.5, 2.0, modes, dtype=torch.float64, device=device)
    target = ml_operator(u0, alpha, rate, times, eig)
    return u0.float(), alpha.float(), rate.float(), target.float(), eig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modes, queries = 16, 16
    times = torch.linspace(0.05, 1.5, queries, dtype=torch.float64, device=device)
    train_u, train_a, train_r, train_y, eig = make_data(device, 1024, modes, times, (0.68, 0.86), (0.45, 0.80), 53000)
    test_u, test_a, test_r, test_y, _ = make_data(device, 256, modes, times, (0.65, 0.90), (0.40, 0.90), 53100)
    model = PureOperatorMLP(modes, queries).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    start = time.perf_counter()
    for _ in range(700):
        opt.zero_grad(set_to_none=True)
        pred = model(train_u, train_a, train_r)
        loss = torch.mean((pred - train_y) ** 2)
        loss.backward()
        opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mlp_elapsed = time.perf_counter() - start
    with torch.no_grad():
        mlp_test = model(test_u, test_a, test_r)
    # The direct MLSL module is evaluated with learnable parameters to verify
    # that the operator path remains differentiable under a batched loss.
    probe_a = test_a.double().detach().clone().requires_grad_(True)
    probe_r = test_r.double().detach().clone().requires_grad_(True)
    start = time.perf_counter()
    direct = ml_operator(test_u.double(), probe_a, probe_r, times, eig)
    loss_direct = torch.mean((direct - test_y.double()) ** 2)
    grad_a, grad_r = torch.autograd.grad(loss_direct, (probe_a, probe_r))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    direct_elapsed = time.perf_counter() - start
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "modes": modes,
        "queries": queries,
        "train_tasks": len(train_u),
        "test_tasks": len(test_u),
        "test_domain": {"alpha": [0.65, 0.90], "rate": [0.40, 0.90]},
        "pure_operator_mlp": {"test_rmse": float(torch.sqrt(torch.mean((mlp_test - test_y) ** 2)).cpu()), "training_seconds": mlp_elapsed, "gradient_finite": all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())},
        "direct_mlsl_operator": {"test_rmse": float(torch.sqrt(torch.mean((direct.detach() - test_y.double()) ** 2)).cpu()), "loss": float(loss_direct.detach().cpu()), "autograd_seconds": direct_elapsed, "gradient_finite": bool(torch.isfinite(grad_a).all() and torch.isfinite(grad_r).all())},
        "interpretation": "operator-style compatibility gate; direct MLSL is a differentiable module reference, not a new operator-learning architecture",
    }
    out = ROOT / "P4" / "results" / "p4_operator_batch_reuse.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
