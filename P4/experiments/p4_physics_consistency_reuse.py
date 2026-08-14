"""Physics-consistency reuse test for the integrated DFSC workflow."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from p4_spectral_compression_feasibility import ml_family_batch  # noqa: E402


class PhysicsAwareModel(torch.nn.Module):
    def __init__(self, context, horizon):
        super().__init__()
        self.horizon = horizon
        self.encoder = torch.nn.Sequential(torch.nn.Linear(context, 64), torch.nn.Tanh(), torch.nn.Linear(64, 32), torch.nn.Tanh())
        self.parameter_head = torch.nn.Linear(32, 2)
        self.trajectory_head = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.Tanh(), torch.nn.Linear(64, horizon))

    def forward(self, context):
        latent = self.encoder(context)
        raw = self.parameter_head(latent)
        alpha = 0.60 + 0.35 * torch.sigmoid(raw[:, 0])
        rate = 0.30 + 0.70 * torch.sigmoid(raw[:, 1])
        return self.trajectory_head(latent), alpha, rate


def make_data(device, n, context_times, target_times, alpha_range, rate_range, seed):
    torch.manual_seed(seed)
    alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * torch.rand(n, dtype=torch.float64, device=device)
    rate = rate_range[0] + (rate_range[1] - rate_range[0]) * torch.rand(n, dtype=torch.float64, device=device)
    times = torch.cat((context_times, target_times))
    repeated = times.expand(n, -1)
    signal = ml_family_batch(alpha[:, None].expand_as(repeated).reshape(-1), rate[:, None].expand_as(repeated).reshape(-1), repeated.reshape(-1), terms=100).reshape(n, -1)
    noisy = signal + 0.005 * torch.randn_like(signal)
    return noisy[:, :len(context_times)].double(), noisy[:, len(context_times):].double(), alpha, rate


def train_one(model, train_context, train_target, target_times, physics_weight, steps=600):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    start = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        trajectory, alpha, rate = model(train_context)
        physics = ml_family_batch(alpha[:, None].expand(-1, len(target_times)).reshape(-1), rate[:, None].expand(-1, len(target_times)).reshape(-1), target_times.expand(len(train_context), -1).reshape(-1), terms=16).reshape_as(trajectory)
        data_loss = torch.mean((trajectory - train_target) ** 2)
        physics_loss = torch.mean((trajectory - physics) ** 2)
        loss = data_loss + physics_weight * physics_loss
        loss.backward()
        opt.step()
    if train_context.is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_times = torch.linspace(0.05, 0.40, 8, dtype=torch.float64, device=device)
    target_times = torch.linspace(0.50, 1.50, 16, dtype=torch.float64, device=device)
    rows = []
    for seed in [54000, 54001, 54002]:
        train_context, train_target, _, _ = make_data(device, 512, context_times, target_times, (0.68, 0.86), (0.45, 0.80), seed)
        test_context, test_target, test_alpha, test_rate = make_data(device, 256, context_times, target_times, (0.65, 0.90), (0.40, 0.90), seed + 100)
        for name, weight in (("data_only", 0.0), ("physics_consistent", 1.0)):
            # Keep the neural model and the reference evaluator in float64 so
            # the comparison is not affected by an implicit dtype conversion.
            model = PhysicsAwareModel(8, 16).to(device).double()
            elapsed = train_one(model, train_context, train_target, target_times, weight)
            trajectory, pred_alpha, pred_rate = model(test_context)
            grad_finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
            rows.append({
                "seed": seed,
                "model": name,
                "test_rmse": float(torch.sqrt(torch.mean((trajectory - test_target) ** 2)).detach().cpu()),
                "alpha_error": float(torch.mean((pred_alpha - test_alpha).abs()).detach().cpu()),
                "rate_error": float(torch.mean((pred_rate - test_rate).abs()).detach().cpu()),
                "physics_residual": float(torch.sqrt(torch.mean((trajectory - ml_family_batch(pred_alpha[:, None].expand(-1, 16).reshape(-1), pred_rate[:, None].expand(-1, 16).reshape(-1), target_times.expand(len(test_context), -1).reshape(-1), terms=16).reshape_as(trajectory)) ** 2)).detach().cpu()),
                "elapsed_seconds": elapsed,
                "gradient_finite": grad_finite,
            })
    summary = {}
    for name in ("data_only", "physics_consistent"):
        group = [r for r in rows if r["model"] == name]
        summary[name] = {}
        for key in ("test_rmse", "alpha_error", "rate_error", "physics_residual", "elapsed_seconds"):
            vals = torch.tensor([r[key] for r in group], dtype=torch.float64)
            summary[name][key + "_mean"] = float(vals.mean())
            summary[name][key + "_std"] = float(vals.std(unbiased=True))
        summary[name]["all_gradients_finite"] = all(r["gradient_finite"] for r in group)
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_tasks": 512,
        "test_tasks": 256,
        "training_domain": {"alpha": [0.68, 0.86], "rate": [0.45, 0.80]},
        "test_domain": {"alpha": [0.65, 0.90], "rate": [0.40, 0.90]},
        "rows": rows,
        "summary": summary,
        "interpretation": "physics-consistency compatibility gate; not a new PINN algorithm claim",
    }
    out = ROOT / "P4" / "results" / "p4_physics_consistency_reuse.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
