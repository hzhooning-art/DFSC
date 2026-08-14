"""Cross-module reuse test for the differentiable fractional primitive."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from p4_spectral_compression_feasibility import ml_family_batch  # noqa: E402


class PureMLP(torch.nn.Module):
    def __init__(self, context, horizon):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(context, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, horizon))

    def forward(self, context):
        return self.net(context)


class MLSLEncoderDecoder(torch.nn.Module):
    def __init__(self, context, target_times):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(context, 64), torch.nn.Tanh(), torch.nn.Linear(64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2))
        self.target_times = target_times

    def forward(self, context):
        raw = self.encoder(context)
        alpha = 0.60 + 0.35 * torch.sigmoid(raw[:, 0])
        rate = 0.30 + 0.70 * torch.sigmoid(raw[:, 1])
        times = self.target_times.expand(context.shape[0], -1)
        return ml_family_batch(alpha[:, None].expand_as(times).reshape(-1), rate[:, None].expand_as(times).reshape(-1), times.reshape(-1), terms=16).reshape_as(times)


class MLSLResidualAdapter(torch.nn.Module):
    def __init__(self, context, target_times):
        super().__init__()
        self.decoder = MLSLEncoderDecoder(context, target_times)
        self.residual = torch.nn.Sequential(torch.nn.Linear(context, 64), torch.nn.Tanh(), torch.nn.Linear(64, len(target_times)))

    def forward(self, context):
        return self.decoder(context) + 0.1 * self.residual(context)


def make_data(device, n, alpha_range, rate_range, context_times, target_times, seed):
    torch.manual_seed(seed)
    alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * torch.rand(n, dtype=torch.float64, device=device)
    rate = rate_range[0] + (rate_range[1] - rate_range[0]) * torch.rand(n, dtype=torch.float64, device=device)
    all_times = torch.cat((context_times, target_times))
    repeated = all_times.expand(n, -1)
    signal = ml_family_batch(alpha[:, None].expand_as(repeated).reshape(-1), rate[:, None].expand_as(repeated).reshape(-1), repeated.reshape(-1), terms=100).reshape(n, -1)
    noisy = signal + 0.005 * torch.randn_like(signal)
    return noisy[:, : len(context_times)].float(), noisy[:, len(context_times):].float(), alpha, rate


def train_eval(model, train_context, train_target, test_context, test_target, steps=500):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    start = time.perf_counter()
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(train_context)
        loss = torch.mean((pred - train_target) ** 2)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    if train_context.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        test_pred = model(test_context)
    return {
        "test_rmse": float(torch.sqrt(torch.mean((test_pred - test_target) ** 2)).cpu()),
        "train_loss_final": losses[-1],
        "elapsed_seconds": elapsed,
        "gradient_finite": all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_times = torch.linspace(0.05, 0.40, 8, dtype=torch.float64, device=device)
    target_times = torch.linspace(0.50, 1.50, 16, dtype=torch.float64, device=device)
    rows = []
    for seed in [52000, 52001, 52002]:
        train_context, train_target, _, _ = make_data(device, 512, (0.68, 0.86), (0.45, 0.80), context_times, target_times, seed)
        test_context, test_target, _, _ = make_data(device, 256, (0.65, 0.90), (0.40, 0.90), context_times, target_times, seed + 100)
        models = {
            "pure_mlp": PureMLP(8, 16).to(device).double(),
            "mlsl_encoder_decoder": MLSLEncoderDecoder(8, target_times).to(device).double(),
            "mlsl_residual_adapter": MLSLResidualAdapter(8, target_times).to(device).double(),
        }
        for name, model in models.items():
            result = train_eval(model, train_context.double(), train_target.double(), test_context.double(), test_target.double())
            result.update({"seed": seed, "model": name})
            rows.append(result)
    summary = {}
    for name in ("pure_mlp", "mlsl_encoder_decoder", "mlsl_residual_adapter"):
        group = [r for r in rows if r["model"] == name]
        vals = torch.tensor([r["test_rmse"] for r in group], dtype=torch.float64)
        summary[name] = {"test_rmse_mean": float(vals.mean()), "test_rmse_std": float(vals.std(unbiased=True)), "elapsed_mean": sum(r["elapsed_seconds"] for r in group) / len(group), "all_gradients_finite": all(r["gradient_finite"] for r in group)}
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_tasks": 512,
        "test_tasks": 256,
        "training_domain": {"alpha": [0.68, 0.86], "rate": [0.45, 0.80]},
        "test_domain": {"alpha": [0.65, 0.90], "rate": [0.40, 0.90]},
        "rows": rows,
        "summary": summary,
        "interpretation": "cross-module compatibility gate; this is not a new neural architecture claim",
    }
    out = ROOT / "P4" / "results" / "p4_module_reuse_feasibility.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
