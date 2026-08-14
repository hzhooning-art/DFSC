"""Independent MLSL validation for the cross-primitive protocol.

This script deliberately exercises the P1 implementation through its public
factory API.  The reference is an arbitrary-precision series evaluation on a
declared negative-real, moderate-spectrum regime; it is not a claim about all
Mittag-Leffler arguments.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import mpmath as mp
import torch


ROOT = Path(__file__).resolve().parents[2]
P1_PACKAGE = ROOT / "P1" / "paper1_mlsl"
RESULTS = ROOT / "P4" / "results"
sys.path.insert(0, str(P1_PACKAGE))

import dfsc  # noqa: E402


torch.set_default_dtype(torch.float64)
mp.mp.dps = 80


def ml_reference(alpha: float, z: float, terms: int = 700) -> float:
    """High-precision E_alpha(z) series on the tested negative-real regime."""

    alpha_mp = mp.mpf(str(alpha))
    z_mp = mp.mpf(str(z))
    value = mp.mpf("0")
    power = mp.mpf("1")
    for k in range(terms):
        value += power / mp.gamma(alpha_mp * k + 1)
        power *= z_mp
        if k > 40 and abs(power / mp.gamma(alpha_mp * (k + 1) + 1)) < mp.mpf("1e-70"):
            break
    return float(value)


def reference_layer(layer, u0: torch.Tensor, t: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Evaluate the same spectral representation with arbitrary precision kernels."""

    phi = layer.eigenvectors.detach().cpu()
    eig = layer.eigenvalues.detach().cpu()
    coeff = torch.matmul(u0.detach().cpu(), layer.projection_vectors.detach().cpu())
    rows = []
    for time_value in t.detach().cpu().reshape(-1).tolist():
        kernels = [
            ml_reference(alpha, -(layer.wave_speed**2) * float(ev) ** (beta / 2.0) * time_value**alpha)
            for ev in eig.tolist()
        ]
        kernel = torch.tensor(kernels, dtype=torch.float64)
        rows.append(torch.matmul(coeff * kernel, phi.transpose(-1, -2)))
    return torch.stack(rows)


def rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = torch.maximum(expected.abs(), torch.full_like(expected, 1e-14))
    return float((torch.abs(actual.detach().cpu() - expected) / scale).max())


def make_case(dtype: torch.dtype = torch.float64):
    cfg = dfsc.MLSLConfig.stable(terms=120, dtype=dtype)
    x, layer = dfsc.build_mlsl(
        dimension=1,
        boundary="dirichlet",
        num_points=16,
        num_modes=3,
        config=cfg,
    )
    u0 = torch.sin(torch.pi * x).to(dtype)
    return x, layer, u0


def finite_difference_loss(layer, u0, t, alpha, beta, eps=2e-5):
    with torch.no_grad():
        plus_a = layer(u0, t, torch.tensor(alpha + eps), beta=torch.tensor(beta))
        minus_a = layer(u0, t, torch.tensor(alpha - eps), beta=torch.tensor(beta))
        plus_b = layer(u0, t, torch.tensor(alpha), beta=torch.tensor(beta + eps))
        minus_b = layer(u0, t, torch.tensor(alpha), beta=torch.tensor(beta - eps))
        return (
            float(((plus_a.square().mean() - minus_a.square().mean()) / (2 * eps))),
            float(((plus_b.square().mean() - minus_b.square().mean()) / (2 * eps))),
        )


def main() -> None:
    torch.manual_seed(20260812)
    x, layer, u0 = make_case()
    alpha_value, beta_value = 0.78, 1.8
    t = torch.tensor([0.01, 0.05, 0.2, 0.6], dtype=torch.float64)

    # Value and parameter-gradient checks use the actual P1 layer.
    alpha = torch.tensor(alpha_value, requires_grad=True)
    beta = torch.tensor(beta_value, requires_grad=True)
    actual = layer(u0, t, alpha, beta=beta)
    expected = reference_layer(layer, u0, t, alpha_value, beta_value)
    loss = actual.square().mean()
    grad_alpha, grad_beta = torch.autograd.grad(loss, (alpha, beta))
    fd_alpha, fd_beta = finite_difference_loss(layer, u0, t, alpha_value, beta_value)
    grad_rel = {
        "alpha": abs(float(grad_alpha) - fd_alpha) / max(abs(fd_alpha), 1e-12),
        "beta": abs(float(grad_beta) - fd_beta) / max(abs(fd_beta), 1e-12),
    }

    # Calibration: recover both differentiable orders from observations.
    target = reference_layer(layer, u0, t, alpha_value, beta_value).detach()
    fit_alpha = torch.tensor(0.9, requires_grad=True)
    fit_beta = torch.tensor(2.1, requires_grad=True)
    optimizer = torch.optim.Adam((fit_alpha, fit_beta), lr=0.025)
    calibration_losses = []
    for _ in range(140):
        optimizer.zero_grad()
        prediction = layer(u0, t, fit_alpha, beta=fit_beta)
        calibration_loss = (prediction - target).square().mean()
        calibration_loss.backward()
        optimizer.step()
        with torch.no_grad():
            fit_alpha.clamp_(0.55, 1.05)
            fit_beta.clamp_(1.1, 2.5)
        calibration_losses.append(float(calibration_loss.detach()))

    # Reuse inside a differentiable residual module.
    batch_u0 = torch.stack([u0, 0.8 * u0, 1.2 * u0])
    residual_head = torch.nn.Sequential(torch.nn.Linear(16, 12), torch.nn.Tanh(), torch.nn.Linear(12, 16))
    reused_alpha = torch.tensor(0.8, requires_grad=True)
    reused_beta = torch.tensor(1.9, requires_grad=True)
    propagated = layer(batch_u0, t[-2:], reused_alpha, beta=reused_beta)
    residual = residual_head(propagated)
    reuse_loss = (propagated + 0.1 * residual).square().mean()
    reuse_loss.backward()
    reuse_grads = [p.grad for p in residual_head.parameters()] + [reused_alpha.grad, reused_beta.grad]

    # OOD and long-horizon checks stay within the declared negative-real scope.
    t_ood = torch.tensor([0.025, 0.35, 0.9], dtype=torch.float64)
    ood_out = layer(u0, t_ood, torch.tensor(0.73), beta=torch.tensor(1.65))
    t_long = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    long_out = layer(u0, t_long, torch.tensor(alpha_value), beta=torch.tensor(beta_value))
    monotone_probe = bool(torch.all(long_out.abs().amax(dim=-1)[1:] <= long_out.abs().amax(dim=-1)[:-1] + 1e-8))

    gpu = {"available": bool(torch.cuda.is_available())}
    if gpu["available"]:
        device = torch.device("cuda")
        gpu_layer = layer.to(device)
        gpu_u0 = u0.to(device).repeat(256, 1)
        gpu_t = torch.tensor([0.05, 0.2, 0.6], device=device)
        for _ in range(10):
            _ = gpu_layer(gpu_u0, gpu_t, torch.tensor(0.8, device=device), beta=torch.tensor(1.8, device=device))
        torch.cuda.synchronize()
        start = time.perf_counter()
        gpu_out = None
        for _ in range(30):
            gpu_out = gpu_layer(gpu_u0, gpu_t, torch.tensor(0.8, device=device), beta=torch.tensor(1.8, device=device))
        torch.cuda.synchronize()
        gpu["batch"] = 256
        gpu["query_times"] = 3
        gpu["mean_ms"] = 1000.0 * (time.perf_counter() - start) / 30.0
        gpu["finite"] = bool(torch.isfinite(gpu_out).all().item())

    gates = {
        "value": bool(torch.isfinite(actual).all().item() and rel_error(actual, expected) < 2e-8),
        "gradient": bool(all(value < 2e-4 for value in grad_rel.values())),
        "calibration": bool(
            torch.isfinite(fit_alpha).item()
            and torch.isfinite(fit_beta).item()
            and abs(float(fit_alpha.detach()) - alpha_value) < 0.08
            and abs(float(fit_beta.detach()) - beta_value) < 0.12
        ),
        "module_reuse": bool(all(g is not None and torch.isfinite(g).all().item() for g in reuse_grads)),
        "ood": bool(torch.isfinite(ood_out).all().item()),
        "long_horizon": bool(torch.isfinite(long_out).all().item() and monotone_probe),
    }
    result = {
        "schema": "DFSC-MLSL-Protocol-Validation-v1",
        "implementation": {"package": "P1/paper1_mlsl/dfsc", "api": "build_mlsl", "preset": "MLSLConfig.stable"},
        "scope": {
            "boundary": "1D Dirichlet",
            "modes": 3,
            "negative_real_spectrum": True,
            "alpha_range_tested": [0.73, 0.9],
            "beta_range_tested": [1.65, 2.1],
            "reference": "80-digit mpmath series on moderate negative-real arguments",
        },
        "value": {"max_relative_error": rel_error(actual, expected), "finite": bool(torch.isfinite(actual).all().item())},
        "gradient": {"relative_error_vs_central_difference": grad_rel, "finite": True},
        "calibration": {
            "true": {"alpha": alpha_value, "beta": beta_value},
            "initial": {"alpha": 0.9, "beta": 2.1},
            "estimated": {"alpha": float(fit_alpha.detach()), "beta": float(fit_beta.detach())},
            "final_loss": calibration_losses[-1],
            "iterations": len(calibration_losses),
        },
        "module_reuse": {"batch": 3, "residual_head": "MLP(16,12,16)", "finite_gradients": gates["module_reuse"]},
        "ood": {"finite": bool(torch.isfinite(ood_out).all().item()), "query_times": t_ood.tolist()},
        "long_horizon": {"finite": bool(torch.isfinite(long_out).all().item()), "monotone_amplitude_probe": monotone_probe, "query_times": t_long.tolist()},
        "gpu": gpu,
        "gates": gates,
        "status": "conformant" if all(gates.values()) else "nonconformant",
        "limitations": [
            "validated only for the stable real-negative spectral regime",
            "the high-precision reference uses the convergent series and does not validate complex or positive arguments",
            "GPU timing is a local profile, not a cross-machine benchmark",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "p4_mlsl_protocol_validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
