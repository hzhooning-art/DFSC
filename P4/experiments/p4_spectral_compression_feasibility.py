"""Feasibility test for a differentiable fractional-structure compression."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))


def ml_family_batch(alpha, rate, times, terms=100):
    """Broadcasted series evaluator used by this benchmark."""
    z = -rate * times.pow(alpha)
    k = torch.arange(terms, dtype=alpha.dtype, device=alpha.device)
    return torch.sum(z[..., None].pow(k) / torch.exp(torch.lgamma(alpha[..., None] * k + 1.0)), dim=-1)


def chebyshev(x, degree):
    values = [torch.ones_like(x)]
    if degree > 1:
        values.append(x)
    for _ in range(2, degree):
        values.append(2.0 * x * values[-1] - values[-2])
    return torch.stack(values, dim=-1)


def features(alpha, rate, times, degrees):
    # The log-time coordinate resolves the rapid early-time variation.
    xa = 2.0 * (alpha - 0.65) / (0.91 - 0.65) - 1.0
    xr = 2.0 * (rate - 0.40) / (0.90 - 0.40) - 1.0
    xt = 2.0 * (torch.log(times) - torch.log(torch.tensor(0.05, dtype=times.dtype, device=times.device))) / (torch.log(torch.tensor(2.0, dtype=times.dtype, device=times.device)) - torch.log(torch.tensor(0.05, dtype=times.dtype, device=times.device))) - 1.0
    pa, pr, pt = (chebyshev(x, d) for x, d in zip((xa, xr, xt), degrees))
    return torch.einsum("ni,nj,nk->nijk", pa, pr, pt).reshape(alpha.shape[0], -1)


def fit_compressor(degrees, device):
    alpha = torch.linspace(0.65, 0.91, 16, dtype=torch.float64, device=device)
    rate = torch.linspace(0.40, 0.90, 16, dtype=torch.float64, device=device)
    times = torch.linspace(0.05, 2.0, 32, dtype=torch.float64, device=device)
    aa, rr, tt = torch.meshgrid(alpha, rate, times, indexing="ij")
    flat_a, flat_r, flat_t = aa.reshape(-1), rr.reshape(-1), tt.reshape(-1)
    x = features(flat_a, flat_r, flat_t, degrees)
    y = ml_family_batch(flat_a, flat_r, flat_t, terms=100)
    coefficients = torch.linalg.lstsq(x, y[:, None]).solution[:, 0]
    return coefficients, degrees


def compressed_value(alpha, rate, times, coefficients, degrees):
    return features(alpha, rate, times, degrees) @ coefficients


def evaluate_model(name, value_fn, test_a, test_r, test_t, reference, ref_ga, ref_gr, repeats=30):
    a = test_a.detach().clone().requires_grad_(True)
    r = test_r.detach().clone().requires_grad_(True)
    start = time.perf_counter()
    for _ in range(repeats):
        pred = value_fn(a, r, test_t)
    if test_t.is_cuda:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    grad_a, grad_r = torch.autograd.grad(pred.sum(), (a, r))
    value_rmse = torch.sqrt(torch.mean((pred.detach() - reference) ** 2))
    grad_rmse = torch.sqrt(torch.mean((grad_a.detach() - ref_ga) ** 2 + (grad_r.detach() - ref_gr) ** 2))
    return {
        "model": name,
        "value_rmse": float(value_rmse.cpu()),
        "gradient_rmse": float(grad_rmse.cpu()),
        "mean_batch_seconds": elapsed,
        "gradient_finite": bool(torch.isfinite(grad_a).all() and torch.isfinite(grad_r).all()),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(48000)
    n_test = 4096
    test_a = 0.65 + 0.26 * torch.rand(n_test, dtype=torch.float64, device=device)
    test_r = 0.40 + 0.50 * torch.rand(n_test, dtype=torch.float64, device=device)
    test_t = 0.05 * (2.0 / 0.05) ** torch.rand(n_test, dtype=torch.float64, device=device)
    ref_a = test_a.detach().clone().requires_grad_(True)
    ref_r = test_r.detach().clone().requires_grad_(True)
    reference = ml_family_batch(ref_a, ref_r, test_t, terms=100)
    ref_ga, ref_gr = torch.autograd.grad(reference.sum(), (ref_a, ref_r))

    results = []
    for terms in [8, 16]:
        results.append(evaluate_model(
            f"mlsl_series_{terms}",
            lambda a, r, t, k=terms: ml_family_batch(a, r, t, terms=k),
            test_a, test_r, test_t, reference.detach(), ref_ga.detach(), ref_gr.detach(),
        ))
    compression_rows = []
    for degrees in [(4, 4, 4), (6, 6, 6), (8, 8, 6)]:
        coefficients, used_degrees = fit_compressor(degrees, device)
        row = evaluate_model(
            "chebyshev_{}x{}x{}".format(*degrees),
            lambda a, r, t, c=coefficients, d=used_degrees: compressed_value(a, r, t, c, d),
            test_a, test_r, test_t, reference.detach(), ref_ga.detach(), ref_gr.detach(),
        )
        row.update({
            "degrees": list(degrees),
            "coefficient_count": int(coefficients.numel()),
            "coefficient_bytes": int(coefficients.numel() * coefficients.element_size()),
        })
        compression_rows.append(row)
        results.append(row)
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "test_samples": n_test,
        "reference_terms": 100,
        "parameter_domain": {"alpha": [0.65, 0.91], "rate": [0.40, 0.90], "time": [0.05, 2.0]},
        "results": results,
        "compression_summary": compression_rows,
        "interpretation": "controlled feasibility only; Chebyshev representation is a candidate compression, not a finalized algorithm",
    }
    out = ROOT / "P4" / "results" / "p4_spectral_compression_feasibility.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
