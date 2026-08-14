"""Matrix-scale probe for fractional spectral compression."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "P1" / "paper1_mlsl"))
from dfsc.mittag_leffler import mittag_leffler_e  # noqa: E402


def ml_coeff(alpha, rate, times, eigenvalues, terms=100):
    z = -rate * times[:, None].pow(alpha) * eigenvalues[None, :]
    return mittag_leffler_e(alpha, z, terms=terms, custom_backward=False, method="series")


def apply_spectral(alpha, rate, times, vectors, basis, eigenvalues, rank=None, terms=100):
    if rank is None:
        basis_used, eig_used = basis, eigenvalues
    else:
        basis_used, eig_used = basis[:, :rank], eigenvalues[:rank]
    modes = vectors @ basis_used
    coeff = ml_coeff(alpha, rate, times, eig_used, terms=terms)
    return (modes * coeff) @ basis_used.T


def benchmark(name, fn, alpha, rate, times, vectors, reference, ref_grad, repeats=5):
    a = alpha.detach().clone().requires_grad_(True)
    start = time.perf_counter()
    for _ in range(repeats):
        output = fn(a, rate, times, vectors)
    if vectors.is_cuda:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    grad = torch.autograd.grad(output.sum(), a)[0]
    return {
        "model": name,
        "value_rmse": float(torch.sqrt(torch.mean((output.detach() - reference) ** 2)).cpu()),
        "gradient_abs_error": float((grad.detach() - ref_grad).abs().cpu()),
        "mean_batch_seconds": elapsed,
        "gradient_finite": bool(torch.isfinite(grad).item()),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(49000)
    n, batch = 256, 512
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, device=device))
    eigenvalues = torch.linspace(0.1, 4.0, n, dtype=torch.float64, device=device)
    alpha = torch.tensor(0.78, dtype=torch.float64, device=device)
    rate = torch.tensor(0.65, dtype=torch.float64, device=device)
    times = 0.20 * (2.0 / 0.20) ** torch.rand(batch, dtype=torch.float64, device=device)
    vectors = torch.randn(batch, n, dtype=torch.float64, device=device)
    reference = apply_spectral(alpha, rate, times, vectors, q, eigenvalues, rank=None, terms=100)
    probe_alpha = alpha.detach().clone().requires_grad_(True)
    ref_probe = apply_spectral(probe_alpha, rate, times, vectors, q, eigenvalues, rank=None, terms=100)
    ref_grad = torch.autograd.grad(ref_probe.sum(), probe_alpha)[0].detach()
    rows = []
    for rank in [32, 64, 128, 256, 512]:
        rows.append(benchmark(
            f"spectral_truncation_r{rank}",
            lambda a, r, t, x, k=rank: apply_spectral(a, r, t, x, q, eigenvalues, rank=k, terms=100),
            alpha, rate, times, vectors, reference, ref_grad,
        ))
        rows[-1].update({
            "rank": rank,
            "basis_bytes": int(n * rank * 8),
            "compression_ratio": n / rank,
        })
    rows.append(benchmark(
        "full_spectral_reference",
        lambda a, r, t, x: apply_spectral(a, r, t, x, q, eigenvalues, rank=None, terms=100),
        alpha, rate, times, vectors, reference, ref_grad,
    ))
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dimension": n,
        "batch": batch,
        "time_domain": [0.20, 2.0],
        "reference_terms": 100,
        "full_basis_bytes": int(n * n * 8),
        "results": rows,
        "exit_rule": "retain compression only if one rank has lower time and memory with value RMSE <= 1e-4 and gradient error <= 1e-3",
        "interpretation": "matrix-scale probe; spectral truncation is a baseline, not a final compression algorithm",
    }
    out = ROOT / "P4" / "results" / "p4_matrix_compression_probe.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
