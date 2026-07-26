"""Calibrate the adaptive Lanczos controller against dense spectral actions."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dfsc


RESULTS = ROOT / "generated_results"
SEEDS = (0, 1, 2, 3, 4)
SIZES = (64, 128)
TOLERANCES = (1e-3, 1e-5, 1e-7)


def random_weighted_laplacian(size: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = 0.4 + torch.rand(size - 1, generator=generator, dtype=torch.float64)
    operator = torch.zeros((size, size), dtype=torch.float64)
    indices = torch.arange(size - 1)
    operator[indices, indices] += weights
    operator[indices + 1, indices + 1] += weights
    operator[indices, indices + 1] -= weights
    operator[indices + 1, indices] -= weights
    return operator.to(device)


def dense_reference(
    operator: torch.Tensor,
    u0: torch.Tensor,
    times: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(operator)
    coefficients = eigenvectors.transpose(-1, -2) @ u0
    z = -times[:, None].pow(alpha) * eigenvalues.clamp_min(0.0)[None, :]
    kernel = dfsc.mittag_leffler_e(alpha, z, terms=180, method="hybrid")
    return (kernel * coefficients[None, :]) @ eigenvectors.transpose(-1, -2)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for size in SIZES:
        for seed in SEEDS:
            torch.manual_seed(seed)
            operator = random_weighted_laplacian(size, seed, device)
            u0 = torch.randn(size, dtype=torch.float64, device=device)
            alpha = torch.tensor(0.55 + 0.1 * (seed % 4), dtype=torch.float64, device=device)
            times = torch.linspace(0.0, 2.0, 17, dtype=torch.float64, device=device)
            reference = dense_reference(operator, u0, times, alpha)
            reference_norm = torch.linalg.vector_norm(reference)
            for tolerance in TOLERANCES:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                values, diagnostics = dfsc.adaptive_lanczos_mittag_leffler_action(
                    operator,
                    u0,
                    times,
                    alpha,
                    dimension_schedule=(4, 8, 12, 16, 24, 32, 48, 64),
                    rtol=tolerance,
                    atol=tolerance * 1e-2,
                    strict=False,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                actual_error = float(torch.linalg.vector_norm(values - reference) / reference_norm)
                estimated_error = diagnostics.relative_disagreements[-1] if diagnostics.relative_disagreements else None
                rows.append(
                    {
                        "device": str(device),
                        "size": size,
                        "seed": seed,
                        "alpha": float(alpha),
                        "requested_rtol": tolerance,
                        "converged": diagnostics.converged,
                        "selected_dimension": diagnostics.selected_dimension,
                        "estimated_relative_disagreement": estimated_error,
                        "actual_relative_error": actual_error,
                        "seconds": elapsed,
                    }
                )

    csv_path = RESULTS / "adaptive_krylov_calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for tolerance in TOLERANCES:
        selected = [row for row in rows if row["requested_rtol"] == tolerance]
        actual = np.asarray([float(row["actual_relative_error"]) for row in selected])
        summary_rows.append(
            {
                "requested_rtol": tolerance,
                "runs": len(selected),
                "convergence_rate": float(np.mean([bool(row["converged"]) for row in selected])),
                "selected_dimension_mean": float(np.mean([int(row["selected_dimension"]) for row in selected])),
                "actual_error_mean": float(np.mean(actual)),
                "actual_error_max": float(np.max(actual)),
                "fraction_actual_error_below_10x_rtol": float(np.mean(actual <= 10.0 * tolerance)),
            }
        )
    payload = {
        "device": str(device),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "estimator_boundary": "successive Krylov disagreement is empirical and is not a rigorous error bound",
        "summary": summary_rows,
    }
    (RESULTS / "adaptive_krylov_calibration_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
