"""Validate sparse and matrix-free Lanczos MLSL representations."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dfsc


def laplacian_dense(size: int, *, device: str = "cpu") -> torch.Tensor:
    diagonal = 2.0 * torch.ones(size, dtype=torch.float64, device=device)
    off_diagonal = -torch.ones(size - 1, dtype=torch.float64, device=device)
    return torch.diag(diagonal) + torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)


def laplacian_matvec(vector: torch.Tensor) -> torch.Tensor:
    result = 2.0 * vector
    left = torch.cat((torch.zeros_like(vector[:1]), vector[:-1]))
    right = torch.cat((vector[1:], torch.zeros_like(vector[:1])))
    return result - left - right


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(torch.finfo(reference.dtype).eps)
    return float((torch.linalg.vector_norm(actual - reference) / denominator).detach().cpu())


def write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates = (
        path,
        path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}"),
        ROOT / f"{path.stem}_{int(time.time())}{path.suffix}",
    )
    for candidate in candidates:
        try:
            handle = candidate.open("w", newline="", encoding="utf-8")
            path = candidate
            break
        except PermissionError:
            continue
    else:
        raise PermissionError(f"could not write {path}")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    torch.manual_seed(37)
    torch.set_default_dtype(torch.float64)
    size = 64
    dense = laplacian_dense(size)
    sparse = dense.to_sparse_coo()
    operator = dfsc.SelfAdjointLinearOperator(
        size,
        laplacian_matvec,
        torch.float64,
        "cpu",
        name="matrix-free-dirichlet-laplacian",
    )
    u0 = torch.randn(size)
    times = torch.linspace(0.0, 0.08, 5)
    alpha = torch.tensor(0.82)
    reference = dfsc.solve(
        dfsc.OperatorSpectralProblem(dense, u0, times, alpha),
        dfsc.MLSLOperator(),
    ).values

    rows: list[dict[str, object]] = []
    for name, representation in (("dense", dense), ("sparse", sparse), ("matrix_free", operator)):
        started = time.perf_counter()
        values, diagnostics = dfsc.lanczos_mittag_leffler_action(
            representation,
            u0,
            times,
            alpha,
            krylov_dimension=size,
        )
        rows.append(
            {
                "representation": name,
                "relative_error_vs_full_eigh": relative_error(values, reference),
                "elapsed_seconds": time.perf_counter() - started,
                "effective_dimension": max(diagnostics.effective_dimensions),
                "finite": bool(torch.isfinite(values).all()),
            }
        )

    scale = torch.tensor(1.1, requires_grad=True)
    differentiable_operator = dfsc.SelfAdjointLinearOperator(
        size,
        lambda vector: scale * laplacian_matvec(vector),
        torch.float64,
        "cpu",
        name="trainable-scaled-laplacian",
    )
    trainable_solution = dfsc.solve(
        dfsc.LinearOperatorSpectralProblem(differentiable_operator, u0, times, alpha),
        dfsc.MLSLKrylov(krylov_dimension=32, estimate_error=False),
    )
    trainable_solution.final.square().mean().backward()

    large_size = 4096
    large_operator = dfsc.SelfAdjointLinearOperator(
        large_size,
        laplacian_matvec,
        torch.float64,
        "cpu",
        name="large-matrix-free-laplacian",
    )
    large_u0 = torch.sin(torch.pi * torch.linspace(0.0, 1.0, large_size))
    started = time.perf_counter()
    large_values, large_diagnostics = dfsc.lanczos_mittag_leffler_action(
        large_operator,
        large_u0,
        times,
        alpha,
        krylov_dimension=32,
    )
    large_elapsed = time.perf_counter() - started

    gpu = {
        "available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "matrix_free_finite": None,
        "alpha_gradient_finite": None,
    }
    if torch.cuda.is_available():
        gpu_size = 1024
        gpu_alpha = torch.tensor(0.82, device="cuda", requires_grad=True)
        gpu_operator = dfsc.SelfAdjointLinearOperator(
            gpu_size,
            laplacian_matvec,
            torch.float64,
            "cuda",
            name="cuda-matrix-free-laplacian",
        )
        gpu_values, _ = dfsc.lanczos_mittag_leffler_action(
            gpu_operator,
            torch.randn(gpu_size, dtype=torch.float64, device="cuda"),
            times.to("cuda"),
            gpu_alpha,
            krylov_dimension=24,
        )
        gpu_values[-1].square().mean().backward()
        gpu["matrix_free_finite"] = bool(torch.isfinite(gpu_values).all())
        gpu["alpha_gradient_finite"] = bool(torch.isfinite(gpu_alpha.grad))

    dense_storage_bytes = large_size * large_size * torch.tensor([], dtype=torch.float64).element_size()
    matrix_free_working_bytes = large_size * 32 * torch.tensor([], dtype=torch.float64).element_size()
    summary = {
        "small_reference_size": size,
        "sparse_relative_error": rows[1]["relative_error_vs_full_eigh"],
        "matrix_free_relative_error": rows[2]["relative_error_vs_full_eigh"],
        "operator_parameter_gradient_finite": bool(torch.isfinite(scale.grad)),
        "large_matrix_free_size": large_size,
        "large_matrix_free_elapsed_seconds": large_elapsed,
        "large_matrix_free_finite": bool(torch.isfinite(large_values).all()),
        "large_effective_krylov_dimension": max(large_diagnostics.effective_dimensions),
        "estimated_dense_storage_bytes": dense_storage_bytes,
        "estimated_krylov_basis_bytes": matrix_free_working_bytes,
        "estimated_storage_reduction_ratio": dense_storage_bytes / matrix_free_working_bytes,
        "gpu": gpu,
    }
    table_path = write_rows(ROOT / "results" / "tables" / "sparse_matrix_free_validation.csv", rows)
    summary_path = ROOT / "results" / "sparse_matrix_free_summary.json"
    try:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except PermissionError:
        summary_path = summary_path.with_name(f"{summary_path.stem}_{int(time.time())}.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["table_path"] = str(table_path)
    summary["summary_path"] = str(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
