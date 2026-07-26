"""Reference, gradient, convergence, and GPU checks for the next dfsc stage."""

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


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(torch.finfo(reference.dtype).eps)
    return float((torch.linalg.vector_norm(actual - reference) / denominator).detach().cpu())


def operator_matrix(size: int, *, device: str = "cpu") -> torch.Tensor:
    diagonal = 2.0 * torch.ones(size, dtype=torch.float64, device=device)
    off_diagonal = -torch.ones(size - 1, dtype=torch.float64, device=device)
    return torch.diag(diagonal) + torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)


def writable_path(path: Path) -> tuple[Path, object]:
    """Open a result path, falling back when a stale Windows handle locks it."""

    candidates = (
        path,
        path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}"),
        ROOT / f"{path.stem}_{int(time.time())}{path.suffix}",
    )
    error: PermissionError | None = None
    for candidate in candidates:
        try:
            return candidate, candidate.open("w", newline="", encoding="utf-8")
        except PermissionError as exc:
            error = exc
    assert error is not None
    raise error


def main() -> None:
    torch.manual_seed(36)
    torch.set_default_dtype(torch.float64)
    size = 32
    operator = operator_matrix(size)
    u0 = torch.randn(size)
    times = torch.linspace(0.0, 0.1, 6)
    alpha = torch.tensor(0.82)
    problem = dfsc.OperatorSpectralProblem(operator, u0, times, alpha)
    reference = dfsc.solve(problem, dfsc.MLSLOperator()).values

    rows: list[dict[str, object]] = []
    for dimension in (4, 8, 16, 32):
        solution = dfsc.solve(
            problem,
            dfsc.MLSLKrylov(krylov_dimension=dimension, estimate_error=True),
        )
        rows.append(
            {
                "krylov_dimension": dimension,
                "relative_error_vs_full_eigh": relative_error(solution.values, reference),
                "embedded_relative_disagreement": solution.diagnostics.get(
                    "embedded_relative_disagreement", float("nan")
                ),
                "effective_dimension": solution.diagnostics["max_effective_krylov_dimension"],
                "finite": bool(torch.isfinite(solution.values).all()),
            }
        )

    alpha_krylov = torch.tensor(0.82, requires_grad=True)
    alpha_full = torch.tensor(0.82, requires_grad=True)
    krylov_grad_solution = dfsc.solve(
        dfsc.OperatorSpectralProblem(operator, u0, times, alpha_krylov),
        dfsc.MLSLKrylov(krylov_dimension=size, estimate_error=False),
    )
    full_grad_solution = dfsc.solve(
        dfsc.OperatorSpectralProblem(operator, u0, times, alpha_full),
        dfsc.MLSLOperator(),
    )
    krylov_grad_solution.values.square().mean().backward()
    full_grad_solution.values.square().mean().backward()
    alpha_gradient_relative_error = float(
        (torch.abs(alpha_krylov.grad - alpha_full.grad) / torch.abs(alpha_full.grad).clamp_min(1e-14)).detach()
    )

    x, layer = dfsc.build_mlsl(
        dimension=1,
        boundary="dirichlet",
        num_points=32,
        num_modes=10,
        config=dfsc.MLSLConfig.stable(terms=100),
    )
    semilinear_alpha = torch.tensor(0.9, requires_grad=True)
    gamma = torch.tensor(0.02, requires_grad=True)
    semilinear_times = torch.linspace(0.0, 0.03, 6)
    semilinear_solution = dfsc.solve(
        dfsc.SemilinearSpectralProblem(
            layer,
            torch.sin(torch.pi * x),
            semilinear_times,
            semilinear_alpha,
            lambda state: -gamma * state.pow(3),
        ),
        dfsc.MLSLPicard(max_iterations=15, tolerance=1e-8, quadrature_points=16),
    )
    semilinear_solution.final.square().mean().backward()

    gpu = {
        "available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "krylov_finite": None,
    }
    if torch.cuda.is_available():
        gpu_operator = operator_matrix(size, device="cuda")
        gpu_values, _ = dfsc.lanczos_mittag_leffler_action(
            gpu_operator,
            u0.to("cuda"),
            times.to("cuda"),
            alpha.to("cuda"),
            krylov_dimension=16,
        )
        gpu["krylov_finite"] = bool(torch.isfinite(gpu_values).all())

    summary = {
        "krylov_full_dimension_relative_error": rows[-1]["relative_error_vs_full_eigh"],
        "krylov_alpha_gradient_relative_error": alpha_gradient_relative_error,
        "krylov_convergence_rows": len(rows),
        "semilinear_retcode": semilinear_solution.retcode,
        "semilinear_iterations": semilinear_solution.diagnostics["picard_iterations"],
        "semilinear_residual": semilinear_solution.diagnostics["picard_residual"],
        "semilinear_alpha_gradient_finite": bool(torch.isfinite(semilinear_alpha.grad)),
        "semilinear_gamma_gradient_finite": bool(torch.isfinite(gamma.grad)),
        "gpu": gpu,
    }

    table_path = ROOT / "results" / "tables" / "krylov_convergence.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path, handle = writable_path(table_path)
    with handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = ROOT / "results" / "krylov_semilinear_summary.json"
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
