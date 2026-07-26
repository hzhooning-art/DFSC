"""High-precision and CUDA validation for complex ML and Arnoldi actions."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import mpmath as mp
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dfsc


def mp_mittag_leffler(alpha: float, z: complex, terms: int = 500) -> complex:
    alpha_mp = mp.mpf(alpha)
    z_mp = mp.mpc(z)
    total = mp.mpc(0)
    for index in range(terms):
        term = z_mp**index / mp.gamma(alpha_mp * index + 1)
        total += term
        if index > 20 and abs(term) < mp.mpf("1e-70"):
            break
    return complex(total)


def mp_matrix_action(alpha: float, matrix: torch.Tensor, time_value: float, vector: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    matrix_mp = mp.matrix([[complex(matrix[i, j]) for j in range(size)] for i in range(size)])
    vector_mp = mp.matrix([complex(value) for value in vector])
    identity = mp.eye(size)
    power = mp.eye(size)
    result = mp.eye(size)
    factor = mp.mpf(time_value) ** mp.mpf(alpha)
    scaled = -factor * matrix_mp
    for index in range(1, 400):
        power = power * scaled
        term = power / mp.gamma(mp.mpf(alpha) * index + 1)
        result += term
        if max(abs(term[i, j]) for i in range(size) for j in range(size)) < mp.mpf("1e-65"):
            break
    output = result * vector_mp
    return torch.tensor([complex(output[i]) for i in range(size)], dtype=torch.complex128)


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(torch.finfo(reference.real.dtype).eps)
    return float((torch.linalg.vector_norm(actual - reference) / denominator).detach().cpu())


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
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
    mp.mp.dps = 80
    torch.manual_seed(39)
    scalar_rows: list[dict[str, object]] = []
    points = (-2.0 + 0.5j, -1.0 - 1.0j, 0.5 + 1.5j, 1.0 - 0.75j)
    for alpha in (0.55, 0.8, 1.2, 1.8):
        z = torch.tensor(points, dtype=torch.complex128)
        evaluation = dfsc.evaluate_complex_mittag_leffler(alpha, z, terms=140)
        reference = torch.tensor(
            [mp_mittag_leffler(alpha, point) for point in points], dtype=torch.complex128
        )
        scalar_rows.append(
            {
                "alpha": alpha,
                "max_absolute_error": float(torch.max(torch.abs(evaluation.values - reference))),
                "relative_l2_error": relative_error(evaluation.values, reference),
                "embedded_relative_disagreement": evaluation.embedded_relative_disagreement,
                "converged": evaluation.converged,
            }
        )

    operator = torch.tensor(
        [[1.0 + 0.2j, 0.8 - 0.1j, 0.0j], [0.0j, 1.3 - 0.15j, 0.4j], [0.0j, 0.0j, 1.8 + 0.1j]],
        dtype=torch.complex128,
    )
    u0 = torch.tensor([1.0 + 0.1j, -0.4j, 0.25 + 0.2j], dtype=torch.complex128)
    times = torch.tensor([0.0, 0.04, 0.1], dtype=torch.float64)
    alpha = torch.tensor(0.85, dtype=torch.float64, requires_grad=True)
    values, diagnostics = dfsc.arnoldi_mittag_leffler_action(
        operator, u0, times, alpha, arnoldi_dimension=3, terms=140
    )
    matrix_reference = torch.stack(
        [mp_matrix_action(0.85, operator, float(time_value), u0) for time_value in times]
    )
    matrix_relative_error = relative_error(values, matrix_reference)
    loss = values.abs().square().mean()
    loss.backward()
    epsilon = 1e-5
    plus, _ = dfsc.arnoldi_mittag_leffler_action(
        operator, u0, times, 0.85 + epsilon, arnoldi_dimension=3, terms=140
    )
    minus, _ = dfsc.arnoldi_mittag_leffler_action(
        operator, u0, times, 0.85 - epsilon, arnoldi_dimension=3, terms=140
    )
    finite_difference = float((plus.abs().square().mean() - minus.abs().square().mean()) / (2 * epsilon))
    alpha_gradient_relative_error = abs(float(alpha.grad) - finite_difference) / max(abs(finite_difference), 1e-14)

    gpu = {
        "available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "relative_error_vs_cpu": None,
        "alpha_gradient_finite": None,
    }
    if torch.cuda.is_available():
        gpu_alpha = torch.tensor(0.85, dtype=torch.float64, device="cuda", requires_grad=True)
        gpu_values, _ = dfsc.arnoldi_mittag_leffler_action(
            operator.to("cuda"),
            u0.to("cuda"),
            times.to("cuda"),
            gpu_alpha,
            arnoldi_dimension=3,
            terms=140,
        )
        gpu_values.abs().square().mean().backward()
        gpu["relative_error_vs_cpu"] = relative_error(gpu_values.cpu(), values.detach())
        gpu["alpha_gradient_finite"] = bool(torch.isfinite(gpu_alpha.grad))

    table_path = write_csv(
        ROOT / "results" / "tables" / "complex_mittag_leffler_reference.csv", scalar_rows
    )
    summary = {
        "complex_scalar_max_absolute_error": max(row["max_absolute_error"] for row in scalar_rows),
        "complex_scalar_max_relative_error": max(row["relative_l2_error"] for row in scalar_rows),
        "complex_scalar_all_converged": all(row["converged"] for row in scalar_rows),
        "arnoldi_matrix_relative_error": matrix_relative_error,
        "arnoldi_alpha_gradient_relative_error": alpha_gradient_relative_error,
        "arnoldi_observed_reduced_radius": diagnostics.observed_reduced_radius,
        "arnoldi_reduced_nonnormality": diagnostics.max_reduced_nonnormality,
        "gpu": gpu,
        "validated_scope": "complex scalar |z|<=4 and Arnoldi reduced radius <=4",
    }
    summary_path = ROOT / "results" / "complex_arnoldi_summary.json"
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
