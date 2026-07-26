"""Boundary-condition generality checks for MLSL."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import (
    MLSLConfig,
    build_dirichlet_mlsl_1d,
    build_mixed_mlsl_1d,
    build_neumann_mlsl_1d,
    build_periodic_mlsl_1d,
)


RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_case(boundary: str, alpha_value: float, beta_value: float) -> dict[str, object]:
    if boundary == "dirichlet":
        x, layer = build_dirichlet_mlsl_1d(
            num_points=80,
            num_modes=16,
            config=MLSLConfig.stable(terms=120),
        )
        u0 = torch.sin(torch.pi * x) + 0.15 * torch.sin(3.0 * torch.pi * x)
    elif boundary == "neumann":
        x, layer = build_neumann_mlsl_1d(
            num_points=80,
            num_modes=16,
            config=MLSLConfig.stable(terms=120),
        )
        u0 = 0.4 + torch.cos(torch.pi * x) + 0.15 * torch.cos(3.0 * torch.pi * x)
    elif boundary == "periodic":
        x, layer = build_periodic_mlsl_1d(
            num_points=80,
            num_modes=17,
            config=MLSLConfig.stable(terms=120),
        )
        u0 = 0.4 + torch.cos(2.0 * torch.pi * x) + 0.15 * torch.sin(4.0 * torch.pi * x)
    elif boundary == "mixed_dn":
        x, layer = build_mixed_mlsl_1d(
            num_points=80,
            num_modes=16,
            boundary="dn",
            config=MLSLConfig.stable(terms=120),
        )
        u0 = layer.eigenvectors[:, 0].to(dtype=torch.float64) + 0.1 * layer.eigenvectors[:, 2].to(dtype=torch.float64)
    elif boundary == "mixed_nd":
        x, layer = build_mixed_mlsl_1d(
            num_points=80,
            num_modes=16,
            boundary="nd",
            config=MLSLConfig.stable(terms=120),
        )
        u0 = layer.eigenvectors[:, 0].to(dtype=torch.float64) + 0.1 * layer.eigenvectors[:, 2].to(dtype=torch.float64)
    else:
        raise ValueError(boundary)

    alpha = torch.tensor(alpha_value, requires_grad=True)
    beta = torch.tensor(beta_value, requires_grad=True)
    times = torch.linspace(0.0, 0.08, 8)
    out = layer(u0, times, alpha, beta=beta)
    loss = out.square().mean()
    loss.backward()

    constant_error = None
    if boundary in {"neumann", "periodic"}:
        const = torch.ones_like(x)
        const_out = layer(const, times, alpha.detach(), beta=beta.detach())
        constant_error = torch.max(torch.abs(const_out - const)).item()

    return {
        "boundary": boundary,
        "alpha": alpha_value,
        "beta": beta_value,
        "output_shape": str(tuple(out.shape)),
        "finite_output": bool(torch.isfinite(out).all().item()),
        "finite_alpha_grad": bool(torch.isfinite(alpha.grad).item()),
        "finite_beta_grad": bool(torch.isfinite(beta.grad).item()),
        "constant_mode_error": "" if constant_error is None else constant_error,
        "passed": bool(
            torch.isfinite(out).all().item()
            and torch.isfinite(alpha.grad).item()
            and torch.isfinite(beta.grad).item()
            and (constant_error is None or constant_error < 1e-10)
        ),
    }


def main() -> None:
    torch.set_default_dtype(torch.float64)
    rows = []
    for boundary in ["dirichlet", "neumann", "periodic", "mixed_dn", "mixed_nd"]:
        for alpha_value in [0.65, 1.25, 1.75]:
            for beta_value in [1.0, 1.5, 2.0]:
                rows.append(run_case(boundary, alpha_value, beta_value))

    write_csv(TABLES / "boundary_generality.csv", rows)
    summary = {
        "boundary_generality_total": len(rows),
        "boundary_generality_passed": sum(1 for r in rows if r["passed"]),
        "boundary_generality_pass_rate": sum(1 for r in rows if r["passed"]) / len(rows),
    }
    (RESULTS / "boundary_generality_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
