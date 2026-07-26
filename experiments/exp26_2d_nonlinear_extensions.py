"""2D boundary and nonlinear-family extensions for MLSL."""

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
    build_mixed_mlsl_2d,
    build_neumann_mlsl_2d,
    build_periodic_mlsl_2d,
)


RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def two_d_boundary_extension() -> tuple[list[dict[str, object]], dict[str, object]]:
    builders = {
        "neumann_2d": lambda: build_neumann_mlsl_2d(
            num_points_1d=12,
            num_modes_1d=4,
            config=MLSLConfig.stable(terms=100),
        ),
        "periodic_2d": lambda: build_periodic_mlsl_2d(
            num_points_1d=12,
            num_modes_1d=5,
            config=MLSLConfig.stable(terms=100),
        ),
        "mixed_dn_2d": lambda: build_mixed_mlsl_2d(
            num_points_1d=12,
            num_modes_1d=4,
            boundary="dn",
            config=MLSLConfig.stable(terms=100),
        ),
        "mixed_nd_2d": lambda: build_mixed_mlsl_2d(
            num_points_1d=12,
            num_modes_1d=4,
            boundary="nd",
            config=MLSLConfig.stable(terms=100),
        ),
    }
    rows = []
    for name, build in builders.items():
        coords, layer = build()
        alpha = torch.tensor(0.9, requires_grad=True)
        beta = torch.tensor(1.4, requires_grad=True)
        u0 = layer.eigenvectors[:, 0].to(dtype=coords.dtype) + 0.1 * layer.eigenvectors[:, -1].to(dtype=coords.dtype)
        out = layer(u0, torch.linspace(0.0, 0.04, 5), alpha, beta=beta)
        out.square().mean().backward()
        rows.append(
            {
                "case": name,
                "num_points": coords.shape[0],
                "num_modes": layer.eigenvalues.numel(),
                "finite_output": bool(torch.isfinite(out).all().item()),
                "finite_alpha_grad": bool(torch.isfinite(alpha.grad).item()),
                "finite_beta_grad": bool(torch.isfinite(beta.grad).item()),
                "output_shape": str(tuple(out.shape)),
            }
        )
    return rows, {
        "2d_extended_boundary_pass_rate": sum(
            1 for r in rows if r["finite_output"] and r["finite_alpha_grad"] and r["finite_beta_grad"]
        )
        / len(rows)
    }


def nonlinear_reaction_correction() -> tuple[list[dict[str, object]], dict[str, object]]:
    """Use MLSL as a linear backbone and one explicit nonlinear correction.

    This is not a full nonlinear solver. It verifies that the primitive can be
    embedded as the history-free linear step inside a semilinear workflow.
    """

    x, layer = build_dirichlet_mlsl_1d(
        num_points=96,
        num_modes=18,
        config=MLSLConfig.stable(terms=120),
    )
    u0 = 0.75 * torch.sin(torch.pi * x) + 0.2 * torch.sin(2.0 * torch.pi * x)
    rows = []
    for gamma in [0.1, 0.3, 0.5]:
        alpha = torch.tensor(0.85, requires_grad=True)
        beta = torch.tensor(1.5, requires_grad=True)
        times = torch.linspace(0.0, 0.04, 6)
        linear = layer(u0, times, alpha, beta=beta)
        correction = -gamma * times[:, None] * linear.pow(3)
        hybrid = linear + correction
        loss = hybrid.square().mean()
        loss.backward()
        rows.append(
            {
                "family": "semilinear_cubic_reaction_backbone",
                "gamma": gamma,
                "finite_output": bool(torch.isfinite(hybrid).all().item()),
                "finite_alpha_grad": bool(torch.isfinite(alpha.grad).item()),
                "finite_beta_grad": bool(torch.isfinite(beta.grad).item()),
                "max_abs_state": float(torch.max(torch.abs(hybrid)).detach()),
            }
        )
    return rows, {
        "semilinear_backbone_pass_rate": sum(
            1 for r in rows if r["finite_output"] and r["finite_alpha_grad"] and r["finite_beta_grad"]
        )
        / len(rows)
    }


def main() -> None:
    torch.set_default_dtype(torch.float64)
    rows_2d, summary_2d = two_d_boundary_extension()
    rows_nl, summary_nl = nonlinear_reaction_correction()
    write_csv(TABLES / "two_d_boundary_extension.csv", rows_2d)
    write_csv(TABLES / "semilinear_backbone_extension.csv", rows_nl)
    summary = {**summary_2d, **summary_nl}
    (RESULTS / "two_d_nonlinear_extension_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
