"""Switching-rule ablation for the Mittag-Leffler evaluator.

This compact diagnostic compares the current smooth transition band with a hard
series/asymptotic switch near the branch threshold. It is not a benchmark of
global Mittag-Leffler algorithms; it quantifies whether the transition used by
MLSL reduces local value and gradient jumps in the spectral regime tested in the
paper.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc.mittag_leffler import (
    _mittag_leffler_negative_asymptotic,
    _mittag_leffler_series_terms,
    mittag_leffler_e,
)

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def hard_switch(alpha: torch.Tensor, z: torch.Tensor, *, threshold: float, terms: int) -> torch.Tensor:
    if torch.abs(z) <= threshold:
        return _mittag_leffler_series_terms(alpha, z, terms=terms)
    return _mittag_leffler_negative_asymptotic(alpha, z, terms=8)


def eval_with_grad(
    alpha_value: float,
    z_abs: torch.Tensor,
    *,
    method: str,
    threshold: float,
    terms: int,
) -> tuple[float, float, bool, bool]:
    alpha = torch.tensor(alpha_value, dtype=torch.float64, requires_grad=True)
    z = -z_abs.clone().detach().to(dtype=torch.float64)
    if method == "smooth":
        value = mittag_leffler_e(alpha, z, terms=terms, method="hybrid")
    elif method == "hard":
        value = hard_switch(alpha, z, threshold=threshold, terms=terms)
    else:
        raise ValueError(method)
    value.backward()
    grad = alpha.grad.detach()
    return (
        float(value.detach()),
        float(grad),
        bool(torch.isfinite(value.detach()).item()),
        bool(torch.isfinite(grad).item()),
    )


def main() -> None:
    torch.set_default_dtype(torch.float64)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    # Match the threshold logic in mittag_leffler_e_hybrid for these alphas.
    cases = [(0.55, 4.0), (0.65, 8.0), (1.25, 8.0)]
    for alpha_value, threshold in cases:
        z_grid = torch.linspace(threshold * 0.75, threshold * 1.25, 41)
        for method in ["hard", "smooth"]:
            method_rows: list[dict[str, Any]] = []
            for z_abs in z_grid:
                value, grad_alpha, finite_value, finite_grad = eval_with_grad(
                    alpha_value,
                    z_abs,
                    method=method,
                    threshold=threshold,
                    terms=160,
                )
                row = {
                    "method": method,
                    "alpha": alpha_value,
                    "threshold": threshold,
                    "z": -float(z_abs),
                    "value": value,
                    "grad_alpha": grad_alpha,
                    "finite_value": finite_value,
                    "finite_grad": finite_grad,
                }
                rows.append(row)
                method_rows.append(row)

            value_jumps = [
                abs(float(right["value"]) - float(left["value"]))
                for left, right in zip(method_rows, method_rows[1:])
            ]
            grad_jumps = [
                abs(float(right["grad_alpha"]) - float(left["grad_alpha"]))
                for left, right in zip(method_rows, method_rows[1:])
            ]
            summary_rows.append(
                {
                    "method": method,
                    "alpha": alpha_value,
                    "threshold": threshold,
                    "cases": len(method_rows),
                    "finite_value_rate": sum(1 for row in method_rows if row["finite_value"]) / len(method_rows),
                    "finite_grad_rate": sum(1 for row in method_rows if row["finite_grad"]) / len(method_rows),
                    "max_adjacent_value_jump": max(value_jumps),
                    "mean_adjacent_value_jump": mean(value_jumps),
                    "max_adjacent_grad_jump": max(grad_jumps),
                    "mean_adjacent_grad_jump": mean(grad_jumps),
                }
            )

    hard_max_grad = max(
        row["max_adjacent_grad_jump"] for row in summary_rows if row["method"] == "hard"
    )
    smooth_max_grad = max(
        row["max_adjacent_grad_jump"] for row in summary_rows if row["method"] == "smooth"
    )
    hard_max_value = max(
        row["max_adjacent_value_jump"] for row in summary_rows if row["method"] == "hard"
    )
    smooth_max_value = max(
        row["max_adjacent_value_jump"] for row in summary_rows if row["method"] == "smooth"
    )
    summary = {
        "switch_ablation_cases": len(rows),
        "hard_max_adjacent_value_jump": hard_max_value,
        "smooth_max_adjacent_value_jump": smooth_max_value,
        "value_jump_reduction_factor": hard_max_value / max(smooth_max_value, 1e-14),
        "hard_max_adjacent_grad_jump": hard_max_grad,
        "smooth_max_adjacent_grad_jump": smooth_max_grad,
        "grad_jump_reduction_factor": hard_max_grad / max(smooth_max_grad, 1e-14),
    }

    write_csv(TABLES / "evaluator_switch_ablation_raw.csv", rows)
    write_csv(TABLES / "evaluator_switch_ablation_summary.csv", summary_rows)
    (RESULTS / "evaluator_switch_ablation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
