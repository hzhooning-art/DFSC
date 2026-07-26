"""Diagnose gradients when the adaptive series budget changes.

The adaptive controller differentiates the selected truncation, not the
discrete stopping decision.  This experiment scans alpha, records every work
budget change, and compares the resulting derivative with a fixed, richer
series.  It also tests inverse recovery over several learning rates.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dfsc


RESULTS = ROOT / "revision_results"
SCHEDULE = (8, 12, 18, 26, 38, 56, 84, 126)


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-14)


def objective(alpha: torch.Tensor, terms: int) -> torch.Tensor:
    z = -torch.linspace(0.0, 3.0, 49, dtype=alpha.dtype, device=alpha.device)
    weights = torch.linspace(0.75, 1.25, z.numel(), dtype=alpha.dtype, device=alpha.device)
    values = dfsc.mittag_leffler_e(alpha, z, terms=terms, method="series")
    return torch.mean(weights * values)


def scan_gradients() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    alphas = torch.linspace(0.52, 0.98, 185, dtype=torch.float64)
    previous_terms: int | None = None
    for alpha_value in alphas:
        alpha = alpha_value.clone().detach().requires_grad_(True)
        z = -torch.linspace(0.0, 3.0, 49, dtype=torch.float64)
        adaptive = dfsc.evaluate_mittag_leffler_adaptive(
            alpha,
            z,
            method="series",
            term_schedule=SCHEDULE,
            rtol=2e-10,
            atol=2e-12,
            strict=False,
        )
        adaptive_objective = torch.mean(
            torch.linspace(0.75, 1.25, z.numel(), dtype=z.dtype) * adaptive.values
        )
        adaptive_gradient = float(torch.autograd.grad(adaptive_objective, alpha)[0])

        alpha_reference = alpha_value.clone().detach().requires_grad_(True)
        reference_objective = objective(alpha_reference, terms=180)
        reference_gradient = float(torch.autograd.grad(reference_objective, alpha_reference)[0])
        switched = previous_terms is not None and adaptive.selected_terms != previous_terms
        rows.append(
            {
                "alpha": float(alpha_value),
                "selected_terms": adaptive.selected_terms,
                "converged": adaptive.converged,
                "work_budget_switched": switched,
                "adaptive_objective": float(adaptive_objective.detach()),
                "reference_objective": float(reference_objective.detach()),
                "adaptive_gradient": adaptive_gradient,
                "reference_gradient": reference_gradient,
                "objective_relative_error": relative_error(
                    float(adaptive_objective.detach()), float(reference_objective.detach())
                ),
                "gradient_relative_error": relative_error(adaptive_gradient, reference_gradient),
            }
        )
        previous_terms = adaptive.selected_terms

    switch_rows = [row for row in rows if bool(row["work_budget_switched"])]
    nonswitch_rows = [row for row in rows if not bool(row["work_budget_switched"])]
    summary = {
        "alpha_probes": len(rows),
        "work_budget_switches": len(switch_rows),
        "finite_gradient_rate": sum(math.isfinite(float(row["adaptive_gradient"])) for row in rows) / len(rows),
        "convergence_rate": sum(bool(row["converged"]) for row in rows) / len(rows),
        "max_objective_relative_error": max(float(row["objective_relative_error"]) for row in rows),
        "max_gradient_relative_error": max(float(row["gradient_relative_error"]) for row in rows),
        "max_gradient_relative_error_at_switch": max(
            (float(row["gradient_relative_error"]) for row in switch_rows), default=0.0
        ),
        "max_gradient_relative_error_away_from_switch": max(
            (float(row["gradient_relative_error"]) for row in nonswitch_rows), default=0.0
        ),
        "interpretation": (
            "The selected evaluator is differentiable inside a fixed-budget region. "
            "Finite gradients and agreement with a richer truncation are empirical diagnostics, "
            "not a claim of differentiability across every budget-selection boundary."
        ),
    }
    return rows, summary


def inverse_stability() -> tuple[list[dict[str, object]], dict[str, object]]:
    z = -torch.linspace(0.0, 3.0, 49, dtype=torch.float64)
    target_alpha = torch.tensor(0.77, dtype=torch.float64)
    target = dfsc.mittag_leffler_e(target_alpha, z, terms=180, method="series").detach()
    rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for learning_rate in (1e-3, 5e-3, 2e-2):
        for start in (0.56, 0.92):
            initial_fraction = (start - 0.45) / (1.05 - 0.45)
            raw = torch.tensor(
                math.log(initial_fraction / (1.0 - initial_fraction)),
                dtype=torch.float64,
                requires_grad=True,
            )
            optimizer = torch.optim.Adam([raw], lr=learning_rate)
            max_abs_gradient = 0.0
            selected_terms_seen: set[int] = set()
            for step in range(161):
                alpha = 0.45 + 0.60 * torch.sigmoid(raw)
                adaptive = dfsc.evaluate_mittag_leffler_adaptive(
                    alpha,
                    z,
                    method="series",
                    term_schedule=SCHEDULE,
                    rtol=2e-10,
                    atol=2e-12,
                    strict=False,
                )
                loss = torch.mean((adaptive.values - target) ** 2)
                selected_terms_seen.add(adaptive.selected_terms)
                if step < 160:
                    optimizer.zero_grad()
                    loss.backward()
                    max_abs_gradient = max(max_abs_gradient, abs(float(raw.grad)))
                    optimizer.step()
                if step % 10 == 0 or step == 160:
                    rows.append(
                        {
                            "learning_rate": learning_rate,
                            "start_alpha": start,
                            "step": step,
                            "alpha": float(alpha.detach()),
                            "loss": float(loss.detach()),
                            "selected_terms": adaptive.selected_terms,
                        }
                    )
            final_rows.append(
                {
                    "learning_rate": learning_rate,
                    "start_alpha": start,
                    "final_alpha": float(alpha.detach()),
                    "alpha_absolute_error": abs(float(alpha.detach()) - float(target_alpha)),
                    "final_loss": float(loss.detach()),
                    "max_abs_raw_gradient": max_abs_gradient,
                    "distinct_work_budgets": len(selected_terms_seen),
                }
            )

    summary = {
        "runs": len(final_rows),
        "learning_rates": [1e-3, 5e-3, 2e-2],
        "starts": [0.56, 0.92],
        "target_alpha": float(target_alpha),
        "successful_recoveries_at_1e-2": sum(
            float(row["alpha_absolute_error"]) <= 1e-2 for row in final_rows
        ),
        "max_final_alpha_absolute_error": max(float(row["alpha_absolute_error"]) for row in final_rows),
        "all_losses_finite": all(math.isfinite(float(row["final_loss"])) for row in final_rows),
        "runs_detail": final_rows,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    RESULTS.mkdir(parents=True, exist_ok=True)
    scan_rows, scan_summary = scan_gradients()
    inverse_rows, inverse_summary = inverse_stability()
    write_csv(RESULTS / "adaptive_gradient_scan.csv", scan_rows)
    write_csv(RESULTS / "adaptive_inverse_stability.csv", inverse_rows)
    payload = {"gradient_scan": scan_summary, "inverse_stability": inverse_summary}
    (RESULTS / "adaptive_gradient_stability_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
