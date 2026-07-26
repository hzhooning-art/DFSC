"""Generate submission figures for adaptive control and real-task evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RESULTS = ROOT / "generated_results"
REVISION_RESULTS = ROOT / "revision_results"
FIGURES = WORKSPACE / "paper_assets" / "figures"


def save(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def adaptive_figure() -> None:
    payload = json.loads((RESULTS / "adaptive_krylov_calibration_summary.json").read_text(encoding="utf-8"))
    rows = payload["summary"]
    tolerances = np.asarray([row["requested_rtol"] for row in rows])
    dimensions = np.asarray([row["selected_dimension_mean"] for row in rows])
    errors = np.asarray([row["actual_error_mean"] for row in rows])
    max_errors = np.asarray([row["actual_error_max"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    axes[0].plot(tolerances, dimensions, "o-", color="#2c5f7c", lw=1.8)
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Requested relative tolerance")
    axes[0].set_ylabel("Mean selected Krylov dimension")
    axes[0].grid(alpha=0.25)
    axes[1].loglog(tolerances, errors, "o-", label="mean actual error", color="#b44632")
    axes[1].loglog(tolerances, max_errors, "s--", label="maximum actual error", color="#526b35")
    axes[1].loglog(tolerances, 10 * tolerances, ":", label=r"$10\times$ requested tolerance", color="#555555")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Requested relative tolerance")
    axes[1].set_ylabel("Relative action error")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_adaptive_krylov_calibration")


def real_task_figure() -> None:
    with (RESULTS / "real_brownian_two_population_summary.csv").open(encoding="utf-8") as handle:
        brownian = list(csv.DictReader(handle))
    with (RESULTS / "real_geomembrane_relaxation_summary.csv").open(encoding="utf-8") as handle:
        membrane = list(csv.DictReader(handle))
    hactin_path = ROOT / "results" / "tables" / "real_spt_model_comparison_summary.csv"
    with hactin_path.open(encoding="utf-8") as handle:
        hactin = list(csv.DictReader(handle))

    common_models = ["Stretched exponential", "Direct MLSL inverse", "Pure MLP", "MLSL + residual MLP"]
    display = ["Stretched exp.", "Direct MLSL", "Pure MLP", "MLSL + residual"]
    task_values = []
    task_names = ["H-actin", "Brownian slow", "Brownian fast", "Geomembrane"]
    for model in common_models:
        hrow = next(row for row in hactin if row["model"] == model)
        values = [float(hrow["late_lag_error_mean"])]
        for condition in ("Brownian slow population", "Brownian fast population"):
            brow = next(row for row in brownian if row["condition"] == condition and row["model"] == model)
            values.append(float(brow["late_error_mean"]))
        membrane_model = "Fractional Zener / MLSL" if model == "Direct MLSL inverse" else model
        mrow = next(row for row in membrane if row["model"] == membrane_model)
        values.append(float(mrow["extrapolation_error_mean"]))
        task_values.append(values)

    values = np.asarray(task_values)
    x = np.arange(len(task_names))
    width = 0.19
    colors = ("#526b35", "#2c5f7c", "#9a7651", "#b44632")
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for index, (label, color) in enumerate(zip(display, colors)):
        ax.bar(x + (index - 1.5) * width, values[index], width, label=label, color=color)
    ax.set_xticks(x, task_names)
    ax.set_ylabel("Held-out / extrapolation relative error")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_real_multitask_evidence")


def adaptive_gradient_figure() -> None:
    with (REVISION_RESULTS / "adaptive_gradient_scan.csv").open(encoding="utf-8") as handle:
        scan = list(csv.DictReader(handle))
    with (REVISION_RESULTS / "adaptive_inverse_stability.csv").open(encoding="utf-8") as handle:
        inverse = list(csv.DictReader(handle))

    alpha = np.asarray([float(row["alpha"]) for row in scan])
    adaptive_gradient = np.asarray([float(row["adaptive_gradient"]) for row in scan])
    reference_gradient = np.asarray([float(row["reference_gradient"]) for row in scan])
    selected_terms = np.asarray([int(row["selected_terms"]) for row in scan])
    switch = np.asarray([row["work_budget_switched"].lower() == "true" for row in scan])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    axes[0].plot(alpha, reference_gradient, color="#333333", lw=2.0, label="fixed 180-term reference")
    axes[0].plot(alpha, adaptive_gradient, "--", color="#2c5f7c", lw=1.5, label="adaptive gradient")
    axes[0].scatter(alpha[switch], adaptive_gradient[switch], color="#b44632", s=24, zorder=3, label="budget switch")
    budget_axis = axes[0].twinx()
    budget_axis.step(alpha, selected_terms, where="mid", color="#526b35", alpha=0.38)
    budget_axis.set_ylabel("Selected series terms", color="#526b35")
    axes[0].set_xlabel(r"Fractional order $\alpha$")
    axes[0].set_ylabel(r"$\partial \mathcal{J}/\partial \alpha$")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")

    colors = {1e-3: "#2c5f7c", 5e-3: "#526b35", 2e-2: "#b44632"}
    for learning_rate in (1e-3, 5e-3, 2e-2):
        for start in (0.56, 0.92):
            selected = [
                row
                for row in inverse
                if float(row["learning_rate"]) == learning_rate and float(row["start_alpha"]) == start
            ]
            axes[1].semilogy(
                [int(row["step"]) for row in selected],
                [float(row["loss"]) for row in selected],
                "-" if start < 0.77 else "--",
                color=colors[learning_rate],
                lw=1.5,
                label=rf"$\eta={learning_rate:g}$" if start < 0.77 else None,
            )
    axes[1].set_xlabel("Adam updates")
    axes[1].set_ylabel("Inverse objective")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_adaptive_gradient_stability")


if __name__ == "__main__":
    adaptive_figure()
    real_task_figure()
    adaptive_gradient_figure()
