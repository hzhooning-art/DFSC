"""Render publication figures for Stages 62--63 without changing results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def baseline_figure(data: dict) -> None:
    labels = ["Shared\nrank 3", "Independent\nNLS", "Fixed-grid\nNNLS", "Prony\nrecurrence"]
    keys = ["shared_rank3", "independent_nls_rank3", "fixed_grid_nnls", "prony_rank3"]
    medians = np.asarray([data["baselines"][key]["median_experiment_nrmse"] for key in keys])
    iqr = np.asarray([data["baselines"][key]["iqr_experiment_nrmse"] for key in keys])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(keys))
    ax.bar(x, medians, color=["#2f6690", "#6c757d", "#d99000", "#7a5195"], width=0.64)
    ax.errorbar(x, medians, yerr=np.vstack((medians - iqr[:, 0], iqr[:, 1] - medians)), fmt="none", color="black", capsize=4)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-batch NRMSE (experiment median)")
    ax.set_title("Common-split recovery prediction on 50 independent exposures")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "fig_stage63_external_baselines")


def separation_figure(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    ranks = [1, 2, 3]
    errors = [data["evaluation"]["rank_records"][str(rank)]["median_prediction_nrmse"] for rank in ranks]
    axes[0].plot(ranks, errors, marker="o", color="#2f6690", linewidth=2)
    axes[0].set_xticks(ranks)
    axes[0].set_xlabel("Candidate shared rank")
    axes[0].set_ylabel("Held-batch median NRMSE")
    axes[0].set_title("Predictive error decreases with rank")
    axes[0].grid(alpha=0.25)
    for rank in (2, 3):
        folds = data["evaluation"]["rank_records"][str(rank)]["folds"]
        for fold_index, fold in enumerate(folds):
            axes[1].scatter([rank] * rank, fold["rates"], s=28, alpha=0.65, color="#d99000" if rank == 2 else "#7a5195")
    axes[1].set_yscale("log")
    axes[1].set_xticks([2, 3])
    axes[1].set_xlabel("Candidate shared rank")
    axes[1].set_ylabel("Fold-specific decay rate")
    axes[1].set_title("Rate coalescence triggers refusal")
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    _save(fig, "fig_stage63_rank_refusal")


def boundary_figure(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    tail = data["tail_coverage_fixed_spacing"]
    axes[0].plot([row["horizon_seconds"] for row in tail], [row["rank_records"]["3"]["median_prediction_nrmse"] for row in tail], marker="o")
    axes[0].set_xlabel("Horizon (s), fixed spacing")
    axes[0].set_ylabel("Rank-3 held NRMSE")
    density = data["sampling_density_fixed_horizon"]
    axes[1].plot([row["samples_per_curve"] for row in density], [row["factor_diagnostics"]["ar1_effective_sample_size_proxy"] for row in density], marker="o", color="#d99000")
    axes[1].set_xlabel("Samples at 15 s")
    axes[1].set_ylabel("AR(1) effective-N proxy")
    starts = data["optimizer_start_budget"]
    axes[2].plot([row["starts"] for row in starts], [row["rank_records"]["3"]["median_prediction_nrmse"] for row in starts], marker="o", color="#7a5195")
    axes[2].set_xlabel("Optimizer starts")
    axes[2].set_ylabel("Rank-3 held NRMSE")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig_stage62_boundary_factorization")


def main() -> None:
    gas = json.loads((RESULTS / "public_uci_gas_recovery.json").read_text(encoding="utf-8"))
    factors = json.loads((RESULTS / "stage62_boundary_factor_audit.json").read_text(encoding="utf-8"))
    baseline_figure(gas)
    separation_figure(gas)
    boundary_figure(factors)


if __name__ == "__main__":
    main()


