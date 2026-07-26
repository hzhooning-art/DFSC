"""Generate manuscript figures for sample/OOD and heated-steam evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "revision_results"
FIGURES = ROOT.parent / "paper_assets" / "figures"


def main() -> None:
    sample = json.loads((RESULTS / "sample_ood_matrix.json").read_text(encoding="utf-8"))
    real = json.loads((RESULTS / "real_heated_steam.json").read_text(encoding="utf-8"))
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    colors = {"pure_mlp": "#4477AA", "dfsc_residual": "#228833"}
    labels = {"pure_mlp": "Pure MLP", "dfsc_residual": "DFSC + residual"}
    for model in ("pure_mlp", "dfsc_residual"):
        rows = [row for row in sample["summary"] if row["model"] == model]
        sizes = [row["train_samples"] for row in rows]
        for scenario, marker, style in (("iid", "o", "-"), ("joint_ood", "s", "--")):
            means = [row[f"{scenario}_mean"] for row in rows]
            stds = [row[f"{scenario}_std"] for row in rows]
            axes[0].errorbar(
                sizes,
                means,
                yerr=stds,
                color=colors[model],
                marker=marker,
                linestyle=style,
                capsize=2,
                label=f"{labels[model]}, {'IID' if scenario == 'iid' else 'joint OOD'}",
            )
    axes[0].set_yscale("log")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([16, 32, 64, 128], labels=["16", "32", "64", "128"])
    axes[0].set_xlabel("Training samples")
    axes[0].set_ylabel("Relative $L_2$ error")
    axes[0].set_title("(a) Controlled sample/OOD matrix")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)

    order = ("integer", "dfsc", "mlp", "hybrid")
    names = ("Integer", "DFSC", "Pure MLP", "DFSC + residual")
    rows = {row["model"]: row for row in real["summary"]}
    means = [rows[name]["heldout_rmse_k_mean"] for name in order]
    stds = [rows[name]["heldout_rmse_k_std"] for name in order]
    parameters = [rows[name]["parameters"] for name in order]
    bars = axes[1].bar(
        np.arange(4),
        means,
        yerr=stds,
        capsize=3,
        color=["#999999", "#CC6677", "#4477AA", "#228833"],
    )
    axes[1].set_xticks(np.arange(4), names, rotation=18, ha="right")
    axes[1].set_ylabel("Held-out RMSE (K)")
    axes[1].set_title("(b) Measured heat-transport OOD")
    axes[1].grid(True, axis="y", alpha=0.25)
    for bar, count in zip(bars, parameters, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.4,
            f"{count:,} par.",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axes[1].set_ylim(0, max(means) * 1.22)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_sample_ood_real_heat.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_sample_ood_real_heat.png", dpi=220, bbox_inches="tight")
    print(FIGURES / "fig_sample_ood_real_heat.pdf")


if __name__ == "__main__":
    main()
