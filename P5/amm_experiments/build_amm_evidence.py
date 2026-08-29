from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]

# The same script runs both in the project tree and in the submission supplement.
if (SCRIPT_DIR / "public_pva_relaxation.json").exists():
    RESULTS = SCRIPT_DIR
    OUT = SCRIPT_DIR / "generated_figures"
    SUMMARY = SCRIPT_DIR
else:
    RESULTS = ROOT / "results"
    OUT = ROOT / "paper_amm" / "figures"
    SUMMARY = ROOT / "amm_results"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY.mkdir(parents=True, exist_ok=True)

COLORS = {"blue": "#35609C", "orange": "#E69F00", "purple": "#7A5195", "gray": "#73777B"}


def load(name: str):
    with (RESULTS / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def save(fig, stem: str):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)


def pva_design_map(pva):
    horizons = sorted({float(x["horizon_seconds"]) for x in pva["boundary"]})
    samples = sorted({int(x["samples_per_curve"]) for x in pva["boundary"]})
    code = {"INDETERMINATE": 0, "SUPPORTED_RANK_1": 1, "SUPPORTED_RANK_2": 2, "SUPPORTED_RANK_3": 3}
    matrix = np.zeros((len(horizons), len(samples)), dtype=int)
    labels = np.empty_like(matrix, dtype=object)
    for row in pva["boundary"]:
        i = horizons.index(float(row["horizon_seconds"]))
        j = samples.index(int(row["samples_per_curve"]))
        matrix[i, j] = code[row["decision"]]
        labels[i, j] = {0: "Unresolved", 1: "r=1", 2: "r=2", 3: "r=3"}[matrix[i, j]]

    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cmap = ListedColormap(["#E8E8E8", "#B9D8C2", "#82B6D9", "#35609C"])
    image = ax.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    for i in range(len(horizons)):
        for j in range(len(samples)):
            color = "white" if matrix[i, j] == 3 else "#20252A"
            ax.text(j, i, labels[i, j], ha="center", va="center", color=color, fontweight="bold")
    ax.set_xticks(range(len(samples)), samples)
    ax.set_yticks(range(len(horizons)), [f"{h:g}" for h in horizons])
    ax.set_xlabel("Retained samples per relaxation curve")
    ax.set_ylabel("Observation horizon (s)")
    ax.set_title("PVA identifiability boundary under four selection criteria")
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3], pad=0.03)
    cbar.ax.set_yticklabels(["Unresolved", "Rank 1", "Rank 2", "Rank 3"])
    fig.tight_layout()
    save(fig, "fig_amm_pva_design_map")
    return {"horizons": horizons, "samples": samples, "decision_matrix": matrix.tolist()}


def baseline_figure(baselines):
    cases = baselines["summary"]["by_case"]
    labels = ["True rank 1", "True rank 2", "True rank 3"]
    methods = [
        ("positive_real_memory", "Shared finite memory", COLORS["blue"]),
        ("regularized_damped_modal", "Damped modal", COLORS["orange"]),
        ("trajectory_mlp", "Trajectory MLP", COLORS["purple"]),
    ]
    x = np.arange(len(cases))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for offset, (key, name, color) in enumerate(methods):
        values = [float(case["methods"][key]["extrapolation_rmse_to_clean"]) for case in cases]
        ax.bar(x + (offset - 1) * width, values, width, label=name, color=color, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-horizon RMSE to clean response")
    ax.set_title("Direct extrapolation baselines on matched sparse observations", pad=46)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save(fig, "fig_amm_direct_baselines")
    return {
        case["case"]: {key: case["methods"][key]["extrapolation_rmse_to_clean"] for key, _, _ in methods}
        for case in cases
    }


def material_evidence(pva, copper):
    pva_records = pva["full_task"]["rank_records"]
    copper_records = copper["evaluation"]["rank_records"]
    ranks = np.array([1, 2, 3])
    pva_nrmse = np.array([pva_records[str(r)]["median_prediction_nrmse"] for r in ranks])
    copper_nrmse = np.array([copper_records[str(r)]["median_prediction_nrmse"] for r in ranks])
    copper_iid = np.array([copper_records[str(r)]["mean_bic"] for r in ranks])
    copper_ar1 = np.array([copper_records[str(r)]["mean_ar1_bic"] for r in ranks])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].plot(ranks, pva_nrmse, "o-", color=COLORS["blue"], linewidth=2.2, label="PVA: held specimen")
    axes[0].plot(ranks, copper_nrmse, "s-", color=COLORS["orange"], linewidth=2.2, label="Copper: held group")
    axes[0].set_yscale("log")
    axes[0].set_xticks(ranks)
    axes[0].set_xlabel("Candidate shared rank")
    axes[0].set_ylabel("Median held-unit NRMSE")
    axes[0].set_title("(a) Transfer evidence")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25, which="both")

    axes[1].plot(ranks, copper_iid - copper_iid.min(), "o-", color=COLORS["purple"], linewidth=2.2, label="IID BIC")
    axes[1].plot(ranks, copper_ar1 - copper_ar1.min(), "s-", color=COLORS["blue"], linewidth=2.2, label="AR(1)-profile BIC")
    axes[1].axhline(10, color=COLORS["gray"], linestyle="--", linewidth=1.2, label="10-unit reference")
    axes[1].set_xticks(ranks)
    axes[1].set_xlabel("Candidate shared rank")
    axes[1].set_ylabel(r"BIC above within-score minimum")
    axes[1].set_title("(b) Copper residual-model sensitivity")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig_amm_material_evidence")
    return {
        "pva_held_nrmse": pva_nrmse.tolist(),
        "copper_held_nrmse": copper_nrmse.tolist(),
        "copper_iid_bic_delta": (copper_iid - copper_iid.min()).tolist(),
        "copper_ar1_bic_delta": (copper_ar1 - copper_ar1.min()).tolist(),
    }


def main():
    style()
    pva = load("public_pva_relaxation.json")
    copper = load("public_kupferdigital_relaxation.json")
    baselines = load("mechanism_vs_trajectory_baselines.json")
    summary = {
        "source_results": [
            "public_pva_relaxation.json",
            "public_kupferdigital_relaxation.json",
            "mechanism_vs_trajectory_baselines.json",
        ],
        "pva_design": pva_design_map(pva),
        "direct_baselines": baseline_figure(baselines),
        "material_evidence": material_evidence(pva, copper),
        "claim_boundary": (
            "Ranks are empirical shared finite realizations on declared observation designs; "
            "they are not unique microscopic mechanisms."
        ),
    }
    with (SUMMARY / "amm_evidence_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
