"""Render publication figures for the Stage 62 direct public-data task."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from probe_public_pva_relaxation import DATA, HORIZONS, SAMPLE_BUDGETS, load_curves


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "public_pva_relaxation.json"
FIGURES = ROOT / "figures"
COLORS = {1: "#285a9f", 2: "#d88c00", 3: "#7a5195"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.1,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, label: str) -> None:
    ax.text(-0.11, 1.06, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")


def save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def statistics_figure(payload: dict) -> None:
    curves = load_curves(DATA)
    cell = payload["full_task"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)

    ax = axes[0, 0]
    for curve in curves:
        ax.plot(curve.time, curve.value, color=COLORS[curve.sample], lw=1.7, alpha=0.78,
                label=f"Specimen {curve.sample}" if curve.cycle == 1 else None)
    ax.set(xlabel="Relaxation time (s)", ylabel="Normalized force", title="Nine directly observed relaxation curves")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    panel_label(ax, "a")

    ax = axes[0, 1]
    x = np.arange(1, 10)
    for rank, marker in zip((1, 2, 3), ("o", "s", "D")):
        errors = np.asarray(cell["rank_records"][str(rank)]["curve_prediction_nrmse"])
        ax.plot(x, errors, marker=marker, lw=1.5, ms=6.5, color=COLORS[rank], label=f"Rank {rank}")
    ax.set(xlabel="Held-out curve", ylabel="Late-window NRMSE", title="Leave-one-specimen-out prediction")
    ax.set_xticks(x)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    panel_label(ax, "b")

    ax = axes[1, 0]
    ranks, centers, lows, highs = [], [], [], []
    for rank in (2, 3):
        row = payload["paired_statistics"][str(rank)]
        ranks.append(f"Rank {rank} vs 1")
        centers.append(row["median_relative_improvement"])
        lows.append(centers[-1] - row["bootstrap_95pct"][0])
        highs.append(row["bootstrap_95pct"][1] - centers[-1])
    y = np.arange(2)
    ax.errorbar(centers, y, xerr=np.vstack((lows, highs)), fmt="o", color="#1f6f8b",
                ecolor="#1f6f8b", elinewidth=2, capsize=5, ms=8)
    ax.axvline(0.0, color="#555555", ls="--", lw=1.2)
    ax.set_yticks(y, labels=ranks)
    ax.set(xlabel="Median relative NRMSE improvement (95% bootstrap CI)", title="Paired predictive improvement")
    ax.grid(axis="x", alpha=0.22)
    panel_label(ax, "c")

    ax = axes[1, 1]
    for rank in (1, 2, 3):
        fold_rates = np.asarray([row["rates"] for row in cell["rank_records"][str(rank)]["folds"]])
        for mode in range(rank):
            timescales = 1.0 / fold_rates[:, mode]
            xpos = rank + (mode - (rank - 1) / 2) * 0.16
            ax.scatter(np.full(3, xpos), timescales, s=48, color=COLORS[rank], alpha=0.8)
            ax.plot([xpos - 0.05, xpos + 0.05], [np.median(timescales)] * 2, color="black", lw=1.5)
    ax.set_yscale("log")
    ax.set_xticks((1, 2, 3), labels=("Rank 1", "Rank 2", "Rank 3"))
    ax.set(ylabel="Shared timescale (s, log scale)", title="Across-fold pole stability")
    ax.grid(axis="y", which="both", alpha=0.22)
    panel_label(ax, "d")

    save(fig, "fig_stage62_public_data_statistics")


def workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(15.4, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#20313f"
    muted = "#536574"
    fills = ("#e8f1f7", "#edf4ee", "#fff2d9", "#f5e9e7")

    banner = FancyBboxPatch((0.025, 0.925), 0.95, 0.055,
                            boxstyle="round,pad=0.005,rounding_size=0.012",
                            fc=ink, ec=ink, lw=1.2)
    ax.add_patch(banner)
    ax.text(0.5, 0.953, "ANALYSIS DESIGN IS FROZEN BEFORE MODEL COMPARISON",
            ha="center", va="center", color="white", fontsize=15.2, fontweight="bold")

    def card(rect, index, title, subtitle, face):
        x, y, w, h = rect
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.007,rounding_size=0.010",
                                   fc="white", ec=ink, lw=1.45))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.075), w, 0.075,
                                   boxstyle="round,pad=0.007,rounding_size=0.010",
                                   fc=face, ec=ink, lw=1.2))
        ax.add_patch(plt.Circle((x + 0.019, y + h - 0.027), 0.0135,
                                facecolor=ink, edgecolor="none", zorder=4))
        ax.text(x + 0.019, y + h - 0.027, str(index), ha="center", va="center",
                fontsize=8.8, fontweight="bold", color="white", zorder=5)
        title_size = 9.0 if len(title) > 19 else 10.0
        ax.text(x + w / 2 + 0.015, y + h - 0.027, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=ink)
        ax.text(x + w / 2, y + h - 0.057, subtitle, ha="center", va="center",
                fontsize=8.6, color=muted)

    top = [
        (0.025, 0.535, 0.215, 0.355),
        (0.265, 0.535, 0.215, 0.355),
        (0.505, 0.535, 0.215, 0.355),
        (0.745, 0.535, 0.230, 0.355),
    ]
    card(top[0], 1, "DECLARED OBSERVATIONS", "units, groups, window, sampling", fills[0])
    card(top[1], 2, "FROZEN PREPROCESSING", "fixed transform and residual treatment", fills[0])
    card(top[2], 3, "SHARED REALIZATIONS", "positive rates; ranks 1, 2, and 3", fills[1])
    card(top[3], 4, "GROUPED TRANSFER", "fit early observations; predict held evidence", fills[2])

    # Panel 1: schematic grouped relaxation curves; no quantitative claim is encoded.
    p = ax.inset_axes([0.048, 0.585, 0.169, 0.218])
    t = np.linspace(0, 1, 100)
    for offset, color, rate in ((0.18, "#315f97", 2.8), (0.03, "#d89000", 2.0), (-0.12, "#7f5a9b", 1.45)):
        for delta in (-0.025, 0.0, 0.025):
            p.plot(t, offset + 0.67 * np.exp(-(rate + delta) * t), color=color, lw=1.35, alpha=0.84)
    p.set(xlabel="declared time window", ylabel="response")
    p.set_xticks([]); p.set_yticks([]); p.grid(alpha=0.18)

    # Panel 2: the analysis window and grouping are fixed before fitting.
    p = ax.inset_axes([0.288, 0.585, 0.169, 0.218])
    raw = 0.18 + 0.72 * np.exp(-2.5 * t) + 0.025 * np.sin(32 * t)
    p.plot(t, raw, color="#9aa5ad", lw=1.1)
    p.axvspan(0.18, 0.82, color="#65a9c6", alpha=0.18)
    p.plot(t[(t >= 0.18) & (t <= 0.82)], raw[(t >= 0.18) & (t <= 0.82)], color="#1f6f8b", lw=2.0)
    p.axvline(0.18, color=ink, ls="--", lw=0.9); p.axvline(0.82, color=ink, ls="--", lw=0.9)
    p.set_xticks([]); p.set_yticks([]); p.grid(alpha=0.18)

    # Panel 3: candidate ranks share rates across units but retain unit-specific amplitudes.
    p = ax.inset_axes([0.528, 0.585, 0.169, 0.218])
    y0 = (0.84, 0.50, 0.16)
    for rank, yy in enumerate(y0, start=1):
        p.text(0.02, yy, f"r = {rank}", transform=p.transAxes, va="center", fontsize=9.0, fontweight="bold")
        for k in range(rank):
            xx = 0.43 + 0.19 * k
            p.scatter(xx, yy, s=62, color=COLORS[rank], edgecolor=ink, linewidth=0.7, transform=p.transAxes, zorder=3)
            p.plot([xx, xx], [yy - 0.12, yy + 0.12], color=COLORS[rank], lw=1.1, transform=p.transAxes)
        p.plot([0.34, 0.92], [yy, yy], color="#b8c1c8", lw=1.0, transform=p.transAxes, zorder=0)
    p.text(0.63, -0.06, r"shared decay-rate spectrum $\{\lambda_k\}$", transform=p.transAxes,
           ha="center", fontsize=8.8, color=ink)
    p.set_xlim(0, 1); p.set_ylim(0, 1); p.axis("off")

    # Panel 4: grouped held-unit transfer with an early/late split.
    p = ax.inset_axes([0.772, 0.585, 0.176, 0.218])
    split = 0.58
    truth = 0.15 + 0.76 * np.exp(-2.2 * t)
    pred = 0.14 + 0.77 * np.exp(-2.16 * t)
    p.axvspan(0, split, color="#dcefe2", alpha=0.75)
    p.axvspan(split, 1, color="#f4dedd", alpha=0.72)
    p.plot(t, truth, color="#404c55", lw=1.2, label="held unit")
    p.plot(t, pred, color="#b54e4e", lw=1.7, ls="--", label="prediction")
    p.axvline(split, color=ink, lw=0.9, ls=":")
    p.text(0.27, 0.87, "fit", transform=p.transAxes, ha="center", fontsize=9.0, fontweight="bold")
    p.text(0.79, 0.87, "predict", transform=p.transAxes, ha="center", fontsize=9.0, fontweight="bold")
    p.set_xticks([]); p.set_yticks([]); p.grid(alpha=0.18)
    p.legend(frameon=False, fontsize=7.6, loc="lower left")

    gate_rect = (0.025, 0.075, 0.615, 0.390)
    decision_rect = (0.670, 0.075, 0.305, 0.390)
    card(gate_rect, 5, "JOINT EVIDENCE GATE", "all criteria must support the same interpretation", fills[1])
    card(decision_rect, 6, "AUDITABLE DECISION", "claim the smallest supported rank or refuse", fills[3])

    gate_titles = ("INFORMATION", "TRANSFER", "STABILITY", "RESOLUTION")
    gate_lines = (r"$\Delta$BIC", "held-unit NRMSE", r"foldwise SD$(\log\lambda)$", r"adjacent $\lambda$ separation")
    gate_x = (0.045, 0.192, 0.339, 0.486)
    for i, (gx, title, line) in enumerate(zip(gate_x, gate_titles, gate_lines)):
        ax.add_patch(FancyBboxPatch((gx, 0.145), 0.126, 0.220,
                                   boxstyle="round,pad=0.006,rounding_size=0.008",
                                   fc="#f8fafb", ec="#73828d", lw=1.0))
        ax.text(gx + 0.063, 0.332, title, ha="center", va="center", fontsize=9.6,
                fontweight="bold", color=ink)
        ax.text(gx + 0.063, 0.296, line, ha="center", va="center", fontsize=8.5, color=muted)
        if i == 0:
            ax.bar(gx + 0.030, 0.040, width=0.019, bottom=0.196, color="#9fb6c9", transform=ax.transData)
            ax.bar(gx + 0.058, 0.062, width=0.019, bottom=0.196, color="#5f8cad", transform=ax.transData)
            ax.bar(gx + 0.086, 0.082, width=0.019, bottom=0.196, color="#285a9f", transform=ax.transData)
        elif i == 1:
            ax.plot([gx + 0.024, gx + 0.057, gx + 0.093], [0.270, 0.235, 0.207],
                    color="#d88c00", marker="o", ms=4.0, lw=1.4, transform=ax.transData)
        elif i == 2:
            for yy in (0.205, 0.229, 0.253):
                ax.scatter([gx + 0.038, gx + 0.064, gx + 0.090], [yy, yy + 0.004, yy - 0.003],
                           s=15, color="#7a5195", transform=ax.transData)
        else:
            ax.plot([gx + 0.025, gx + 0.102], [0.232, 0.232], color="#638a68", lw=1.4, transform=ax.transData)
            ax.scatter([gx + 0.034, gx + 0.093], [0.232, 0.232], s=35, color="#638a68", transform=ax.transData)
        ax.text(gx + 0.063, 0.169, "PASS / FAIL", ha="center", va="center",
                fontsize=8.2, fontweight="bold", color=ink)

    ax.text(0.3325, 0.111,
            "Predictive improvement alone is insufficient for a mechanism-level rank claim.",
            ha="center", va="center", fontsize=9.3, fontweight="bold", color=ink)

    # Final record: two scientifically distinct outcomes plus provenance.
    ax.add_patch(FancyBboxPatch((0.700, 0.286), 0.245, 0.065,
                               boxstyle="round,pad=0.005,rounding_size=0.008",
                               fc="#dcebdc", ec="#55765a", lw=1.1))
    ax.text(0.8225, 0.3185, r"SUPPORTED: smallest rank $r^*$", ha="center", va="center",
            fontsize=10.2, fontweight="bold", color="#294b31")
    ax.text(0.8225, 0.259, "OR", ha="center", va="center", fontsize=8.8, fontweight="bold", color=muted)
    ax.add_patch(FancyBboxPatch((0.700, 0.166), 0.245, 0.065,
                               boxstyle="round,pad=0.005,rounding_size=0.008",
                               fc="#f3dfdc", ec="#9b5b55", lw=1.1))
    ax.text(0.8225, 0.1985, "UNRESOLVED: explicit reason codes", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="#743b37")
    ax.text(0.8225, 0.126, "design  |  groups  |  thresholds  |  residuals  |  provenance",
            ha="center", va="center", fontsize=8.5, color=ink)

    arrow_opts = dict(arrowstyle="-|>", mutation_scale=16, color=ink, lw=1.55)
    for start, end in (((0.240, 0.710), (0.265, 0.710)),
                       ((0.480, 0.710), (0.505, 0.710)),
                       ((0.720, 0.710), (0.745, 0.710))):
        ax.add_patch(FancyArrowPatch(start, end, **arrow_opts))
    ax.plot([0.86, 0.86, 0.56], [0.535, 0.493, 0.493], color=ink, lw=1.55,
            solid_capstyle="round", zorder=5)
    ax.add_patch(FancyArrowPatch((0.56, 0.493), (0.56, 0.465), **arrow_opts))
    ax.add_patch(FancyArrowPatch((0.640, 0.270), (0.670, 0.270), **arrow_opts))

    ax.text(0.5, 0.022,
            "Observation design, candidate ranks, grouping, and thresholds remain fixed throughout evaluation.",
            ha="center", va="center", fontsize=10.1, fontweight="bold", color=ink)
    save(fig, "fig_stage62_method_workflow")


def boundary_figure(payload: dict) -> None:
    decisions = np.empty((len(SAMPLE_BUDGETS), len(HORIZONS)), dtype=int)
    nrmse = np.empty_like(decisions, dtype=float)
    code = {"INDETERMINATE": 0, "SUPPORTED_RANK_1": 1, "SUPPORTED_RANK_2": 2, "SUPPORTED_RANK_3": 3}
    for cell in payload["boundary"]:
        row = SAMPLE_BUDGETS.index(cell["samples_per_curve"])
        col = HORIZONS.index(cell["horizon_seconds"])
        decisions[row, col] = code[cell["decision"]]
        selected = int(cell["decision"].rsplit("_", 1)[-1]) if cell["decision"].startswith("SUPPORTED") else 1
        nrmse[row, col] = cell["rank_records"][str(selected)]["median_prediction_nrmse"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), constrained_layout=True)
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(("#d9d9d9", "#9ecae1", "#f4c36a", "#9b8ac4"))
    norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), cmap.N)
    ax = axes[0]
    image = ax.imshow(decisions, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    labels = {0: "REFUSE", 1: "R1", 2: "R2", 3: "R3"}
    for r in range(decisions.shape[0]):
        for c in range(decisions.shape[1]):
            ax.text(c, r, labels[decisions[r, c]], ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(HORIZONS)), labels=[f"{v:g}" for v in HORIZONS])
    ax.set_yticks(range(len(SAMPLE_BUDGETS)), labels=[str(v) for v in SAMPLE_BUDGETS])
    ax.set(xlabel="Observation horizon (s)", ylabel="Samples per curve", title="Evidence-controlled rank boundary")
    panel_label(ax, "a")
    cbar = fig.colorbar(image, ax=ax, ticks=(0, 1, 2, 3), shrink=0.82)
    cbar.ax.set_yticklabels(("Refuse", "Rank 1", "Rank 2", "Rank 3"))

    ax = axes[1]
    log_nrmse = np.log10(nrmse)
    heat = ax.imshow(log_nrmse, origin="lower", aspect="auto", cmap="viridis_r")
    for r in range(nrmse.shape[0]):
        for c in range(nrmse.shape[1]):
            ax.text(c, r, f"{nrmse[r, c]:.3f}", ha="center", va="center", fontsize=10.5,
                    color="white" if log_nrmse[r, c] > np.median(log_nrmse) else "black")
    ax.set_xticks(range(len(HORIZONS)), labels=[f"{v:g}" for v in HORIZONS])
    ax.set_yticks(range(len(SAMPLE_BUDGETS)), labels=[str(v) for v in SAMPLE_BUDGETS])
    ax.set(xlabel="Observation horizon (s)", ylabel="Samples per curve", title="Held-specimen late-window NRMSE")
    panel_label(ax, "b")
    cbar = fig.colorbar(heat, ax=ax, shrink=0.82)
    cbar.set_label("log10 NRMSE")
    save(fig, "fig_stage62_identifiability_boundary")


def main() -> None:
    style()
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    statistics_figure(payload)
    workflow_figure()
    boundary_figure(payload)
    print(f"wrote Stage 62 figures to {FIGURES}")


if __name__ == "__main__":
    main()
