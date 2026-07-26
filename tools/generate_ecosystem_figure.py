"""Generate the DFSC ecosystem architecture figure used by both manuscripts."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT = ROOT / "paper_assets" / "figures"


def box(ax, x, y, w, h, title, lines, face, edge="#263746"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.4))
    ax.text(x + 0.02, y + h - 0.06, title, fontsize=10.5, fontweight="bold", va="top")
    ax.text(x + 0.02, y + h - 0.14, "\n".join(lines), fontsize=8.3, va="top", linespacing=1.35)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.2, color="#3c4852"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 5.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.03, 0.62, 0.28, 0.28, "Differentiable numerical core",
        ["Mittag-Leffler evaluators", "MLSL / forced MLSL", "Lanczos, Arnoldi, sparse actions"],
        "#e8f1f7")
    box(ax, 0.36, 0.62, 0.28, 0.28, "Learning workflows",
        ["Trainable alpha and beta", "Inverse-order estimation", "Hybrid residual composition"],
        "#edf4ea")
    box(ax, 0.69, 0.62, 0.28, 0.28, "Application surfaces",
        ["Anomalous and graph diffusion", "Advection-diffusion / relaxation", "Experimental SPT workflow"],
        "#f8f0df")
    arrow(ax, (0.31, 0.76), (0.36, 0.76))
    arrow(ax, (0.64, 0.76), (0.69, 0.76))

    box(ax, 0.03, 0.16, 0.28, 0.27, "Problem-algorithm-solve",
        ["Structured problems and solutions", "Automatic algorithm selection", "Direct and history-aware paths"],
        "#f1eef6")
    box(ax, 0.36, 0.16, 0.28, 0.27, "Reliability contract",
        ["Validated-domain status", "Convergence and finite checks", "Warnings and error-estimate kind"],
        "#f6ecec")
    box(ax, 0.69, 0.16, 0.28, 0.27, "Research software layer",
        ["Single dfsc namespace", "Tests, examples, documentation", "Artifact and pre-release gates"],
        "#eceff1")
    arrow(ax, (0.31, 0.295), (0.36, 0.295))
    arrow(ax, (0.64, 0.295), (0.69, 0.295))

    arrow(ax, (0.17, 0.62), (0.17, 0.43))
    arrow(ax, (0.50, 0.62), (0.50, 0.43))
    arrow(ax, (0.83, 0.62), (0.83, 0.43))
    ax.text(0.5, 0.035, "PyTorch autograd  |  batching  |  CPU/GPU  |  reproducible diagnostics",
            ha="center", fontsize=9.5, color="#263746")

    fig.tight_layout(pad=0.4)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig_dfsc_ecosystem.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
