"""Render the final matched search records as an embedded-font vector figure."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT/"results/prediction_time_search_20261020/run_05/result.json").read_text())
    summaries = list(result["summary"].values())
    labels = ["Fixed fine", "Progressive", "Add physical pair"]
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained")
    bars = axes[0].bar(labels, [r["lp_calls"] for r in summaries], color=["#89999e", "#297f9c", "#166b5c"])
    axes[0].bar_label(bars, padding=4)
    axes[0].set_ylim(0, 1850)
    axes[0].set_ylabel("Attempted endpoint programs")
    axes[0].set_title("Matched computational work")
    x = list(range(3))
    bars = axes[1].bar([v-.17 for v in x], [r["states"]["sufficient"] for r in summaries], .32, label="Sufficient", color="#166b5c")
    axes[1].bar_label(bars, padding=3)
    bars = axes[1].bar([v+.17 for v in x], [r["states"]["unresolved"] for r in summaries], .32, label="Unresolved", color="#c18c3a")
    axes[1].bar_label(bars, padding=3)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 170)
    axes[1].set_ylabel("Candidate requests")
    axes[1].set_title("Decisions on the same targets")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
    folder = ROOT/"paper/prediction_time_search_figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder/"search_work.pdf")
    fig.savefig(folder/"search_work.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
