"""Plot matched work at the two final target-list densities."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT/"results/prediction_time_transport_20261020/run_02/result.json").read_text())
    plt.rcParams.update({"font.size": 10.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.8, 3.6), layout="constrained")
    labels = ["Prior physical pair", "Monotone time hulls", "Secant transport", "Curvature extension"]
    for offset, density, color in [(-.18, "17", "#89999e"), (.18, "65", "#166b5c")]:
        values = [r["lp_calls"] for r in result["summary"][density].values()]
        bars = ax.barh([i+offset for i in range(4)], values, .32,
                       color=color, label=f"{len(result['grids'][density])} candidate times")
        ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 9800)
    ax.set_xlabel("Attempted endpoint programs")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
    folder = ROOT/"paper/prediction_time_transport_figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder/"transport_work.pdf")
    fig.savefig(folder/"transport_work.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
