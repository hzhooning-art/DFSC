"""Generate manuscript figures from the auditable paper-data bundle."""

from __future__ import annotations

import json
import matplotlib
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "paper_data.json").read_text(encoding="utf-8"))
REAL_DATA = json.loads((HERE.parent / "results" / "p4_real_data_evidence.json").read_text(encoding="utf-8"))
SIMULATION = json.loads((HERE.parent / "results" / "p4_engineering_simulation_benchmark.json").read_text(encoding="utf-8"))
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 8.2,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig: plt.Figure, name: str, bottom: float = 0.16) -> None:
    # Reserve a dedicated bottom band only when a figure has an external legend.
    fig.tight_layout(rect=[0.0, bottom, 1.0, 0.98], pad=0.45)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_panels(axes) -> None:
    """Place panel identifiers outside the upper-left axes corners."""
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", np.ravel(axes)):
        ax.text(0.0, 1.025, label, transform=ax.transAxes,
                va="bottom", ha="left", fontsize=9, weight="bold",
                clip_on=False)


def enlarge_text(fig: plt.Figure, factor: float = 2.0) -> None:
    """Scale all figure text while preserving the plotted data geometry."""
    for text in fig.findobj(match=matplotlib.text.Text):
        text.set_fontsize(text.get_fontsize() * factor)


def protocol_figure() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.10, 5.55)
    ax.axis("off")
    groups = [
        (0.20, 3.62, 6.35, 1.80, "INPUT\nCONTRACT", "domain\nparameters\nreference", "#e7f0f7"),
        (7.45, 3.62, 6.35, 1.80, "NUMERICAL\nCHECKS", "value\ngradient\nfinite outputs", "#e8f3ec"),
        (7.45, 1.62, 6.35, 1.80, "LEARNING\nCHECKS", "calibration\nmodule reuse\nOOD", "#fff3dc"),
        (0.20, 1.62, 6.35, 1.80, "RELIABILITY\nRECORD", "long horizon\nperformance\nscope limits", "#f1e8f5"),
    ]
    for x, y, w, h, title, body, color in groups:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#263746", linewidth=0.9))
        ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=10.4,
                weight="bold", linespacing=0.88)
        ax.text(x + w / 2, y + 0.62, body, ha="center", va="center", fontsize=10.6, linespacing=0.96)
    arrow = dict(arrowstyle="-|>", lw=1.15, color="#263746")
    ax.annotate("", xy=(7.32, 4.52), xytext=(6.68, 4.52), arrowprops=arrow)
    ax.annotate("", xy=(10.625, 3.47), xytext=(10.625, 3.57), arrowprops=arrow)
    ax.annotate("", xy=(6.68, 2.52), xytext=(7.32, 2.52), arrowprops=arrow)
    ax.text(0.20, 1.22, "Every gate is evaluated on the declared domain", fontsize=10.0, weight="bold")
    ax.plot([0.20, 13.80], [0.98, 0.98], color="#6b7785", lw=0.8)
    ax.text(0.20, 0.43, "backend  |  dtype  |  device  |  batch  |  seed  |  reference  |  scope limit",
            fontsize=9.8, color="#263746")
    ax.text(0.20, -0.04, "CONFORMANT / NONCONFORMANT  +  explicit scope fields", fontsize=9.5, color="#555555")
    enlarge_text(fig)
    save(fig, "fig_protocol.pdf", bottom=0.02)


def common_ood_figure() -> None:
    rows = DATA["common_module_ood"]["results"]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.4), sharey=True)
    colors = {"primitive": "#2f5597", "baseline": "#e69f00"}
    panel_titles = {
        "matrix": "Matrix-exponential backend",
        "logistic": "Logistic RK4 backend",
    }
    for ax, backend in zip(axes, ("matrix", "logistic")):
        keys = sorted(rows)
        x = np.arange(len(keys))
        p_mean, p_std, b_mean, b_std = [], [], [], []
        for key in keys:
            item = rows[key][backend]["summary"]
            p_mean.append(item["primitive_ood_rmse"]["mean"])
            p_std.append(item["primitive_ood_rmse"]["std"])
            b_mean.append(item["baseline_ood_rmse"]["mean"])
            b_std.append(item["baseline_ood_rmse"]["std"])
        width = 0.34
        ax.bar(x - width / 2, p_mean, width, yerr=p_std, capsize=2.5,
               label="Primitive", color=colors["primitive"], edgecolor="#1f3a5f", linewidth=0.45)
        ax.bar(x + width / 2, b_mean, width, yerr=b_std, capsize=2.5,
               label="Pure MLP", color=colors["baseline"], edgecolor="#8a5700", linewidth=0.45,
               hatch="//")
        labels = []
        for key in keys:
            if "noise" in key and key.startswith("t"):
                time_value = key.split("_noise", 1)[0][1:]
                noise_value = key.split("noise", 1)[1]
                labels.append(f"$t={time_value}$\n$\\sigma={noise_value}$")
            else:
                labels.append(key)
        ax.set_xticks(x, labels)
        ax.grid(axis="y", color="#d9dfe3", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(panel_titles[backend], loc="center", pad=8)
    fig.supylabel("OOD RMSE (mean $\\pm$ std; 3 seeds)\nLower is better", x=0.025)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), borderaxespad=0.0)
    label_panels(axes)
    enlarge_text(fig)
    save(fig, "fig_common_ood.pdf")


def profile_figure() -> None:
    rows = DATA["profile"]["rows"]
    backends = sorted({r["backend"] for r in rows})
    batches = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.6), sharex=True)
    palette = {"matrix_exponential_action": "#2f5597", "rk4_linear_ode_step": "#e69f00", "logistic_rk4_step": "#7a5195"}
    for backend in backends:
        values = [next(r["samples_per_second"] for r in rows if r["backend"] == backend and r["batch"] == b) for b in batches]
        latencies = [next(r["mean_latency_ms"] for r in rows if r["backend"] == backend and r["batch"] == b) for b in batches]
        label = backend.replace("_", " ").replace("matrix exponential action", "matrix exp action")
        style = dict(marker="o", linewidth=1.4, markersize=3.8, color=palette.get(backend, "#333333"), label=label)
        axes[0].plot(batches, values, **style)
        axes[1].plot(batches, latencies, **style)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size (log$_2$ scale)")
        ax.grid(True, which="major", color="#d9dfe3", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Throughput (samples/s; log scale)")
    axes[1].set_ylabel("Mean latency (ms)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=1, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), borderaxespad=0.0)
    label_panels(axes)
    enlarge_text(fig)
    save(fig, "fig_gpu_profile.pdf", bottom=0.25)


def precision_tradeoff_figure() -> None:
    """Report dtype-specific accuracy and resource trade-offs."""
    rows = DATA["precision_tradeoff"]["rows"]
    primitives = ["matrix_exponential_action", "periodic_heat_2d_spectral"]
    labels = ["Matrix action", "2D heat"]
    colors = {"float32": "#d67a2e", "float64": "#2f5597"}
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.4))

    x = np.arange(len(primitives))
    offsets = {"float32": -0.10, "float64": 0.10}
    for dtype in ("float32", "float64"):
        selected = [next(row for row in rows if row["primitive"] == primitive and row["dtype"] == dtype)
                    for primitive in primitives]
        value = [row["max_abs_value_error"] for row in selected]
        gradient = [row.get("directional_gradient_relative_error", row.get("energy_gradient_relative_error"))
                    for row in selected]
        axes[0].scatter(x + offsets[dtype], value, marker="o", s=34, color=colors[dtype],
                        edgecolor="white", linewidth=0.5, label=f"{dtype}: value", zorder=3)
        axes[0].scatter(x + offsets[dtype], gradient, marker="D", s=28, facecolor="white",
                        edgecolor=colors[dtype], linewidth=1.2, label=f"{dtype}: gradient", zorder=3)

    float64 = {row["primitive"]: row for row in rows if row["dtype"] == "float64"}
    float32 = {row["primitive"]: row for row in rows if row["dtype"] == "float32"}
    latency_ratio = [float32[p]["mean_forward_ms"] / float64[p]["mean_forward_ms"] for p in primitives]
    memory_ratio = [float32[p]["peak_incremental_cuda_bytes"] / float64[p]["peak_incremental_cuda_bytes"]
                    for p in primitives]
    width = 0.34
    axes[1].bar(x - width / 2, latency_ratio, width=width, color="#176b87",
                edgecolor="#263746", linewidth=0.45, label="Latency ratio")
    axes[1].bar(x + width / 2, memory_ratio, width=width, color="#b5a76b", hatch="//",
                edgecolor="#5b5130", linewidth=0.45, label="Memory ratio")
    axes[1].axhline(1.0, color="#666666", linewidth=0.9, linestyle="--")

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Error against independent reference")
    axes[0].set_xticks(x, labels)
    axes[1].set_ylabel("Float32 / float64 resource ratio")
    axes[1].set_xticks(x, labels)
    for ax in axes:
        ax.grid(axis="y", which="both", color="#d9dfe3", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    fig.legend(handles0 + handles1, labels0 + labels1, frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.015), borderaxespad=0.0)
    label_panels(axes)
    enlarge_text(fig)
    save(fig, "fig_precision_tradeoff.pdf", bottom=0.21)


def real_data_figure() -> None:
    fewshot = next(item for item in REAL_DATA["datasets"] if "few-shot" in item["name"])
    steam = next(item for item in REAL_DATA["datasets"] if "steam" in item["name"])
    fractions = sorted({row["train_fraction"] for row in fewshot["models"]})
    def lookup(frac, model):
        return next(row for row in fewshot["models"]
                    if row["train_fraction"] == frac and row["model"] == model)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.6))
    palette = {"DFSC + residual MLP": "#2f5597", "Pure MLP": "#999999", "DFSC": "#e69f00"}
    for model in ("DFSC + residual MLP", "Pure MLP", "DFSC"):
        means = [lookup(frac, model)["cycle2_error_mean"] for frac in fractions]
        stds = [lookup(frac, model)["cycle2_error_std"] for frac in fractions]
        axes[0].errorbar(np.array(fractions) * 100, means, yerr=stds, marker="o", markersize=3.8,
                         linewidth=1.35, capsize=2.5, label=model, color=palette[model])
    axes[0].set_xlabel("Uniform first-cycle coverage (%)")
    axes[0].set_ylabel("Cycle-2 relative error\nmean $\\pm$ std; 3 seeds")
    axes[0].set_xticks([20, 40, 60, 100])
    axes[0].grid(True, color="#d9dfe3", linewidth=0.55)

    names = ["Integer", "DFSC", "Hybrid", "Pure MLP"]
    rows = {row["model"].lower(): row for row in steam["models"]}
    key_map = {"Integer": "integer", "DFSC": "dfsc", "Hybrid": "hybrid", "Pure MLP": "mlp"}
    means = [rows[key_map[name]]["heldout_rmse_k_mean"] for name in names]
    stds = [rows[key_map[name]]["heldout_rmse_k_std"] for name in names]
    colors = ["#bcbcbc", "#d67a2e", "#176b87", "#777777"]
    axes[1].bar(np.arange(len(names)), means, yerr=stds, capsize=2.5, color=colors,
                edgecolor="#263746", linewidth=0.45, hatch=["", "", "", "//"])
    axes[1].set_ylabel("RMSE (K; mean $\\pm$ std)")
    axes[1].set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
    axes[1].grid(axis="y", color="#d9dfe3", linewidth=0.55)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), borderaxespad=0.0)
    label_panels(axes)
    enlarge_text(fig)
    save(fig, "fig_real_data_evidence.pdf", bottom=0.18)


def simulation_figure() -> None:
    """Plot trajectory error growth for the engineering-simulation test."""
    rows = SIMULATION["rows"]
    horizons = SIMULATION["horizons"]
    steps = [str(value) for value in SIMULATION["rk4_steps"]]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.4), sharey=True)
    colors = {"0.2": "#cc6677", "0.1": "#e69f00", "0.05": "#2f5597"}
    for step in steps:
        means, stds = [], []
        for horizon in horizons:
            values = [row["rk4"][step] for row in rows if row["horizon"] == horizon]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)))
        axes[0].errorbar(horizons, means, yerr=stds, marker="o", markersize=3.5,
                         linewidth=1.3, capsize=2.2, color=colors[step], label=f"RK4, h={step}")
    matrix_by_horizon = [max(row["matrix_action_abs_error"] for row in rows if row["horizon"] == horizon) for horizon in horizons]
    axes[0].plot(horizons, matrix_by_horizon, "--", color="#555555", linewidth=1.1, label="Matrix action")
    axes[0].set_xlabel("Final time horizon")
    axes[0].set_ylabel("Max absolute error (mean $\\pm$ std)")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", color="#d9dfe3", linewidth=0.55)

    long_values = []
    for step in steps:
        values = [row["rk4"][step] for row in rows if row["horizon"] == max(horizons)]
        long_values.append((float(np.mean(values)), float(np.std(values, ddof=1))))
    means, stds = zip(*long_values)
    axes[1].errorbar([float(step) for step in steps], means, yerr=stds, marker="o", color="#176b87",
                     linewidth=1.4, capsize=2.5)
    axes[1].set_xlabel("RK4 step size $h$")
    axes[1].set_ylabel("Max absolute error (log scale)")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", color="#d9dfe3", linewidth=0.55)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), borderaxespad=0.0)
    label_panels(axes)
    enlarge_text(fig)
    save(fig, "fig_engineering_simulation.pdf", bottom=0.20)


def reliability_signature_figure() -> None:
    """Show backend-specific error distributions without pooling rankings."""
    records = DATA["multi_seed"]
    names = ["Matrix action", "Linear RK4", "Logistic RK4"]
    keys = ["matrix_exponential_action", "rk4_linear_ode_step", "logistic_rk4_step"]
    value_fields = ["long_horizon_max_abs_error", "ood_max_abs_error", "ood_max_abs_error"]
    colors = ["#2f5597", "#e69f00", "#7a5195"]
    metrics = [
        ("Value error", lambda row, field: row[field]),
        ("Directional-gradient error", lambda row, field: row["gradient_directional_relative_error"]),
        ("Calibration parameter error", lambda row, field: row["calibration"]["parameter_l1_error"]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.6))
    rng = np.random.default_rng(20260814)
    for ax, (ylabel, getter) in zip(axes, metrics):
        for index, (name, key, field, color) in enumerate(zip(names, keys, value_fields, colors)):
            values = np.asarray([getter(row, field) for row in records[key]["rows"]], dtype=float)
            jitter = rng.uniform(-0.065, 0.065, size=len(values))
            ax.scatter(np.full(len(values), index) + jitter, values, s=20, color=color,
                       edgecolor="white", linewidth=0.45, alpha=0.9, zorder=3)
            median = float(np.median(values))
            ax.plot([index - 0.16, index + 0.16], [median, median], color="#202020",
                    linewidth=1.35, zorder=4)
        ax.set_yscale("log")
        ax.set_xticks(range(len(names)), names)
        ax.set_ylabel(ylabel + " (log scale)", fontsize=9.2)
        ax.tick_params(labelsize=8.3)
        ax.grid(axis="y", which="both", color="#d9dfe3", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    label_panels(axes)
    save(fig, "fig_reliability_signatures.pdf", bottom=0.04)


def calibration_reproducibility_figure() -> None:
    """Visualize paired optimizer outcomes and recovered parameters by seed."""
    data = DATA["calibration_baselines"]
    rows = data["per_seed"]
    seeds = np.asarray([row["seed"] for row in rows])
    adam_error = np.asarray([row["adam"]["parameter_l2_error"] for row in rows])
    lbfgs_error = np.asarray([row["lbfgs_b"]["parameter_l2_error"] for row in rows])
    adam_theta = np.asarray([row["adam"]["theta"] for row in rows])
    lbfgs_theta = np.asarray([row["lbfgs_b"]["theta"] for row in rows])
    truth = np.asarray(data["truth"])

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 9.0))
    for seed, left, right in zip(seeds, adam_error, lbfgs_error):
        axes[0].plot([0, 1], [left, right], color="#aeb7bf", linewidth=0.85, zorder=1)
        axes[0].text(1.035, right, str(seed), va="center", fontsize=6.2, color="#555555")
    axes[0].scatter(np.zeros_like(adam_error), adam_error, color="#2f5597", s=27,
                    edgecolor="white", linewidth=0.45, label="Adam/autodiff", zorder=3)
    axes[0].scatter(np.ones_like(lbfgs_error), lbfgs_error, color="#e69f00", marker="D", s=25,
                    edgecolor="white", linewidth=0.45, label="L-BFGS-B/classical", zorder=3)
    axes[0].set_xticks([0, 1], ["Adam/autodiff", "L-BFGS-B/classical"])
    axes[0].set_ylabel(r"Parameter $\ell_2$ error", fontsize=9.2)
    axes[0].grid(axis="y", color="#d9dfe3", linewidth=0.55)

    axes[1].scatter(adam_theta[:, 0], adam_theta[:, 1], color="#2f5597", s=30,
                    edgecolor="white", linewidth=0.45, label="Adam/autodiff")
    axes[1].scatter(lbfgs_theta[:, 0], lbfgs_theta[:, 1], color="#e69f00", marker="D", s=28,
                    edgecolor="white", linewidth=0.45, label="L-BFGS-B/classical")
    axes[1].scatter([truth[0]], [truth[1]], marker="*", s=95, color="#1b1b1b",
                    edgecolor="white", linewidth=0.5, label="Truth", zorder=4)
    for index, seed in enumerate(seeds):
        axes[1].plot([adam_theta[index, 0], lbfgs_theta[index, 0]],
                     [adam_theta[index, 1], lbfgs_theta[index, 1]],
                     color="#c6cbd0", linewidth=0.7, zorder=0)
    axes[1].set_xlabel(r"Recovered parameter $\theta_1$", fontsize=9.2)
    axes[1].set_ylabel(r"Recovered parameter $\theta_2$", fontsize=9.2)
    axes[1].grid(True, color="#d9dfe3", linewidth=0.55)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=1, loc="lower center", fontsize=8.2,
               bbox_to_anchor=(0.5, 0.015), borderaxespad=0.0)
    for ax in axes:
        ax.tick_params(labelsize=8.3)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    label_panels(axes)
    enlarge_text(fig)
    save(fig, "fig_calibration_reproducibility.pdf", bottom=0.20)


def periodic_heat_2d_figure() -> None:
    """Summarize accuracy and scaling of the direct 2D PDE audit."""
    rows = DATA["periodic_heat_2d"]["rows"]
    grids = sorted({row["grid"] for row in rows})
    value_error = [max(row["max_abs_value_error"] for row in rows if row["grid"] == grid) for grid in grids]
    gradient_error = [max(row["energy_gradient_relative_error"] for row in rows if row["grid"] == grid) for grid in grids]
    latency = [float(np.median([row["mean_forward_ms"] for row in rows if row["grid"] == grid])) for grid in grids]
    memory = [max((row["peak_incremental_cuda_bytes"] or 0) for row in rows if row["grid"] == grid) / 2**20 for grid in grids]

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 10.2))
    axes[0].plot(grids, value_error, marker="o", color="#2f5597", linewidth=1.35,
                 markersize=3.8, label="Value max")
    axes[0].plot(grids, gradient_error, marker="D", color="#e69f00", linewidth=1.35,
                 markersize=3.5, label="Gradient relative max")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Error (log scale)")
    axes[0].legend(frameon=False, fontsize=6.5)

    axes[1].plot(grids, latency, marker="o", color="#176b87", linewidth=1.35, markersize=3.8)
    axes[1].set_ylabel("Median forward latency (ms)")

    axes[2].plot(grids, memory, marker="o", color="#7a5195", linewidth=1.35, markersize=3.8)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Peak incremental CUDA memory (MiB)")

    for ax in axes:
        ax.set_xlabel("Grid size $N$ ($N\\times N$ state)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(grids, [str(grid) for grid in grids])
        ax.grid(True, which="both", color="#d9dfe3", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    label_panels(axes)
    enlarge_text(fig)
    save(fig, "fig_periodic_heat_2d.pdf", bottom=0.04)


if __name__ == "__main__":
    protocol_figure()
    common_ood_figure()
    profile_figure()
    precision_tradeoff_figure()
    real_data_figure()
    simulation_figure()
    reliability_signature_figure()
    calibration_reproducibility_figure()
    periodic_heat_2d_figure()
    print(f"wrote figures to {OUT}")
