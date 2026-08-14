"""Create the standards-facing interoperability and mutation-test figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "p4_csi_conformance_validation.json"
OUT = Path(__file__).resolve().parent / "figures" / "fig_conformance_validation.pdf"


def main() -> None:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    interop = data["interoperability"]
    mutation = data["fault_injection"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.15), constrained_layout=True)

    labels = ["API--CLI", "Key order", "Evidence kept", "Requalification"]
    values = [
        interop["api_cli_equivalence_rate"],
        interop["key_order_invariance_rate"],
        interop["migration_evidence_preservation_rate"],
        interop["migration_requalification_rate"],
    ]
    axes[0].bar(labels, np.asarray(values) * 100, color=["#35669a", "#4f86b6", "#72a6c7"], width=0.62)
    axes[0].set_ylabel("Equivalent records (%)")
    axes[0].set_ylim(0, 112)
    axes[0].set_title("Interface interoperability")
    for i, value in enumerate(values):
        axes[0].text(i, value * 100 + 2.2, f"{int(round(value * 100))}%", ha="center", va="bottom")

    faults = {entry["fault"]: entry for entry in mutation["catalogue"]}
    fault_labels = [
        "Truncation", "Detached grad.", "Dtype", "Batch", "Device", "OOD", "Units",
        "Provenance", "Scope freeze", "Scope coverage"
    ]
    fault_keys = [
        "inadequate_truncation", "detached_gradient", "silent_dtype_downgrade", "batch_crosstalk",
        "silent_cpu_fallback", "ood_scope_misuse", "unit_mismatch", "missing_provenance",
        "unfrozen_scope", "insufficient_scope_coverage",
    ]
    rates = np.asarray([faults[key]["detection_rate"] for key in fault_keys]) * 100
    lower = np.asarray([faults[key]["wilson_95"][0] for key in fault_keys]) * 100
    axes[1].barh(fault_labels, rates, color="#2f7f62", height=0.62)
    axes[1].errorbar(rates, np.arange(len(rates)), xerr=[rates - lower, np.zeros_like(rates)],
                     fmt="none", ecolor="#173f32", capsize=3, linewidth=1.2)
    axes[1].set_xlim(75, 103)
    axes[1].set_xlabel("Injected faults detected (%)")
    axes[1].set_title("Specification mutation tests")
    axes[1].invert_yaxis()
    axes[1].text(0.99, 0.02, "Clean false rejections: 0/40",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8.5)

    for label, ax in zip(["a", "b"], axes):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.grid(axis="y" if ax is axes[0] else "x", color="#d9dfe3", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
