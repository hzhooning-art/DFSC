"""Stage 64: third-domain audit on public hydraulic cooling transients."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p5_memory_protocol import (  # noqa: E402
    CurveRecord,
    GateConfig,
    decide,
    evaluate,
    fit,
    fixed_grid_nnls_error,
    identifiability_certificate,
    prony_error,
    report,
)


DATA = ROOT / "data" / "external" / "uci_hydraulic_447"
ARCHIVE = DATA / "uci_447.zip"
OUTPUT = ROOT / "results" / "public_uci_hydraulic_transients.json"
SUMMARY = ROOT / "results" / "public_uci_hydraulic_transients.md"
FIGURE_DIR = Path(os.environ.get("P5_FIGURE_DIR", ROOT / "figures"))
FIGURE_PDF = FIGURE_DIR / "fig_stage64_hydraulic_transients.pdf"
FIGURE_PNG = FIGURE_DIR / "fig_stage64_hydraulic_transients.png"
EXPECTED_SHA256 = "24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2"
DOI = "10.24432/C5CW21"
CHANNELS = ("CE", "CP", "TS1", "TS2")
WINDOW = slice(20, 56)
CALIBRATION_POINTS = 22
GATES = GateConfig()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_curves() -> list[CurveRecord]:
    if sha256(ARCHIVE) != EXPECTED_SHA256:
        raise ValueError("UCI hydraulic archive checksum differs from the frozen source")
    profile = np.loadtxt(DATA / "profile.txt", dtype=int)
    signals = {name: np.loadtxt(DATA / f"{name}.txt") for name in CHANNELS}
    if profile.shape != (2205, 5) or any(signal.shape != (2205, 60) for signal in signals.values()):
        raise ValueError("unexpected UCI hydraulic schema")

    # Hold valve, pump, accumulator, and stability fixed; cooler condition is
    # the only experimental factor varied across the three held-out groups.
    selected = np.flatnonzero(
        (profile[:, 1] == 100)
        & (profile[:, 2] == 0)
        & (profile[:, 3] == 130)
        & (profile[:, 4] == 0)
    )
    curves = []
    time = np.arange(WINDOW.stop - WINDOW.start, dtype=float)
    for index in selected:
        for channel in CHANNELS:
            segment = signals[channel][index, WINDOW].astype(float)
            calibration = segment[:CALIBRATION_POINTS]
            scale = max(float(np.ptp(calibration)), 1e-3)
            value = (segment - segment[0]) / scale
            curves.append(CurveRecord(
                unit=f"cycle-{index + 1:04d}",
                group=f"cooler-{profile[index, 0]:03d}",
                channel=channel,
                time=time,
                value=value,
            ))
    counts = defaultdict(int)
    for curve in curves:
        counts[curve.group] += 1
    if len(curves) != 120 or set(counts.values()) != {40}:
        raise ValueError("expected 30 cycles, four channels, and ten cycles per cooler condition")
    return curves


def _unit_medians(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["unit"]].append(float(row["nrmse"]))
    return {unit: float(np.median(values)) for unit, values in grouped.items()}


def _cluster_statistics(evaluation: dict, seed: int = 6401) -> dict:
    rng = np.random.default_rng(seed)
    baseline = _unit_medians(evaluation["rank_records"]["1"]["curve_errors"])
    output = {}
    for rank in (2, 3):
        candidate = _unit_medians(evaluation["rank_records"][str(rank)]["curve_errors"])
        units = sorted(baseline)
        base = np.asarray([baseline[unit] for unit in units])
        test = np.asarray([candidate[unit] for unit in units])
        improvement = (base - test) / np.maximum(base, 1e-15)
        boot = np.asarray([
            np.median(rng.choice(improvement, size=len(improvement), replace=True))
            for _ in range(5000)
        ])
        signed = wilcoxon(base, test, alternative="greater", method="auto")
        output[str(rank)] = {
            "independent_unit": "hydraulic load cycle",
            "n_units": len(units),
            "median_relative_improvement": float(np.median(improvement)),
            "cluster_bootstrap_95pct": np.quantile(boot, [0.025, 0.975]).tolist(),
            "wilcoxon_p_one_sided": float(signed.pvalue),
        }
    return output


def _threshold_sensitivity(evaluation: dict) -> list[dict]:
    rows = []
    for gain in (0.02, 0.05, 0.10):
        for stability in (0.50, 0.80, 1.20):
            for separation in (1.10, 1.20, 1.50):
                gates = GateConfig(
                    predictive_gain=gain,
                    max_log_rate_std=stability,
                    min_rate_ratio=separation,
                )
                rows.append({
                    "predictive_gain": gain,
                    "max_log_rate_std": stability,
                    "min_rate_ratio": separation,
                    "decision": decide(evaluation, gates)["decision"],
                })
    return rows


def _stretched_exponential_error(curve: CurveRecord) -> float:
    split = CALIBRATION_POINTS
    time = curve.time
    value = curve.value

    def residual(parameters: np.ndarray) -> np.ndarray:
        offset, amplitude, log_tau, logit_beta = parameters
        tau = np.exp(log_tau)
        beta = 0.2 + 1.8 / (1.0 + np.exp(-logit_beta))
        prediction = offset + amplitude * np.exp(-np.power(time[:split] / tau, beta))
        return prediction - value[:split]

    initial = np.asarray((value[split - 1], value[0] - value[split - 1], np.log(8.0), 0.0))
    result = least_squares(residual, initial, max_nfev=500)
    offset, amplitude, log_tau, logit_beta = result.x
    tau = np.exp(log_tau)
    beta = 0.2 + 1.8 / (1.0 + np.exp(-logit_beta))
    prediction = offset + amplitude * np.exp(-np.power(time[split:] / tau, beta))
    scale = max(float(np.ptp(value[:split])), 0.10)
    return float(np.sqrt(np.mean((prediction - value[split:]) ** 2)) / scale)


def _ar2_error(curve: CurveRecord) -> float:
    split = CALIBRATION_POINTS
    observed = curve.value[:split].astype(float).tolist()
    design = np.column_stack((np.ones(split - 2), curve.value[1:split - 1], curve.value[:split - 2]))
    coefficients = np.linalg.lstsq(design, curve.value[2:split], rcond=None)[0]
    for _ in range(len(curve.value) - split):
        observed.append(float(coefficients @ np.asarray((1.0, observed[-1], observed[-2]))))
    prediction = np.asarray(observed[split:])
    scale = max(float(np.ptp(curve.value[:split])), 0.10)
    return float(np.sqrt(np.mean((prediction - curve.value[split:]) ** 2)) / scale)


def _model_family_baselines(curves: list[CurveRecord], evaluation: dict, seed: int = 6402) -> dict:
    rng = np.random.default_rng(seed)
    shared = _unit_medians(evaluation["rank_records"]["1"]["curve_errors"])
    curve_records = {
        "stretched_exponential": [],
        "fixed_grid_nnls": [],
        "ar2_recurrence": [],
        "prony_recurrence": [],
    }
    for curve in curves:
        curve_records["stretched_exponential"].append({"unit": curve.unit, "nrmse": _stretched_exponential_error(curve)})
        curve_records["fixed_grid_nnls"].append({"unit": curve.unit, "nrmse": fixed_grid_nnls_error(curve, CALIBRATION_POINTS / len(curve.time))})
        curve_records["ar2_recurrence"].append({"unit": curve.unit, "nrmse": _ar2_error(curve)})
        curve_records["prony_recurrence"].append({"unit": curve.unit, "nrmse": prony_error(curve, rank=3, calibration_fraction=CALIBRATION_POINTS / len(curve.time))})

    output = {"shared_rank_1": {"median_cycle_nrmse": float(np.median(list(shared.values())))}}
    for name, records in curve_records.items():
        baseline = _unit_medians(records)
        units = sorted(shared)
        shared_values = np.asarray([shared[unit] for unit in units])
        baseline_values = np.asarray([baseline[unit] for unit in units])
        difference = baseline_values - shared_values
        bootstrap = np.asarray([
            np.median(rng.choice(difference, size=len(difference), replace=True)) for _ in range(4000)
        ])
        test = wilcoxon(baseline_values, shared_values, alternative="greater", method="auto")
        output[name] = {
            "median_cycle_nrmse": float(np.median(baseline_values)),
            "shared_minus_baseline_median_nrmse": float(np.median(shared_values - baseline_values)),
            "baseline_minus_shared_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "wilcoxon_p_baseline_greater": float(test.pvalue),
            "independent_units": len(units),
        }
    output["interpretation"] = (
        "These are matched early-to-late predictive controls, not claims that one model family "
        "identifies the physical cooling mechanism."
    )
    return output


def _full_fit_certificates(curves: list[CurveRecord]) -> dict:
    output = {}
    for rank in (1, 2, 3):
        fitted = fit(curves, rank=rank, starts=8, rate_bounds=(1.0 / 600.0, 1.5))
        output[str(rank)] = {
            "fit": fitted,
            "identifiability": identifiability_certificate(curves, fitted["rates"]),
        }
    return output


def _plot(payload: dict) -> None:
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10})
    curves = load_curves()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    colors = {"cooler-003": "#9b59b6", "cooler-020": "#e69f00", "cooler-100": "#2f5aa0"}
    for group in sorted(colors):
        rows = [row for row in curves if row.group == group and row.channel == "CE"]
        values = np.stack([row.value for row in rows])
        axes[0].plot(rows[0].time, np.mean(values, axis=0), color=colors[group], label=group.replace("cooler-", "cooler "))
        axes[0].fill_between(
            rows[0].time,
            np.quantile(values, 0.25, axis=0),
            np.quantile(values, 0.75, axis=0),
            color=colors[group], alpha=0.16,
        )
    axes[0].set_xlabel("Time after declared transient start (s)")
    axes[0].set_ylabel("Normalized cooling-efficiency response")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    ranks = np.asarray([1, 2, 3])
    errors = [payload["evaluation"]["rank_records"][str(rank)]["median_prediction_nrmse"] for rank in ranks]
    indices = [payload["full_fit_certificates"][str(rank)]["identifiability"]["local_boundary_index"] for rank in ranks]
    axes[1].plot(ranks, errors, "o-", color="#1f778d", linewidth=2, label="held-cycle NRMSE")
    axes[1].set_xlabel("Candidate shared rank")
    axes[1].set_ylabel("Median held-cycle NRMSE", color="#1f778d")
    axes[1].tick_params(axis="y", labelcolor="#1f778d")
    axes[1].set_xticks(ranks)
    axes[1].grid(alpha=0.25)
    twin = axes[1].twinx()
    finite_indices = [value if np.isfinite(value) else np.nan for value in indices]
    twin.plot(ranks, finite_indices, "s--", color="#c44e52", linewidth=2, label="local boundary index")
    twin.set_ylabel("Local boundary index", color="#c44e52")
    twin.tick_params(axis="y", labelcolor="#c44e52")
    handles = axes[1].lines + twin.lines
    axes[1].legend(
        handles,
        [line.get_label() for line in handles],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        borderaxespad=0.0,
    )

    for label, axis in zip(("a", "b"), axes):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontsize=13, fontweight="bold")
    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_payload() -> dict:
    curves = load_curves()
    evaluation = evaluate(
        curves,
        starts=6,
        calibration_fraction=CALIBRATION_POINTS / (WINDOW.stop - WINDOW.start),
        rate_bounds=(1.0 / 600.0, 1.5),
    )
    payload = {
        "schema_version": "1.0.0",
        "experiment": "stage64_public_uci_hydraulic_transients",
        "protocol_frozen_before_fit": True,
        "source": {
            "title": "Condition monitoring of hydraulic systems",
            "doi": DOI,
            "url": "https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems",
            "license": "CC-BY-4.0",
            "archive_sha256": sha256(ARCHIVE),
            "independent_cycles": len({row.unit for row in curves}),
            "channels": list(CHANNELS),
            "held_groups": sorted({row.group for row in curves}),
        },
        "preprocessing": {
            "fixed_conditions": "valve=100, pump leakage=0, accumulator=130 bar, stable flag=0",
            "varied_condition": "cooler efficiency in {3, 20, 100} percent",
            "cycle_window_seconds": [20, 55],
            "normalization": "subtract first window value and divide by early-calibration range",
            "tail_leakage": False,
            "smoothing": "none",
        },
        "evaluation": evaluation,
        "decision": decide(evaluation, GATES),
        "statistics": _cluster_statistics(evaluation),
        "model_family_baselines": _model_family_baselines(curves, evaluation),
        "threshold_sensitivity": _threshold_sensitivity(evaluation),
        "full_fit_certificates": _full_fit_certificates(curves),
        "claim_boundary": (
            "The task audits a shared empirical cooling-transient realization across controlled "
            "hydraulic load cycles; it does not identify a unique component-level mechanism."
        ),
    }
    payload["checks"] = {
        "checksum": payload["source"]["archive_sha256"] == EXPECTED_SHA256,
        "three_cooler_groups": len(evaluation["groups"]) == 3,
        "thirty_independent_cycles": payload["source"]["independent_cycles"] == 30,
        "four_channels": len(payload["source"]["channels"]) == 4,
        "clustered_inference": all(row["independent_unit"] == "hydraulic load cycle" for row in payload["statistics"].values()),
        "model_family_controls": all(
            payload["model_family_baselines"][name]["independent_units"] == 30
            for name in ("stretched_exponential", "fixed_grid_nnls", "ar2_recurrence", "prony_recurrence")
        ),
        "threshold_grid_complete": len(payload["threshold_sensitivity"]) == 27,
        "certificates_finite": all(
            np.isfinite(row["identifiability"]["minimum_projected_information_eigenvalue"])
            for row in payload["full_fit_certificates"].values()
        ),
    }
    payload["route_pass"] = all(payload["checks"].values())
    return payload


def write_summary(payload: dict) -> None:
    lines = [
        "# Public UCI hydraulic-transient audit",
        "",
        f"Decision: **{payload['decision']['decision']}**.",
        "",
        f"Route pass: **{payload['route_pass']}**.",
        "",
        "| Rank | Mean BIC | Median held-cycle NRMSE | Local boundary index |",
        "|---:|---:|---:|---:|",
    ]
    for rank in (1, 2, 3):
        record = payload["evaluation"]["rank_records"][str(rank)]
        index = payload["full_fit_certificates"][str(rank)]["identifiability"]["local_boundary_index"]
        index_text = "not applicable" if not np.isfinite(index) else f"{index:.5g}"
        lines.append(f"| {rank} | {record['mean_bic']:.3f} | {record['median_prediction_nrmse']:.5f} | {index_text} |")
    lines.extend([
        "",
        "Inference is clustered by independent 60-s hydraulic load cycle; the four channels are not treated as replicates.",
    ])
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload()
    report(payload, OUTPUT)
    write_summary(payload)
    _plot(payload)
    print(json.dumps({
        "decision": payload["decision"]["decision"],
        "route_pass": payload["route_pass"],
        "statistics": payload["statistics"],
    }, indent=2))


if __name__ == "__main__":
    main()
