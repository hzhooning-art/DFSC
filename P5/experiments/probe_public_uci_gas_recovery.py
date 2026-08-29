"""Stage 63: independent public gas-sensor recovery audit."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p5_memory_protocol import CurveRecord, GateConfig, decide, evaluate, fit, report  # noqa: E402
from p5_memory_protocol.core import fixed_grid_nnls_error, independent_nls_error, prony_error  # noqa: E402


DATA = ROOT / "data" / "external" / "uci_gas_flow_308" / "files" / "rawdata.csv.gz"
OUTPUT = ROOT / "results" / "public_uci_gas_recovery.json"
SUMMARY = ROOT / "results" / "public_uci_gas_recovery.md"
EXPECTED_SHA256 = "ed2aae124aa733fd475af0ede16431252103e15644fb43d519bb322800c3925d"
DOI = "10.24432/C5BG7G"
SAMPLE_RATE = 25
RECOVERY_START = 180 * SAMPLE_RATE
CYCLE_POINTS = 12 * SAMPLE_RATE
RECOVERY_CYCLES = 10
GATES = GateConfig()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_curves(path: Path = DATA) -> list[CurveRecord]:
    if sha256(path) != EXPECTED_SHA256:
        raise ValueError("UCI raw-data checksum differs from the frozen source")
    curves = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != 7509 or header[9] != "dR_t1" or header[-1] != "dR_t7500":
            raise ValueError("unexpected UCI raw-data schema")
        for row in reader:
            if row[6] == "air":
                continue
            trace = np.asarray(row[9 + RECOVERY_START: 9 + RECOVERY_START + RECOVERY_CYCLES * CYCLE_POINTS], dtype=float)
            cycle_values = np.median(trace.reshape(RECOVERY_CYCLES, CYCLE_POINTS), axis=1)
            scale = float(cycle_values[0])
            if not np.isfinite(scale) or abs(scale) < 1e-8:
                continue
            value = cycle_values / scale
            time = (np.arange(RECOVERY_CYCLES, dtype=float) + 0.5) * 12.0
            curves.append(CurveRecord(
                unit=f"sample-{int(row[1]):03d}", group=row[3], channel=f"sensor-{int(row[0]):02d}", time=time, value=value,
            ))
    if len({row.unit for row in curves}) != 50 or len({row.channel for row in curves}) != 16:
        raise ValueError("expected 50 non-air experiments and 16 sensor channels")
    return curves


def _unit_medians(error_rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in error_rows:
        grouped[row["unit"]].append(float(row["nrmse"]))
    return {unit: float(np.median(values)) for unit, values in grouped.items()}


def _cluster_statistics(evaluation: dict, seed: int = 6301) -> dict:
    baseline = _unit_medians(evaluation["rank_records"]["1"]["curve_errors"])
    rng = np.random.default_rng(seed)
    output = {}
    for rank in (2, 3):
        candidate = _unit_medians(evaluation["rank_records"][str(rank)]["curve_errors"])
        units = sorted(set(baseline) & set(candidate))
        base = np.asarray([baseline[unit] for unit in units])
        test = np.asarray([candidate[unit] for unit in units])
        improvement = (base - test) / np.maximum(base, 1e-15)
        boot = np.asarray([np.median(rng.choice(improvement, size=len(improvement), replace=True)) for _ in range(5000)])
        signed = wilcoxon(base, test, alternative="greater", method="auto")
        output[str(rank)] = {
            "independent_unit": "gas exposure experiment",
            "n_units": len(units),
            "median_relative_improvement": float(np.median(improvement)),
            "cluster_bootstrap_95pct": np.quantile(boot, [0.025, 0.975]).tolist(),
            "wilcoxon_statistic": float(signed.statistic),
            "wilcoxon_p_one_sided": float(signed.pvalue),
        }
    return output


def _baseline_audit(curves: list[CurveRecord], evaluation: dict) -> dict:
    shared = evaluation["rank_records"]["3"]["curve_errors"]
    shared_map = {(row["unit"], row["channel"]): row["nrmse"] for row in shared}
    methods = {"shared_rank3": [], "independent_nls_rank3": [], "fixed_grid_nnls": [], "prony_rank3": []}
    for curve in curves:
        key = (curve.unit, curve.channel)
        methods["shared_rank3"].append((curve.unit, float(shared_map[key])))
        methods["independent_nls_rank3"].append((curve.unit, independent_nls_error(curve, rank=3)))
        methods["fixed_grid_nnls"].append((curve.unit, fixed_grid_nnls_error(curve)))
        methods["prony_rank3"].append((curve.unit, prony_error(curve, rank=3)))
    summary = {}
    unit_maps = {}
    for name, values in methods.items():
        grouped: dict[str, list[float]] = defaultdict(list)
        for unit, error in values:
            grouped[unit].append(error)
        unit_values = np.asarray([np.median(grouped[unit]) for unit in sorted(grouped)])
        unit_maps[name] = {unit: float(np.median(grouped[unit])) for unit in sorted(grouped)}
        summary[name] = {
            "median_experiment_nrmse": float(np.median(unit_values)),
            "iqr_experiment_nrmse": np.quantile(unit_values, [0.25, 0.75]).tolist(),
            "n_experiments": len(unit_values),
        }
    shared_units = unit_maps["shared_rank3"]
    for name in ("independent_nls_rank3", "fixed_grid_nnls", "prony_rank3"):
        units = sorted(set(shared_units) & set(unit_maps[name]))
        shared_values = np.asarray([shared_units[unit] for unit in units])
        baseline_values = np.asarray([unit_maps[name][unit] for unit in units])
        improvement = (baseline_values - shared_values) / np.maximum(baseline_values, 1e-15)
        test = wilcoxon(baseline_values, shared_values, alternative="greater", method="auto")
        summary[name]["shared_rank3_comparison"] = {
            "median_relative_improvement_of_shared": float(np.median(improvement)),
            "wilcoxon_p_one_sided": float(test.pvalue),
            "independent_unit": "gas exposure experiment",
        }
    return summary


def _threshold_sensitivity(evaluation: dict) -> list[dict]:
    rows = []
    for gain in (0.02, 0.05, 0.10):
        for stability in (0.50, 0.80, 1.20):
            for separation in (1.10, 1.20, 1.50):
                gates = GateConfig(predictive_gain=gain, max_log_rate_std=stability, min_rate_ratio=separation)
                outcome = decide(evaluation, gates)
                rows.append({"predictive_gain": gain, "max_log_rate_std": stability, "min_rate_ratio": separation, "decision": outcome["decision"]})
    return rows


def _multistart_audit(curves: list[CurveRecord]) -> list[dict]:
    rows = []
    for starts in (2, 4, 8):
        fitted = fit(curves, rank=3, starts=starts)
        rows.append({"starts": starts, "rates": fitted["rates"], "sse": fitted["sse"], "success": fitted["success"]})
    return rows


def build_payload() -> dict:
    curves = load_curves()
    evaluation = evaluate(curves, starts=6)
    decision_record = decide(evaluation, GATES)
    payload = {
        "schema_version": "1.0.0",
        "experiment": "stage63_public_uci_gas_recovery",
        "protocol_frozen_before_fit": True,
        "source": {
            "title": "Gas sensor array under flow modulation",
            "doi": DOI,
            "url": "https://archive.ics.uci.edu/dataset/308/gas%2Bsensor%2Barray%2Bunder%2Bflow%2Bmodulation",
            "license": "CC-BY-4.0",
            "sha256": sha256(DATA),
            "independent_non_air_experiments": len({row.unit for row in curves}),
            "channels": len({row.channel for row in curves}),
            "acquisition_batches": len({row.group for row in curves}),
        },
        "preprocessing": {
            "recovery_seconds": [180, 300],
            "registered_cycle_aggregation_seconds": 12,
            "aggregation": "median within ventilator cycle",
            "smoothing": "none",
            "normalization": "divide by first recovery-cycle value; no tail leakage",
        },
        "evaluation": evaluation,
        "decision": decision_record,
        "statistics": _cluster_statistics(evaluation),
        "baselines": _baseline_audit(curves, evaluation),
        "threshold_sensitivity": _threshold_sensitivity(evaluation),
        "multistart_audit": _multistart_audit(curves),
        "claim_boundary": "The decision concerns a shared empirical recovery realization across declared batches; it is not a unique chemical mechanism assignment.",
    }
    payload["checks"] = {
        "checksum": payload["source"]["sha256"] == EXPECTED_SHA256,
        "clustered_independence": all(item["independent_unit"] == "gas exposure experiment" for item in payload["statistics"].values()),
        "five_held_batches": len(evaluation["groups"]) == 5,
        "three_external_baselines": len(payload["baselines"]) >= 4,
        "multistart_complete": len(payload["multistart_audit"]) == 3,
        "threshold_grid_complete": len(payload["threshold_sensitivity"]) == 27,
    }
    payload["route_pass"] = all(payload["checks"].values())
    return payload


def write_summary(payload: dict) -> None:
    lines = [
        "# Public UCI gas-recovery audit", "",
        f"Decision: **{payload['decision']['decision']}**.", "",
        f"Route pass: **{payload['route_pass']}**.", "",
        "## External baseline audit", "",
        "| Method | Median experiment NRMSE | IQR |",
        "|---|---:|---:|",
    ]
    for name, row in payload["baselines"].items():
        lines.append(f"| {name} | {row['median_experiment_nrmse']:.5f} | [{row['iqr_experiment_nrmse'][0]:.5f}, {row['iqr_experiment_nrmse'][1]:.5f}] |")
    lines.extend(["", "Sensor channels were clustered within each independent gas exposure before inference."])
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload()
    report(payload, OUTPUT)
    write_summary(payload)
    print(json.dumps({"decision": payload["decision"]["decision"], "route_pass": payload["route_pass"], "baselines": payload["baselines"]}, indent=2))


if __name__ == "__main__":
    main()
