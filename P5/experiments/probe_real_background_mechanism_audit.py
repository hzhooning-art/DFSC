"""Stage 54: real-background semisynthetic truth audit and blind real-data audit.

Public elastomer relaxation traces supply sampling and residual morphology.  A
known shared/mild/severe spectral mechanism is injected for the confirmatory
part.  A separate blind audit of normalized public curves has no truth label
and is never counted as mechanism-recovery evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from probe_budget_consensus_abstention import consensus_decision
from probe_conditional_contract_transfer import CONSTRUCTIONS, DRIFTS, score_matrix
from probe_cross_start_consistency_transfer import select_consistent_candidate
from probe_decomposed_tolerance_transfer import (
    BLOCKS, HORIZON, NOISE_CORRELATION, PROXY_SCOPE, TRUE_CENTRAL_RATES,
    frozen_total_tolerance,
)
from probe_extended_refinement_transfer import ADAM_STEPS, load_frozen_stage48
from probe_high_dimensional_shared_spectrum import DEVICE, DTYPE, fit_candidate, independent_lifted_response
from probe_nested_group_sharing_gate import fit_grouped_candidate
from probe_noise_aware_sharing_gate import BIC_EVIDENCE_LIMIT, classify_with_limit, second_difference_correlation_proxy
from probe_noise_scale_optimizer_transfer import GROUPED_STARTS, SHARED_STARTS, mixed_difference_noise_scale, scale_correction
from probe_optimizer_budget_stability import LBFGS_BUDGETS, decision_class


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "results"
STAGE52_RESULT = RESULTS / "high_noise_conditional_calibration.json"
DATA_DIR = PROJECT_ROOT / "P3" / "data" / "public_elastomer" / "extracted" / "Stress relaxation"
BACKGROUND_FILES = (
    "Ecoflex 00-30-relaxation.csv",
    "Dragon Skin 20-relaxation.csv",
    "Mold Star 30-relaxation.csv",
)
BLIND_FILES = (
    "Cheetah-relaxation.csv",
    "Dragon Skin 20-relaxation.csv",
    "Dragon Skin 30-relaxation.csv",
    "Ecoflex 00-20-relaxation.csv",
    "Ecoflex 00-30-relaxation.csv",
    "Ecoflex 00-50-relaxation.csv",
    "Mold Max 14NV-relaxation.csv",
    "Mold Star 30-relaxation.csv",
)
SEED_BASE = 201000
CHANNELS = 48
NOISE_STD = 1.6e-3
REPEATS = 2
ROW_STRIDE = 100

MAX_FALSE_REFUSAL = 0.10
MIN_SEVERE_REFUSAL = 0.75
MIN_SELECTIVE_ACCURACY = 0.85
MIN_COVERAGE = 0.60


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_relaxation_columns(path: Path, stride: int = ROW_STRIDE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, strains, stresses = [], [], []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Time (s)", "Strain (%)", "Stress (MPa)"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"missing expected columns in {path.name}")
        for index, row in enumerate(reader):
            if index % stride:
                continue
            try:
                time_value = float(row["Time (s)"])
                strain_value = float(row["Strain (%)"])
                stress_value = float(row["Stress (MPa)"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(time_value + strain_value + stress_value):
                times.append(time_value)
                strains.append(strain_value)
                stresses.append(stress_value)
    if len(times) < 200:
        raise ValueError(f"insufficient finite relaxation samples in {path.name}")
    return np.asarray(times), np.asarray(strains), np.asarray(stresses)


@lru_cache(maxsize=None)
def standardized_real_residual(path: Path) -> np.ndarray:
    _, _, stress = load_relaxation_columns(path)
    window = min(101, max(11, (len(stress) // 40) | 1))
    smooth = np.convolve(stress, np.ones(window) / window, mode="same")
    margin = window
    residual = stress[margin:-margin] - smooth[margin:-margin]
    residual -= np.median(residual)
    scale = 1.4826 * np.median(np.abs(residual))
    if not np.isfinite(scale) or scale <= 1e-15:
        scale = float(np.std(residual))
    if not np.isfinite(scale) or scale <= 1e-15:
        raise ValueError(f"degenerate residual background in {path.name}")
    return residual / scale


def residual_segment(background: np.ndarray, length: int, start: int) -> np.ndarray:
    indices = (np.arange(length) + start) % len(background)
    segment = background[indices].astype(float, copy=True)
    segment -= np.mean(segment)
    std = float(np.std(segment))
    return segment / max(std, 1e-12)


def build_real_background_observation(path: Path, construction: str, drift: float, seed: int):
    rng = np.random.default_rng(seed)
    background = standardized_real_residual(path)
    times = torch.linspace(0.0, HORIZON, 65, dtype=DTYPE, device=DEVICE)
    labels = np.repeat(np.arange(BLOCKS), CHANNELS // BLOCKS)
    scores = score_matrix(construction)[labels]
    rates = np.exp(np.log(TRUE_CENTRAL_RATES)[None, :] + drift * scores)
    amplitude = np.linspace(0.28, 0.78, CHANNELS)[:, None]
    tilt = np.linspace(0.80, 1.20, CHANNELS)[:, None]
    weights = np.concatenate([0.58 * amplitude / tilt, 0.42 * amplitude * tilt], axis=1)
    clean = independent_lifted_response(
        times,
        torch.tensor(weights, dtype=DTYPE, device=DEVICE),
        torch.tensor(rates, dtype=DTYPE, device=DEVICE),
    )
    common = residual_segment(background, 65, int(rng.integers(len(background))))[:, None]
    independent = np.column_stack([
        residual_segment(background, 65, int(rng.integers(len(background)))) for _ in range(CHANNELS)
    ])
    noise = math.sqrt(NOISE_CORRELATION) * common + math.sqrt(1.0 - NOISE_CORRELATION) * independent
    noise /= max(float(np.std(noise)), 1e-12)
    observations = clean + NOISE_STD * torch.tensor(noise, dtype=DTYPE, device=DEVICE)
    train = np.sort(rng.choice(np.arange(1, 48), size=34, replace=False))
    return (
        times, observations,
        torch.tensor(train, dtype=torch.long, device=DEVICE),
        torch.tensor(np.arange(48, 65), dtype=torch.long, device=DEVICE),
        labels,
    )


def fit_observation(
    times, observations, train_idx, val_idx, labels, seed: int, budget: int,
    multiplier: float, noise_calibration: dict, consistency_calibration: dict,
) -> dict:
    candidates = [
        fit_candidate(
            times, observations, train_idx, val_idx, 2, True, seed * 10 + start,
            adam_steps=ADAM_STEPS, lbfgs_steps=budget,
        )
        for start in range(SHARED_STARTS)
    ]
    shared, diagnostics, adequate, signal_scale = select_consistent_candidate(
        candidates, times, observations, consistency_calibration
    )
    grouped = min(
        [
            fit_grouped_candidate(
                times, observations, train_idx, val_idx,
                torch.tensor(labels, dtype=torch.long, device=DEVICE),
                seed * 10 + 20 + start, adam_steps=ADAM_STEPS, lbfgs_steps=budget,
            )
            for start in range(GROUPED_STARTS)
        ], key=lambda candidate: candidate.bic,
    )
    noise_proxy = mixed_difference_noise_scale(observations)
    correlation_proxy = second_difference_correlation_proxy(observations)
    base = frozen_total_tolerance(correlation_proxy) + scale_correction(noise_proxy, noise_calibration)
    adjusted = base * multiplier
    in_scope = (
        noise_calibration["noise_proxy_min"] <= noise_proxy <= noise_calibration["noise_proxy_max"]
        and PROXY_SCOPE[0] <= correlation_proxy <= PROXY_SCOPE[1]
    )
    if shared is None:
        support, shared_val = None, None
        adjusted_decision = uncalibrated_decision = "INDETERMINATE_OPTIMIZATION"
    else:
        support = shared.bic - grouped.bic
        shared_val = shared.val_rmse
        adjusted_decision = classify_with_limit(support, shared.val_rmse, grouped.val_rmse, adjusted)
        uncalibrated_decision = classify_with_limit(support, shared.val_rmse, grouped.val_rmse, base)
    bic_class = "REFUSE" if support is not None and support >= BIC_EVIDENCE_LIMIT else ("RETAIN" if support is not None else "INDETERMINATE")
    return {
        "lbfgs_steps": budget,
        "decision": adjusted_decision,
        "decision_class": decision_class(adjusted_decision),
        "uncalibrated_decision": uncalibrated_decision,
        "uncalibrated_class": decision_class(uncalibrated_decision),
        "bic_only_class": bic_class,
        "group_bic_support": support,
        "shared_val_rmse": shared_val,
        "grouped_val_rmse": grouped.val_rmse,
        "base_tolerance": base,
        "adjusted_tolerance": adjusted,
        "noise_scale_proxy": noise_proxy,
        "correlation_proxy": correlation_proxy,
        "diagnostics_in_scope": in_scope,
        "adequate_shared_starts": adequate,
        "cross_start_diagnostics": diagnostics,
        "signal_scale": signal_scale,
    }


def semisynthetic_records(multiplier: float, noise_calibration: dict, consistency_calibration: dict) -> list[dict]:
    records = []
    for file_index, filename in enumerate(BACKGROUND_FILES):
        path = DATA_DIR / filename
        for construction in CONSTRUCTIONS:
            for drift in DRIFTS:
                for repeat in range(REPEATS):
                    seed = SEED_BASE + file_index * 10000 + CONSTRUCTIONS.index(construction) * 1000 + int(drift * 1000) + repeat
                    observation = build_real_background_observation(path, construction, drift, seed)
                    for budget in LBFGS_BUDGETS:
                        fitted = fit_observation(*observation, seed, budget, multiplier, noise_calibration, consistency_calibration)
                        records.append({
                            "background_file": filename, "construction": construction,
                            "drift": drift, "repeat": repeat, "seed": seed, **fitted,
                        })
                        print(f"stage54 background={filename} construction={construction} drift={drift:.2f} repeat={repeat} budget={budget} decision={fitted['decision']}", flush=True)
    return records


def method_metrics(predictions: list[tuple[str, str]]) -> dict:
    determinate = [(truth, pred) for truth, pred in predictions if pred != "INDETERMINATE"]
    correct = sum(truth == pred for truth, pred in determinate)
    acceptable = [(truth, pred) for truth, pred in predictions if truth == "RETAIN"]
    severe = [(truth, pred) for truth, pred in predictions if truth == "REFUSE"]
    return {
        "coverage": len(determinate) / max(len(predictions), 1),
        "selective_accuracy": correct / max(len(determinate), 1),
        "false_refusal_fraction": sum(pred == "REFUSE" for _, pred in acceptable) / max(len(acceptable), 1),
        "severe_refusal_fraction": sum(pred == "REFUSE" for _, pred in severe) / max(len(severe), 1),
    }


def summarize_semisynthetic(records: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (record["background_file"], record["construction"], record["drift"], record["repeat"], record["seed"])
        grouped[key].append(record)
    pairs = []
    proposed_predictions, uncalibrated_predictions, bic_predictions, single_predictions = [], [], [], []
    for key, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: item["lbfgs_steps"])
        truth = "RETAIN" if key[2] <= 0.05 else "REFUSE"
        proposed, reason = consensus_decision([item["decision_class"] for item in ordered])
        uncalibrated, _ = consensus_decision([item["uncalibrated_class"] for item in ordered])
        bic_counts = Counter(item["bic_only_class"] for item in ordered)
        bic = "REFUSE" if bic_counts["REFUSE"] >= 2 else ("RETAIN" if bic_counts["RETAIN"] >= 2 else "INDETERMINATE")
        single = next(item["decision_class"] for item in ordered if item["lbfgs_steps"] == 160)
        proposed_predictions.append((truth, proposed))
        uncalibrated_predictions.append((truth, uncalibrated))
        bic_predictions.append((truth, bic))
        single_predictions.append((truth, single))
        pairs.append({
            "background_file": key[0], "construction": key[1], "drift": key[2],
            "repeat": key[3], "seed": key[4], "truth_class": truth,
            "proposed_class": proposed, "proposed_reason": reason,
            "uncalibrated_class": uncalibrated, "bic_only_class": bic,
            "single_budget_160_class": single,
        })
    methods = {
        "conditional_consensus": method_metrics(proposed_predictions),
        "uncalibrated_consensus": method_metrics(uncalibrated_predictions),
        "bic_only_consensus": method_metrics(bic_predictions),
        "single_budget_160": method_metrics(single_predictions),
    }
    proposed = methods["conditional_consensus"]
    checks = {
        "complete_real_background_matrix": len(pairs) == 36,
        "proposed_false_refusal_at_most_0_10": proposed["false_refusal_fraction"] <= MAX_FALSE_REFUSAL,
        "proposed_severe_refusal_at_least_0_75": proposed["severe_refusal_fraction"] >= MIN_SEVERE_REFUSAL,
        "proposed_selective_accuracy_at_least_0_85": proposed["selective_accuracy"] >= MIN_SELECTIVE_ACCURACY,
        "proposed_coverage_at_least_0_60": proposed["coverage"] >= MIN_COVERAGE,
        "proposed_not_worse_than_bic_by_more_than_0_05": proposed["selective_accuracy"] + 0.05 >= methods["bic_only_consensus"]["selective_accuracy"],
    }
    return {"pairs": pairs, "methods": methods, "checks": checks, "route_pass": all(checks.values())}


def normalized_blind_curves() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, list[dict]]:
    curves, provenance = [], []
    grid = np.linspace(0.0, 1.0, 65)
    for filename in BLIND_FILES:
        path = DATA_DIR / filename
        time_values, strain, stress = load_relaxation_columns(path)
        threshold = 0.95 * float(np.nanmax(strain))
        eligible = np.flatnonzero(strain >= threshold)
        start = int(eligible[0]) if len(eligible) else int(np.argmax(strain))
        local_time = time_values[start:] - time_values[start]
        local_stress = stress[start:]
        finite = np.isfinite(local_time) & np.isfinite(local_stress)
        local_time, local_stress = local_time[finite], local_stress[finite]
        if len(local_time) < 65 or local_time[-1] <= 0:
            raise ValueError(f"insufficient hold segment in {filename}")
        normalized_time = local_time / local_time[-1]
        sampled = np.interp(grid, normalized_time, local_stress)
        low, high = float(np.min(sampled)), float(np.max(sampled))
        curve = (sampled - low) / max(high - low, 1e-12)
        if curve[0] < curve[-1]:
            curve = 1.0 - curve
        curves.append(curve)
        provenance.append({"file": filename, "sha256": file_sha256(path), "rows_after_stride": len(time_values)})
    observations = torch.tensor(np.column_stack(curves), dtype=DTYPE, device=DEVICE)
    times = torch.linspace(0.0, HORIZON, 65, dtype=DTYPE, device=DEVICE)
    train = torch.tensor(np.arange(1, 48), dtype=torch.long, device=DEVICE)
    val = torch.tensor(np.arange(48, 65), dtype=torch.long, device=DEVICE)
    labels = np.repeat(np.arange(BLOCKS), len(BLIND_FILES) // BLOCKS)
    return times, observations, train, val, labels, provenance


def blind_audit(multiplier: float, noise_calibration: dict, consistency_calibration: dict) -> dict:
    observation = normalized_blind_curves()
    times, observations, train, val, labels, provenance = observation
    records = []
    seed = 231000
    for budget in LBFGS_BUDGETS:
        records.append({"budget": budget, **fit_observation(times, observations, train, val, labels, seed, budget, multiplier, noise_calibration, consistency_calibration)})
    classes = [record["decision_class"] if record["diagnostics_in_scope"] else "INDETERMINATE" for record in records]
    consensus, reason = consensus_decision(classes)
    return {
        "truth_available": False,
        "provenance": provenance,
        "records": records,
        "scope_aware_classes": classes,
        "consensus_class": consensus,
        "consensus_reason": reason,
        "interpretation": "Diagnostic only; no mechanism-recovery claim is made without ground truth.",
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "real_background_mechanism_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    methods = payload["semisynthetic_summary"]["methods"]
    lines = ["# Real-background mechanism audit", "", f"Route pass: **{payload['semisynthetic_summary']['route_pass']}**.", "", "| Method | Coverage | Selective accuracy | False refusal | Severe refusal |", "|:---|---:|---:|---:|---:|"]
    for name, row in methods.items():
        lines.append(f"| {name} | {row['coverage']:.3f} | {row['selective_accuracy']:.3f} | {row['false_refusal_fraction']:.3f} | {row['severe_refusal_fraction']:.3f} |")
    lines.extend(["", f"Blind real-data consensus: `{payload['blind_audit']['consensus_class']}` ({payload['blind_audit']['consensus_reason']}).", "", "The blind result is diagnostic and has no truth label."])
    (RESULTS / "real_background_mechanism_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    stage52 = json.loads(STAGE52_RESULT.read_text(encoding="utf-8"))["calibration"]
    multiplier = stage52["validation_tolerance_multiplier"]
    noise_calibration, consistency_calibration = load_frozen_stage48()
    records = semisynthetic_records(multiplier, noise_calibration, consistency_calibration)
    summary = summarize_semisynthetic(records)
    blind = blind_audit(multiplier, noise_calibration, consistency_calibration)
    payload = {
        "experiment": "real_background_mechanism_audit",
        "device": str(DEVICE), "dtype": str(DTYPE),
        "protocol": {
            "public_data_directory": str(DATA_DIR), "background_files": list(BACKGROUND_FILES),
            "blind_files": list(BLIND_FILES), "row_stride": ROW_STRIDE,
            "noise_std": NOISE_STD, "repeats": REPEATS, "budgets": list(LBFGS_BUDGETS),
            "stage52_reused_without_recalibration": True,
        },
        "background_provenance": [{"file": name, "sha256": file_sha256(DATA_DIR / name)} for name in BACKGROUND_FILES],
        "semisynthetic_records": records,
        "semisynthetic_summary": summary,
        "blind_audit": blind,
        "exit_rule": {"failure_action": "retain a reliability/abstention result; do not claim real-data mechanism recovery"},
    }
    write_outputs(payload)
    print(json.dumps({"methods": summary["methods"], "checks": summary["checks"], "route_pass": summary["route_pass"], "blind": {"class": blind["consensus_class"], "reason": blind["consensus_reason"]}}, indent=2))


if __name__ == "__main__":
    main()
