"""Stage 58: frozen-contract confirmation on independent multiphysics data.

Residual morphologies come from brain tissue, copper-alloy, and martensitic-
steel relaxation datasets.  Neither thresholds nor decision logic are changed
from Stage 57.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from probe_conditional_contract_transfer import CONSTRUCTIONS, DRIFTS, score_matrix
from probe_decomposed_tolerance_transfer import (
    BLOCKS, HORIZON, NOISE_CORRELATION, TRUE_CENTRAL_RATES,
)
from probe_extended_refinement_transfer import load_frozen_stage48
from probe_high_dimensional_shared_spectrum import DEVICE, DTYPE, independent_lifted_response
from probe_morphology_calibrated_asymmetric_hierarchy import (
    apply_rule,
    frozen_checks,
    grouped_records,
    pair_features,
)
from probe_multiaxis_hierarchical_contract import decision_metrics
from probe_optimizer_budget_stability import LBFGS_BUDGETS
from probe_real_background_mechanism_audit import (
    CHANNELS, NOISE_STD, STAGE52_RESULT, fit_observation, residual_segment,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXTERNAL = ROOT / "data" / "external"
STAGE57_RESULT = RESULTS / "morphology_calibrated_asymmetric_hierarchy.json"
PARTIAL_RESULT = RESULTS / "external_multiphysics_confirmation.partial.json"
SEED_BASE = 251000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardized_curve_residual(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 30:
        raise ValueError("external curve is too short for residual extraction")
    window = min(101, max(11, (len(values) // 20) | 1))
    if window >= len(values) // 2:
        window = max(5, ((len(values) // 4) | 1))
    smooth = np.convolve(values, np.ones(window) / window, mode="same")
    residual = values[window:-window] - smooth[window:-window]
    residual -= np.median(residual)
    scale = 1.4826 * np.median(np.abs(residual))
    if not np.isfinite(scale) or scale <= 1e-15:
        scale = float(np.std(residual))
    if not np.isfinite(scale) or scale <= 1e-15:
        raise ValueError("external residual is degenerate")
    return residual / scale


def load_brain(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",")[:, 1]
    return standardized_curve_residual(values)


def load_copper(path: Path) -> list[np.ndarray]:
    table = np.loadtxt(path, delimiter=",", comments="#")
    return [standardized_curve_residual(table[:, column]) for column in range(1, 4)]


def load_steel(path: Path) -> np.ndarray:
    curves: list[list[float]] = []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[4:]:
        tokens = [token for token in re.split(r"\t+", line.strip()) if token]
        numeric = []
        for token in tokens:
            try:
                numeric.append(float(token.replace(",", ".")))
            except ValueError:
                continue
        if len(numeric) < 4:
            continue
        pairs = len(numeric) // 2
        while len(curves) < pairs:
            curves.append([])
        for index in range(pairs):
            curves[index].append(numeric[2 * index + 1])
    residuals = [standardized_curve_residual(np.asarray(curve)) for curve in curves if len(curve) >= 30]
    if not residuals:
        raise ValueError(f"no usable steel curves in {path.name}")
    return np.concatenate(residuals)


def external_backgrounds() -> tuple[dict[str, np.ndarray], list[dict]]:
    brain_paths = {
        "brain_corona_radiata": EXTERNAL / "brain_tissue" / "cyclic_relaxation_corona_radiata.csv",
        "brain_visual_cortex": EXTERNAL / "brain_tissue" / "cyclic_relaxation_visual_cortex.csv",
    }
    copper_path = EXTERNAL / "c19010" / "discrete_time-stress-curve.csv"
    steel_paths = {
        "steel_270MPa_repeats": EXTERNAL / "martensitic_steel" / "X20 SRT_270 MPa - relaxation R1 to R7.txt",
        "steel_300MPa_repeats": EXTERNAL / "martensitic_steel" / "X20 SRT_300 MPa - relaxation R1 to R8.txt",
    }
    backgrounds = {name: load_brain(path) for name, path in brain_paths.items()}
    for temperature, residual in zip((20, 100, 150), load_copper(copper_path)):
        backgrounds[f"copper_{temperature}C"] = residual
    backgrounds.update({name: load_steel(path) for name, path in steel_paths.items()})
    provenance = [
        {"source": "brain_tissue", "doi": "10.5281/zenodo.13960486", "file": str(path), "sha256": file_sha256(path)}
        for path in brain_paths.values()
    ]
    provenance.append({"source": "c19010_copper", "doi": "10.5281/zenodo.10796926", "file": str(copper_path), "sha256": file_sha256(copper_path)})
    provenance.extend([
        {"source": "martensitic_steel", "doi": "10.5281/zenodo.14051050", "file": str(path), "sha256": file_sha256(path)}
        for path in steel_paths.values()
    ])
    return backgrounds, provenance


def build_observation(
    background: np.ndarray,
    construction: str,
    drift: float,
    seed: int,
    *,
    noise_multiplier: float = 1.0,
    train_size: int = 34,
    horizon_multiplier: float = 1.0,
):
    rng = np.random.default_rng(seed)
    times = torch.linspace(0.0, HORIZON * horizon_multiplier, 65, dtype=DTYPE, device=DEVICE)
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
        residual_segment(background, 65, int(rng.integers(len(background))))
        for _ in range(CHANNELS)
    ])
    noise = math.sqrt(NOISE_CORRELATION) * common + math.sqrt(1.0 - NOISE_CORRELATION) * independent
    noise /= max(float(np.std(noise)), 1e-12)
    observations = clean + NOISE_STD * noise_multiplier * torch.tensor(noise, dtype=DTYPE, device=DEVICE)
    train = np.sort(rng.choice(np.arange(1, 48), size=train_size, replace=False))
    return (
        times,
        observations,
        torch.tensor(train, dtype=torch.long, device=DEVICE),
        torch.tensor(np.arange(48, 65), dtype=torch.long, device=DEVICE),
        labels,
    )


def collect_records(backgrounds: dict[str, np.ndarray]) -> list[dict]:
    stage52 = json.loads(STAGE52_RESULT.read_text(encoding="utf-8"))["calibration"]
    multiplier = stage52["validation_tolerance_multiplier"]
    noise_calibration, consistency_calibration = load_frozen_stage48()
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = []
    if PARTIAL_RESULT.exists():
        records = json.loads(PARTIAL_RESULT.read_text(encoding="utf-8"))
    completed = {
        (row["background_file"], row["construction"], float(row["drift"]), int(row["budget"]))
        for row in records
    }
    for source_index, (source, background) in enumerate(backgrounds.items()):
        for construction_index, construction in enumerate(CONSTRUCTIONS):
            for drift in DRIFTS:
                seed = SEED_BASE + source_index * 10000 + construction_index * 1000 + int(drift * 1000)
                observation = build_observation(background, construction, drift, seed)
                for budget in LBFGS_BUDGETS:
                    key = (source, construction, float(drift), int(budget))
                    if key in completed:
                        continue
                    fitted = fit_observation(
                        *observation, seed, budget, multiplier,
                        noise_calibration, consistency_calibration,
                    )
                    records.append({
                        "background_file": source,
                        "construction": construction,
                        "drift": drift,
                        "repeat": 0,
                        "seed": seed,
                        "budget": int(budget),
                        **fitted,
                    })
                    PARTIAL_RESULT.write_text(json.dumps(records, indent=2), encoding="utf-8")
                    completed.add(key)
                    print(f"stage58 source={source} construction={construction} drift={drift:.2f} budget={budget} decision={fitted['decision_class']}", flush=True)
    return records


def build_payload(records: list[dict], provenance: list[dict]) -> dict:
    stage57 = json.loads(STAGE57_RESULT.read_text(encoding="utf-8"))
    thresholds = stage57["thresholds"]
    pairs = [pair_features(items) for items in grouped_records(records)]
    evaluated = apply_rule(pairs, thresholds)
    metrics = decision_metrics(evaluated, "morphology_calibrated_class")
    checks = {
        "frozen_stage57_thresholds": thresholds == stage57["thresholds"],
        "three_independent_physical_sources": len({row["source"] for row in provenance}) == 3,
        "complete_external_matrix": len(evaluated) == 42,
        **frozen_checks(metrics),
    }
    return {
        "experiment": "stage58_external_multiphysics_confirmation",
        "protocol": {
            "thresholds_and_rule_frozen_from_stage57": True,
            "external_sources": ["brain_tissue", "c19010_copper", "martensitic_steel"],
            "budgets": list(LBFGS_BUDGETS),
            "post_hoc_adjustment_forbidden": True,
        },
        "provenance": provenance,
        "frozen_thresholds": thresholds,
        "records": records,
        "pairs": evaluated,
        "metrics": metrics,
        "checks": checks,
        "route_pass": all(checks.values()),
        "exit_rule": {"failure_action": "do not recalibrate; restrict the declared morphology domain or retain a negative external-transfer result"},
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "external_multiphysics_confirmation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics = payload["metrics"]
    lines = [
        "# External multiphysics confirmation",
        "",
        f"Route pass: **{payload['route_pass']}**.",
        "",
        f"- Coverage: {metrics['coverage']:.4f}",
        f"- Selective accuracy: {metrics['selective_accuracy']:.4f}",
        f"- False refusal: {metrics['false_refusal_fraction']:.4f}",
        f"- Severe refusal: {metrics['severe_refusal_fraction']:.4f}",
        "",
        "Stage 57 thresholds and decision logic were frozen before these external sources were evaluated.",
    ]
    (RESULTS / "external_multiphysics_confirmation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    backgrounds, provenance = external_backgrounds()
    records = collect_records(backgrounds)
    payload = build_payload(records, provenance)
    write_outputs(payload)
    if PARTIAL_RESULT.exists():
        PARTIAL_RESULT.unlink()
    print(json.dumps({"metrics": payload["metrics"], "checks": payload["checks"], "route_pass": payload["route_pass"]}, indent=2))


if __name__ == "__main__":
    main()
