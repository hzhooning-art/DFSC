"""Stage 60: frozen-rule scope map under harder observation regimes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probe_conditional_contract_transfer import CONSTRUCTIONS, DRIFTS
from probe_extended_refinement_transfer import load_frozen_stage48
from probe_external_multiphysics_confirmation import build_observation, external_backgrounds
from probe_morphology_calibrated_asymmetric_hierarchy import (
    apply_rule,
    frozen_checks,
    grouped_records,
    pair_features,
)
from probe_multiaxis_hierarchical_contract import decision_metrics
from probe_optimizer_budget_stability import LBFGS_BUDGETS
from probe_real_background_mechanism_audit import STAGE52_RESULT, fit_observation


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STAGE57 = RESULTS / "morphology_calibrated_asymmetric_hierarchy.json"
STAGE58 = RESULTS / "external_multiphysics_confirmation.json"
PARTIAL = RESULTS / "external_scope_boundary_map.partial.json"
SOURCES = ("brain_corona_radiata", "copper_100C", "steel_270MPa_repeats")
REGIMES = {
    "noise_x2": {"noise_multiplier": 2.0, "train_size": 34, "horizon_multiplier": 1.0},
    "sparse_train": {"noise_multiplier": 1.0, "train_size": 18, "horizon_multiplier": 1.0},
    "short_window": {"noise_multiplier": 1.0, "train_size": 34, "horizon_multiplier": 0.45},
}
SEED_BASE = 260000


def collect_records(backgrounds: dict[str, np.ndarray]) -> list[dict]:
    stage52 = json.loads(STAGE52_RESULT.read_text(encoding="utf-8"))["calibration"]
    multiplier = stage52["validation_tolerance_multiplier"]
    noise_calibration, consistency_calibration = load_frozen_stage48()
    records = json.loads(PARTIAL.read_text(encoding="utf-8")) if PARTIAL.exists() else []
    complete = {
        (row["regime"], row["background_file"], row["construction"], float(row["drift"]), int(row["budget"]))
        for row in records
    }
    for regime_index, (regime, settings) in enumerate(REGIMES.items()):
        for source_index, source in enumerate(SOURCES):
            background = backgrounds[source]
            for construction_index, construction in enumerate(CONSTRUCTIONS):
                for drift in DRIFTS:
                    seed = SEED_BASE + regime_index * 100000 + source_index * 10000 + construction_index * 1000 + int(drift * 1000)
                    observation = build_observation(background, construction, drift, seed, **settings)
                    for budget in LBFGS_BUDGETS:
                        key = (regime, source, construction, float(drift), int(budget))
                        if key in complete:
                            continue
                        fitted = fit_observation(
                            *observation, seed, budget, multiplier,
                            noise_calibration, consistency_calibration,
                        )
                        records.append({
                            "regime": regime,
                            "background_file": source,
                            "construction": construction,
                            "drift": drift,
                            "repeat": 0,
                            "seed": seed,
                            "budget": int(budget),
                            **fitted,
                        })
                        PARTIAL.write_text(json.dumps(records, indent=2), encoding="utf-8")
                        complete.add(key)
                        print(f"stage60 regime={regime} source={source} construction={construction} drift={drift:.2f} budget={budget}", flush=True)
    return records


def classify(metrics: dict) -> str:
    return "SUPPORTED" if all(frozen_checks(metrics).values()) else "SCOPE_LIMITED"


def build_payload(records: list[dict]) -> dict:
    stage57 = json.loads(STAGE57.read_text(encoding="utf-8"))
    stage58 = json.loads(STAGE58.read_text(encoding="utf-8"))
    thresholds = stage57["thresholds"]
    regimes = {
        "external_baseline": {
            "settings": {"noise_multiplier": 1.0, "train_size": 34, "horizon_multiplier": 1.0},
            "pairs": stage58["pairs"],
            "metrics": stage58["metrics"],
            "status": classify(stage58["metrics"]),
        }
    }
    for name, settings in REGIMES.items():
        selected = [row for row in records if row["regime"] == name]
        pairs = apply_rule([pair_features(items) for items in grouped_records(selected)], thresholds)
        metrics = decision_metrics(pairs, "morphology_calibrated_class")
        regimes[name] = {"settings": settings, "pairs": pairs, "metrics": metrics, "status": classify(metrics)}
    statuses = {row["status"] for row in regimes.values()}
    checks = {
        "stage57_thresholds_unchanged": thresholds == stage58["frozen_thresholds"],
        "complete_stress_matrix_162_fits": len(records) == 162,
        "all_three_physical_sources_per_stress_regime": all(
            len({row["background_file"] for row in records if row["regime"] == regime}) == 3
            for regime in REGIMES
        ),
        "all_regime_metrics_finite": all(
            np.isfinite(list(row["metrics"].values())).all() for row in regimes.values()
        ),
        "baseline_remains_supported": regimes["external_baseline"]["status"] == "SUPPORTED",
    }
    return {
        "experiment": "stage60_external_scope_boundary_map",
        "frozen_thresholds": thresholds,
        "sources": list(SOURCES),
        "regimes": regimes,
        "observed_statuses": sorted(statuses),
        "boundary_detected": len(statuses) > 1,
        "checks": checks,
        "route_pass": all(checks.values()),
        "interpretation_rule": "A scope-limited regime is reported as a boundary result; thresholds are never retuned after observing it.",
    }


def write_outputs(payload: dict) -> None:
    (RESULTS / "external_scope_boundary_map.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# External scope boundary map", "", f"Route pass: **{payload['route_pass']}**.", ""]
    for name, row in payload["regimes"].items():
        metrics = row["metrics"]
        lines.append(
            f"- {name}: **{row['status']}**, coverage={metrics['coverage']:.4f}, "
            f"selective accuracy={metrics['selective_accuracy']:.4f}, "
            f"false refusal={metrics['false_refusal_fraction']:.4f}, "
            f"severe refusal={metrics['severe_refusal_fraction']:.4f}."
        )
    (RESULTS / "external_scope_boundary_map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    backgrounds, _ = external_backgrounds()
    records = collect_records(backgrounds)
    payload = build_payload(records)
    write_outputs(payload)
    if PARTIAL.exists():
        PARTIAL.unlink()
    print(json.dumps({
        "statuses": {name: row["status"] for name, row in payload["regimes"].items()},
        "checks": payload["checks"],
        "boundary_detected": payload["boundary_detected"],
        "route_pass": payload["route_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
