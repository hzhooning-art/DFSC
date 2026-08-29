"""Stage 57: morphology-calibrated asymmetric evidence hierarchy.

Seven public elastomer residual morphologies calibrate validation and
structural evidence envelopes.  The three Stage 54 residual backgrounds stay
locked for evaluation and never influence either threshold or the rule.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from probe_asymmetric_evidence_hierarchy import repeated_exceedance_score
from probe_budget_consensus_abstention import consensus_decision
from probe_conditional_contract_transfer import CONSTRUCTIONS, DRIFTS
from probe_extended_refinement_transfer import load_frozen_stage48
from probe_multiaxis_hierarchical_contract import (
    MAX_FALSE_REFUSAL,
    MIN_ADEQUATE_STARTS,
    MIN_COVERAGE,
    MIN_SELECTIVE_ACCURACY,
    MIN_SEVERE_REFUSAL,
    MIN_STABLE_BUDGET_VOTES,
    decision_metrics,
    structural_class,
)
from probe_optimizer_budget_stability import LBFGS_BUDGETS
from probe_real_background_mechanism_audit import (
    BACKGROUND_FILES,
    DATA_DIR,
    STAGE52_RESULT,
    build_real_background_observation,
    file_sha256,
    fit_observation,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STAGE54_RESULT = RESULTS / "real_background_mechanism_audit.json"

CALIBRATION_FILES = (
    "Cheetah-relaxation.csv",
    "Dragon Skin 30-relaxation.csv",
    "Ecoflex 00-20-relaxation.csv",
    "Ecoflex 00-50-relaxation.csv",
    "Filaflex 60A-relaxation.csv",
    "FsCO-BMI689-relaxation.csv",
    "Mold Max 14NV-relaxation.csv",
)
CALIBRATION_REPEATS = 1
SEED_BASE = 241000


def repeated_structural_score(records: list[dict]) -> float | None:
    """Return the second-largest finite grouped-BIC support value."""
    scores = sorted(
        float(record["group_bic_support"])
        for record in records
        if record.get("group_bic_support") is not None
        and math.isfinite(float(record["group_bic_support"]))
    )
    return scores[-2] if len(scores) >= 2 else None


def collect_calibration_records(
    multiplier: float,
    noise_calibration: dict,
    consistency_calibration: dict,
) -> list[dict]:
    records = []
    for file_index, filename in enumerate(CALIBRATION_FILES):
        path = DATA_DIR / filename
        for construction_index, construction in enumerate(CONSTRUCTIONS):
            for drift in DRIFTS:
                for repeat in range(CALIBRATION_REPEATS):
                    seed = (
                        SEED_BASE
                        + file_index * 10000
                        + construction_index * 1000
                        + int(drift * 1000)
                        + repeat
                    )
                    observation = build_real_background_observation(
                        path, construction, drift, seed
                    )
                    for budget in LBFGS_BUDGETS:
                        fitted = fit_observation(
                            *observation,
                            seed,
                            budget,
                            multiplier,
                            noise_calibration,
                            consistency_calibration,
                        )
                        records.append({
                            "background_file": filename,
                            "construction": construction,
                            "drift": drift,
                            "repeat": repeat,
                            "seed": seed,
                            **fitted,
                        })
                        print(
                            "stage57 "
                            f"background={filename} construction={construction} "
                            f"drift={drift:.2f} budget={budget} "
                            f"decision={fitted['decision_class']}",
                            flush=True,
                        )
    return records


def grouped_records(records: list[dict]) -> list[list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (
            record["background_file"],
            record["construction"],
            record["drift"],
            record["repeat"],
            record["seed"],
        )
        grouped[key].append(record)
    return [
        sorted(items, key=lambda item: item["lbfgs_steps"])
        for _, items in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def pair_features(items: list[dict]) -> dict:
    validation_axis, validation_reason = consensus_decision(
        [item["decision_class"] for item in items]
    )
    structural_classes = [structural_class(item) for item in items]
    structural_axis, structural_reason = consensus_decision(structural_classes)
    stable_votes = sum(
        int(item.get("adequate_shared_starts", 0)) >= MIN_ADEQUATE_STARTS
        for item in items
    )
    scope_all = all(bool(item["diagnostics_in_scope"]) for item in items)
    drift = float(items[0]["drift"])
    return {
        "background_file": items[0]["background_file"],
        "construction": items[0]["construction"],
        "drift": drift,
        "repeat": items[0]["repeat"],
        "seed": items[0]["seed"],
        "truth_class": "RETAIN" if drift <= 0.05 else "REFUSE",
        "validation_axis": validation_axis,
        "validation_reason": validation_reason,
        "structural_axis": structural_axis,
        "structural_reason": structural_reason,
        "structural_refuse_votes": structural_classes.count("REFUSE"),
        "validation_score": repeated_exceedance_score(items),
        "structural_score": repeated_structural_score(items),
        "scope_all": scope_all,
        "stable_budget_votes": stable_votes,
        "numerical_eligible": (
            scope_all and stable_votes >= MIN_STABLE_BUDGET_VOTES
        ),
    }


def calibrate_thresholds(calibration_pairs: list[dict]) -> dict:
    acceptable = [
        pair for pair in calibration_pairs if pair["truth_class"] == "RETAIN"
    ]
    validation_scores = [
        float(pair["validation_score"])
        for pair in acceptable
        if pair["validation_score"] is not None
    ]
    structural_scores = [
        float(pair["structural_score"])
        for pair in acceptable
        if pair["structural_score"] is not None
    ]
    if not validation_scores or not structural_scores:
        raise ValueError("acceptable morphology calibration scores are required")
    return {
        "validation_threshold": math.nextafter(max(validation_scores), math.inf),
        "structural_threshold": math.nextafter(max(structural_scores), math.inf),
        "acceptable_validation_max": max(validation_scores),
        "acceptable_structural_max": max(structural_scores),
        "acceptable_pairs": len(acceptable),
        "rule": "next representable float above each maximum acceptable morphology score",
    }


def morphology_decision(
    pair: dict, validation_threshold: float, structural_threshold: float
) -> tuple[str, str]:
    if not pair["numerical_eligible"]:
        return "INDETERMINATE", "NUMERICAL_AXIS_INELIGIBLE"
    if (
        pair["structural_score"] is not None
        and pair["structural_score"] > structural_threshold
    ):
        return "REFUSE", "STRONG_STRUCTURAL_REFUSAL"
    if (
        pair["validation_score"] is not None
        and pair["validation_score"] > validation_threshold
        and pair["structural_refuse_votes"] >= 1
    ):
        return "REFUSE", "STRONG_VALIDATION_WITH_STRUCTURAL_SUPPORT"
    if pair["validation_axis"] == "RETAIN" and pair["structural_axis"] == "RETAIN":
        return "RETAIN", "CONCORDANT_RETENTION"
    return "INDETERMINATE", "INSUFFICIENT_CONCORDANT_EVIDENCE"


def apply_rule(pairs: list[dict], thresholds: dict) -> list[dict]:
    evaluated = []
    for pair in pairs:
        decision, reason = morphology_decision(
            pair,
            thresholds["validation_threshold"],
            thresholds["structural_threshold"],
        )
        evaluated.append({
            **pair,
            "morphology_calibrated_class": decision,
            "morphology_calibrated_reason": reason,
        })
    return evaluated


def frozen_checks(metrics: dict) -> dict:
    return {
        "false_refusal_at_most_0_10": (
            metrics["false_refusal_fraction"] <= MAX_FALSE_REFUSAL
        ),
        "severe_refusal_at_least_0_75": (
            metrics["severe_refusal_fraction"] >= MIN_SEVERE_REFUSAL
        ),
        "selective_accuracy_at_least_0_85": (
            metrics["selective_accuracy"] >= MIN_SELECTIVE_ACCURACY
        ),
        "coverage_at_least_0_60": metrics["coverage"] >= MIN_COVERAGE,
    }


def build_payload(calibration_records: list[dict]) -> dict:
    overlap = sorted(set(CALIBRATION_FILES) & set(BACKGROUND_FILES))
    if overlap:
        raise ValueError(f"calibration/evaluation background overlap: {overlap}")
    stage54 = json.loads(STAGE54_RESULT.read_text(encoding="utf-8"))
    calibration_pairs = [pair_features(items) for items in grouped_records(calibration_records)]
    thresholds = calibrate_thresholds(calibration_pairs)
    calibration_pairs = apply_rule(calibration_pairs, thresholds)
    evaluation_pairs = [
        pair_features(items)
        for items in grouped_records(stage54["semisynthetic_records"])
    ]
    evaluation_pairs = apply_rule(evaluation_pairs, thresholds)
    calibration_metrics = decision_metrics(
        calibration_pairs, "morphology_calibrated_class"
    )
    evaluation_metrics = decision_metrics(
        evaluation_pairs, "morphology_calibrated_class"
    )
    checks = {
        "disjoint_background_files": not overlap,
        "complete_calibration_matrix": len(calibration_pairs) == 42,
        "complete_locked_evaluation_matrix": len(evaluation_pairs) == 36,
        **frozen_checks(evaluation_metrics),
    }
    return {
        "experiment": "stage57_morphology_calibrated_asymmetric_hierarchy",
        "protocol": {
            "calibration_files": list(CALIBRATION_FILES),
            "locked_evaluation_files": list(BACKGROUND_FILES),
            "calibration_repeats": CALIBRATION_REPEATS,
            "budgets": list(LBFGS_BUDGETS),
            "evaluation_records_reused_without_refitting": True,
            "locked_evaluation_labels_not_used_for_calibration": True,
        },
        "provenance": [
            {"file": name, "sha256": file_sha256(DATA_DIR / name)}
            for name in CALIBRATION_FILES
        ],
        "thresholds": thresholds,
        "calibration_records": calibration_records,
        "calibration_pairs": calibration_pairs,
        "evaluation_pairs": evaluation_pairs,
        "calibration_metrics": calibration_metrics,
        "evaluation_metrics": evaluation_metrics,
        "checks": checks,
        "route_pass": all(checks.values()),
        "exit_rule": {
            "failure_action": "do not tune on locked labels; report the frozen failure and redesign the evidence representation",
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "morphology_calibrated_asymmetric_hierarchy.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    calibration = payload["calibration_metrics"]
    evaluation = payload["evaluation_metrics"]
    lines = [
        "# Morphology-calibrated asymmetric evidence hierarchy",
        "",
        f"Route pass: **{payload['route_pass']}**.",
        "",
        "| Domain | Coverage | Selective accuracy | False refusal | Severe refusal |",
        "|:---|---:|---:|---:|---:|",
        f"| Disjoint morphology calibration | {calibration['coverage']:.3f} | {calibration['selective_accuracy']:.3f} | {calibration['false_refusal_fraction']:.3f} | {calibration['severe_refusal_fraction']:.3f} |",
        f"| Locked Stage 54 evaluation | {evaluation['coverage']:.3f} | {evaluation['selective_accuracy']:.3f} | {evaluation['false_refusal_fraction']:.3f} | {evaluation['severe_refusal_fraction']:.3f} |",
        "",
        "Both evidence thresholds were fixed from seven disjoint real residual morphologies before the Stage 54 labels were evaluated.",
    ]
    (RESULTS / "morphology_calibrated_asymmetric_hierarchy.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    stage52 = json.loads(STAGE52_RESULT.read_text(encoding="utf-8"))["calibration"]
    noise_calibration, consistency_calibration = load_frozen_stage48()
    calibration_records = collect_calibration_records(
        stage52["validation_tolerance_multiplier"],
        noise_calibration,
        consistency_calibration,
    )
    payload = build_payload(calibration_records)
    write_outputs(payload)
    print(json.dumps({
        "thresholds": payload["thresholds"],
        "calibration_metrics": payload["calibration_metrics"],
        "evaluation_metrics": payload["evaluation_metrics"],
        "checks": payload["checks"],
        "route_pass": payload["route_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
