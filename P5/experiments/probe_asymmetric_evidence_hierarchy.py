"""Stage 56: disjointly calibrated asymmetric evidence hierarchy.

Stage 53 Gaussian-core records form the development/calibration set.  The
locked Stage 54 real-residual records are evaluated only after the strong
validation threshold is fixed.  Retention requires agreement between the
validation and structural axes.  Refusal is allowed by structural consensus,
or by repeatable strong validation exceedance supported by at least one BIC
refusal vote.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from probe_budget_consensus_abstention import consensus_decision
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


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STAGE53_RESULT = RESULTS / "conditional_contract_transfer.json"
STAGE54_RESULT = RESULTS / "real_background_mechanism_audit.json"


def repeated_exceedance_score(records: list[dict]) -> float | None:
    """Return the second-largest finite validation/tolerance ratio."""
    scores = sorted(
        float(record["shared_val_rmse"]) / float(record["adjusted_tolerance"])
        for record in records
        if record.get("shared_val_rmse") is not None
        and record.get("adjusted_tolerance") not in (None, 0)
        and math.isfinite(float(record["shared_val_rmse"]))
        and math.isfinite(float(record["adjusted_tolerance"]))
    )
    return scores[-2] if len(scores) >= 2 else None


def asymmetric_decision(
    numerical_eligible: bool,
    validation_axis: str,
    structural_axis: str,
    structural_refuse_votes: int,
    exceedance_score: float | None,
    strong_threshold: float,
) -> tuple[str, str]:
    if not numerical_eligible:
        return "INDETERMINATE", "NUMERICAL_AXIS_INELIGIBLE"
    if structural_axis == "REFUSE":
        return "REFUSE", "STRUCTURAL_CONSENSUS_REFUSAL"
    if (
        validation_axis == "REFUSE"
        and structural_refuse_votes >= 1
        and exceedance_score is not None
        and exceedance_score > strong_threshold
    ):
        return "REFUSE", "STRONG_VALIDATION_WITH_STRUCTURAL_SUPPORT"
    if validation_axis == "RETAIN" and structural_axis == "RETAIN":
        return "RETAIN", "CONCORDANT_RETENTION"
    return "INDETERMINATE", "INSUFFICIENT_CONCORDANT_EVIDENCE"


def group_records(records: list[dict], source: str) -> list[list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if source == "gaussian_core":
            if record["noise_domain"] != "core":
                continue
            key = (
                record["channels"], record["noise_std"], record["construction"],
                record["drift"], record["repeat"], record["seed"],
            )
        else:
            key = (
                record["background_file"], record["construction"], record["drift"],
                record["repeat"], record["seed"],
            )
        grouped[key].append(record)
    return [sorted(items, key=lambda item: item["lbfgs_steps"]) for _, items in sorted(grouped.items(), key=lambda item: str(item[0]))]


def pair_features(items: list[dict], source: str) -> dict:
    validation_axis, validation_reason = consensus_decision(
        [item["decision_class"] for item in items]
    )
    structural_classes = [structural_class(item) for item in items]
    structural_axis, structural_reason = consensus_decision(structural_classes)
    scope_field = "diagnostics_in_calibration_scope" if source == "gaussian_core" else "diagnostics_in_scope"
    scope_all = all(bool(item[scope_field]) for item in items)
    stable_votes = sum(
        int(item.get("adequate_shared_starts", 0)) >= MIN_ADEQUATE_STARTS
        for item in items
    )
    drift = float(items[0]["drift"])
    pair = {
        "drift": drift,
        "truth_class": "RETAIN" if drift <= 0.05 else "REFUSE",
        "validation_axis": validation_axis,
        "validation_reason": validation_reason,
        "structural_axis": structural_axis,
        "structural_reason": structural_reason,
        "structural_refuse_votes": structural_classes.count("REFUSE"),
        "scope_all": scope_all,
        "stable_budget_votes": stable_votes,
        "numerical_eligible": scope_all and stable_votes >= MIN_STABLE_BUDGET_VOTES,
        "repeated_exceedance_score": repeated_exceedance_score(items),
    }
    if source == "gaussian_core":
        pair.update({
            "channels": items[0]["channels"],
            "noise_std": items[0]["noise_std"],
            "construction": items[0]["construction"],
            "repeat": items[0]["repeat"],
            "seed": items[0]["seed"],
        })
    else:
        pair.update({
            "background_file": items[0]["background_file"],
            "construction": items[0]["construction"],
            "repeat": items[0]["repeat"],
            "seed": items[0]["seed"],
        })
    return pair


def calibrate_threshold(development_pairs: list[dict]) -> dict:
    acceptable_scores = [
        pair["repeated_exceedance_score"]
        for pair in development_pairs
        if pair["truth_class"] == "RETAIN"
        and pair["repeated_exceedance_score"] is not None
    ]
    severe_scores = [
        pair["repeated_exceedance_score"]
        for pair in development_pairs
        if pair["truth_class"] == "REFUSE"
        and pair["repeated_exceedance_score"] is not None
    ]
    if not acceptable_scores or not severe_scores:
        raise ValueError("calibration requires acceptable and severe development scores")
    threshold = math.nextafter(max(acceptable_scores), math.inf)
    return {
        "strong_validation_threshold": threshold,
        "acceptable_score_max": max(acceptable_scores),
        "severe_score_min": min(severe_scores),
        "strict_development_gap": max(acceptable_scores) < min(severe_scores),
        "acceptable_pairs": len(acceptable_scores),
        "severe_pairs": len(severe_scores),
        "calibration_rule": "next representable float above maximum acceptable two-budget exceedance score",
    }


def apply_rule(pairs: list[dict], threshold: float) -> list[dict]:
    evaluated = []
    for pair in pairs:
        decision, reason = asymmetric_decision(
            pair["numerical_eligible"],
            pair["validation_axis"],
            pair["structural_axis"],
            pair["structural_refuse_votes"],
            pair["repeated_exceedance_score"],
            threshold,
        )
        evaluated.append({**pair, "asymmetric_class": decision, "asymmetric_reason": reason})
    return evaluated


def frozen_checks(metrics: dict) -> dict:
    return {
        "false_refusal_at_most_0_10": metrics["false_refusal_fraction"] <= MAX_FALSE_REFUSAL,
        "severe_refusal_at_least_0_75": metrics["severe_refusal_fraction"] >= MIN_SEVERE_REFUSAL,
        "selective_accuracy_at_least_0_85": metrics["selective_accuracy"] >= MIN_SELECTIVE_ACCURACY,
        "coverage_at_least_0_60": metrics["coverage"] >= MIN_COVERAGE,
    }


def build_payload() -> dict:
    stage53 = json.loads(STAGE53_RESULT.read_text(encoding="utf-8"))
    stage54 = json.loads(STAGE54_RESULT.read_text(encoding="utf-8"))
    development = [pair_features(items, "gaussian_core") for items in group_records(stage53["records"], "gaussian_core")]
    calibration = calibrate_threshold(development)
    development = apply_rule(development, calibration["strong_validation_threshold"])
    evaluation = [pair_features(items, "real_background") for items in group_records(stage54["semisynthetic_records"], "real_background")]
    evaluation = apply_rule(evaluation, calibration["strong_validation_threshold"])
    development_metrics = decision_metrics(development, "asymmetric_class")
    evaluation_metrics = decision_metrics(evaluation, "asymmetric_class")
    checks = {
        "complete_development_matrix": len(development) == 48,
        "complete_locked_evaluation_matrix": len(evaluation) == 36,
        "calibration_has_strict_score_gap": calibration["strict_development_gap"],
        **frozen_checks(evaluation_metrics),
    }
    return {
        "experiment": "stage56_disjoint_asymmetric_evidence_hierarchy",
        "protocol": {
            "development_source": "Stage 53 Gaussian core only",
            "locked_evaluation_source": "Stage 54 real residual backgrounds",
            "retention_rule": "validation and structural axes must both retain",
            "refusal_rule": "structural consensus, or repeated strong validation exceedance with at least one structural refusal vote",
            "minimum_stable_budget_votes": MIN_STABLE_BUDGET_VOTES,
            "minimum_adequate_starts": MIN_ADEQUATE_STARTS,
        },
        "calibration": calibration,
        "development_pairs": development,
        "evaluation_pairs": evaluation,
        "development_metrics": development_metrics,
        "evaluation_metrics": evaluation_metrics,
        "checks": checks,
        "route_pass": all(checks.values()),
        "exit_rule": {
            "failure_action": "do not tune on real-background labels; redesign the evidence representation or obtain new disjoint calibration data",
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "asymmetric_evidence_hierarchy.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    dev, real = payload["development_metrics"], payload["evaluation_metrics"]
    lines = [
        "# Disjointly calibrated asymmetric evidence hierarchy",
        "",
        f"Route pass: **{payload['route_pass']}**.",
        f"Frozen strong-validation threshold: {payload['calibration']['strong_validation_threshold']:.6f}.",
        "",
        "| Domain | Coverage | Selective accuracy | False refusal | Severe refusal |",
        "|:---|---:|---:|---:|---:|",
        f"| Gaussian development | {dev['coverage']:.3f} | {dev['selective_accuracy']:.3f} | {dev['false_refusal_fraction']:.3f} | {dev['severe_refusal_fraction']:.3f} |",
        f"| Locked real background | {real['coverage']:.3f} | {real['selective_accuracy']:.3f} | {real['false_refusal_fraction']:.3f} | {real['severe_refusal_fraction']:.3f} |",
        "",
        "The real-background labels were not used to calibrate the threshold or choose the decision rule.",
    ]
    (RESULTS / "asymmetric_evidence_hierarchy.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    payload = build_payload()
    write_outputs(payload)
    print(json.dumps({
        "calibration": payload["calibration"],
        "development_metrics": payload["development_metrics"],
        "evaluation_metrics": payload["evaluation_metrics"],
        "checks": payload["checks"],
        "route_pass": payload["route_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
