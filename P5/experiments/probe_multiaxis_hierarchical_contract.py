"""Stage 55: frozen multi-axis hierarchical decision audit.

This stage consumes the locked Stage 53 and Stage 54 fit records.  It does not
refit a model or tune a threshold.  Numerical eligibility, predictive
validation, and structural BIC evidence remain separate axes.  A binary
mechanism decision is emitted only when the eligible predictive and structural
axes agree; disagreement is an auditable abstention.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from probe_budget_consensus_abstention import consensus_decision
from probe_noise_aware_sharing_gate import BIC_EVIDENCE_LIMIT


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STAGE53_RESULT = RESULTS / "conditional_contract_transfer.json"
STAGE54_RESULT = RESULTS / "real_background_mechanism_audit.json"

MIN_STABLE_BUDGET_VOTES = 2
MIN_ADEQUATE_STARTS = 2

MAX_FALSE_REFUSAL = 0.10
MIN_SEVERE_REFUSAL = 0.75
MIN_SELECTIVE_ACCURACY = 0.85
MIN_COVERAGE = 0.60


def structural_class(record: dict) -> str:
    support = record.get("group_bic_support")
    if support is None or not math.isfinite(float(support)):
        return "INDETERMINATE"
    return "REFUSE" if float(support) >= BIC_EVIDENCE_LIMIT else "RETAIN"


def hierarchical_decision(
    numerical_eligible: bool,
    validation_axis: str,
    structural_axis: str,
) -> tuple[str, str]:
    if not numerical_eligible:
        return "INDETERMINATE", "NUMERICAL_AXIS_INELIGIBLE"
    if validation_axis not in {"RETAIN", "REFUSE"}:
        return "INDETERMINATE", "VALIDATION_AXIS_ABSTAINS"
    if structural_axis not in {"RETAIN", "REFUSE"}:
        return "INDETERMINATE", "STRUCTURAL_AXIS_ABSTAINS"
    if validation_axis != structural_axis:
        return "INDETERMINATE", "CROSS_AXIS_CONFLICT"
    return validation_axis, "AXES_AGREE"


def decision_metrics(pairs: list[dict], field: str) -> dict:
    predictions = [(pair["truth_class"], pair[field]) for pair in pairs]
    determinate = [(truth, pred) for truth, pred in predictions if pred != "INDETERMINATE"]
    acceptable = [(truth, pred) for truth, pred in predictions if truth == "RETAIN"]
    severe = [(truth, pred) for truth, pred in predictions if truth == "REFUSE"]
    return {
        "trials": len(predictions),
        "coverage": len(determinate) / max(len(predictions), 1),
        "selective_accuracy": sum(truth == pred for truth, pred in determinate) / max(len(determinate), 1),
        "false_refusal_fraction": sum(pred == "REFUSE" for _, pred in acceptable) / max(len(acceptable), 1),
        "severe_refusal_fraction": sum(pred == "REFUSE" for _, pred in severe) / max(len(severe), 1),
        "indeterminate_fraction": sum(pred == "INDETERMINATE" for _, pred in predictions) / max(len(predictions), 1),
    }


def aggregate_records(records: list[dict], source: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if source == "gaussian_transfer":
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

    pairs = []
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        ordered = sorted(items, key=lambda item: item["lbfgs_steps"])
        validation_axis, validation_reason = consensus_decision(
            [item["decision_class"] for item in ordered]
        )
        structural_axis, structural_reason = consensus_decision(
            [structural_class(item) for item in ordered]
        )
        scope_field = "diagnostics_in_calibration_scope" if source == "gaussian_transfer" else "diagnostics_in_scope"
        scope_all = all(bool(item[scope_field]) for item in ordered)
        stable_votes = sum(
            int(item.get("adequate_shared_starts", 0)) >= MIN_ADEQUATE_STARTS
            for item in ordered
        )
        numerical_eligible = scope_all and stable_votes >= MIN_STABLE_BUDGET_VOTES
        hierarchical, hierarchical_reason = hierarchical_decision(
            numerical_eligible, validation_axis, structural_axis
        )
        drift = float(ordered[0]["drift"])
        pair = {
            "source": source,
            "drift": drift,
            "truth_class": "RETAIN" if drift <= 0.05 else "REFUSE",
            "validation_axis": validation_axis,
            "validation_reason": validation_reason,
            "structural_axis": structural_axis,
            "structural_reason": structural_reason,
            "numerical_eligible": numerical_eligible,
            "scope_all": scope_all,
            "stable_budget_votes": stable_votes,
            "hierarchical_class": hierarchical,
            "hierarchical_reason": hierarchical_reason,
        }
        if source == "gaussian_transfer":
            pair.update({
                "channels": ordered[0]["channels"],
                "noise_std": ordered[0]["noise_std"],
                "noise_domain": ordered[0]["noise_domain"],
                "construction": ordered[0]["construction"],
                "repeat": ordered[0]["repeat"],
                "seed": ordered[0]["seed"],
            })
        else:
            pair.update({
                "background_file": ordered[0]["background_file"],
                "construction": ordered[0]["construction"],
                "repeat": ordered[0]["repeat"],
                "seed": ordered[0]["seed"],
            })
        pairs.append(pair)
    return pairs


def summarize_domain(pairs: list[dict]) -> dict:
    return {
        "validation_axis": decision_metrics(pairs, "validation_axis"),
        "structural_axis": decision_metrics(pairs, "structural_axis"),
        "hierarchical": decision_metrics(pairs, "hierarchical_class"),
    }


def frozen_checks(metrics: dict, prefix: str) -> dict:
    return {
        f"{prefix}_false_refusal_at_most_0_10": metrics["false_refusal_fraction"] <= MAX_FALSE_REFUSAL,
        f"{prefix}_severe_refusal_at_least_0_75": metrics["severe_refusal_fraction"] >= MIN_SEVERE_REFUSAL,
        f"{prefix}_selective_accuracy_at_least_0_85": metrics["selective_accuracy"] >= MIN_SELECTIVE_ACCURACY,
        f"{prefix}_coverage_at_least_0_60": metrics["coverage"] >= MIN_COVERAGE,
    }


def build_payload() -> dict:
    stage53 = json.loads(STAGE53_RESULT.read_text(encoding="utf-8"))
    stage54 = json.loads(STAGE54_RESULT.read_text(encoding="utf-8"))
    gaussian_pairs = aggregate_records(stage53["records"], "gaussian_transfer")
    gaussian_core = [pair for pair in gaussian_pairs if pair["noise_domain"] == "core"]
    gaussian_stress = [pair for pair in gaussian_pairs if pair["noise_domain"] == "stress"]
    real_pairs = aggregate_records(stage54["semisynthetic_records"], "real_background")

    summaries = {
        "gaussian_core": summarize_domain(gaussian_core),
        "gaussian_stress": summarize_domain(gaussian_stress),
        "real_background": summarize_domain(real_pairs),
    }
    checks = {
        "complete_gaussian_matrix": len(gaussian_pairs) == 72,
        "complete_real_background_matrix": len(real_pairs) == 36,
        **frozen_checks(summaries["gaussian_core"]["hierarchical"], "gaussian_core"),
        **frozen_checks(summaries["real_background"]["hierarchical"], "real_background"),
    }
    return {
        "experiment": "stage55_frozen_multiaxis_hierarchical_contract",
        "protocol": {
            "minimum_stable_budget_votes": MIN_STABLE_BUDGET_VOTES,
            "minimum_adequate_starts": MIN_ADEQUATE_STARTS,
            "decision_rule": "binary output only when numerical eligibility holds and validation/BIC axes agree",
            "thresholds_reused_from_stage54": {
                "maximum_false_refusal": MAX_FALSE_REFUSAL,
                "minimum_severe_refusal": MIN_SEVERE_REFUSAL,
                "minimum_selective_accuracy": MIN_SELECTIVE_ACCURACY,
                "minimum_coverage": MIN_COVERAGE,
            },
        },
        "pairs": {"gaussian_transfer": gaussian_pairs, "real_background": real_pairs},
        "summary": summaries,
        "checks": checks,
        "route_pass": all(checks.values()),
        "exit_rule": {
            "failure_action": "do not tune another scalar threshold; retain separate axes and redesign the decision architecture",
        },
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "multiaxis_hierarchical_contract.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Frozen multi-axis hierarchical contract audit",
        "",
        f"Route pass: **{payload['route_pass']}**.",
        "",
        "| Domain | Method | Coverage | Selective accuracy | False refusal | Severe refusal |",
        "|:---|:---|---:|---:|---:|---:|",
    ]
    labels = {
        "gaussian_core": "Gaussian core",
        "gaussian_stress": "Gaussian stress",
        "real_background": "Real residual background",
    }
    for domain, methods in payload["summary"].items():
        for method, metrics in methods.items():
            lines.append(
                f"| {labels[domain]} | {method} | {metrics['coverage']:.3f} | "
                f"{metrics['selective_accuracy']:.3f} | {metrics['false_refusal_fraction']:.3f} | "
                f"{metrics['severe_refusal_fraction']:.3f} |"
            )
    lines.extend([
        "",
        "The decision architecture and success criteria were frozen before this locked-output audit. No model was refit and no threshold was recalibrated.",
    ])
    (RESULTS / "multiaxis_hierarchical_contract.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    payload = build_payload()
    write_outputs(payload)
    print(json.dumps({"summary": payload["summary"], "checks": payload["checks"], "route_pass": payload["route_pass"]}, indent=2))


if __name__ == "__main__":
    main()
