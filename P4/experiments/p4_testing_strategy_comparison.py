"""Compare qualification records with progressively stronger test strategies.

The comparison reuses the frozen ten-class fault catalogue from the P4
conformance experiment.  Each strategy sees the same records.  Detection is
reported both per injected instance and per fault class because the repeated
instances within one class are implementation trials, not independent defect
families.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from p4_csi_conformance_validation import base_record, inject, is_rejected, wilson  # noqa: E402


FAULTS = (
    "inadequate_truncation",
    "detached_gradient",
    "silent_dtype_downgrade",
    "batch_crosstalk",
    "silent_cpu_fallback",
    "ood_scope_misuse",
    "unit_mismatch",
    "missing_provenance",
    "unfrozen_scope",
    "insufficient_scope_coverage",
)


def envelope_and_numerical(record: dict) -> bool:
    """A conventional record/schema check plus value and gradient assertions."""
    required = ("schema", "component", "operating_domain", "evidence", "provenance")
    return all(key in record for key in required) and all(
        record.get("evidence", {}).get(key, False)
        for key in ("value_accuracy", "gradient_accuracy")
    )


def numerical_property_suite(record: dict) -> bool:
    """Numerical checks plus common batch, repeatability, and OOD properties."""
    return envelope_and_numerical(record) and all(
        record.get("evidence", {}).get(key, False)
        for key in (
            "batch_shape",
            "batch_independence",
            "repeatability",
            "ood_control",
            "long_horizon",
        )
    )


def execution_aware_suite(record: dict) -> bool:
    """Add execution and unit checks without scope-coverage semantics."""
    requested = record.get("requested_execution", {})
    observed = record.get("observed_execution", {})
    return numerical_property_suite(record) and all(
        record.get("evidence", {}).get(key, False)
        for key in ("dtype_conformance", "device_local", "unit_consistency")
    ) and requested.get("dtype") == observed.get("dtype") and requested.get("device") == observed.get("device")


STRATEGIES = (
    ("envelope_value_gradient", envelope_and_numerical, 3),
    ("numerical_property_suite", numerical_property_suite, 8),
    ("execution_aware_suite", execution_aware_suite, 13),
    ("executable_evidence_record", lambda record: not is_rejected(record), 17),
)


def evaluate_strategy(name: str, accepts, declared_checks: int) -> dict:
    matrix = []
    for fault in FAULTS:
        detections = []
        for seed in range(20):
            record = inject(base_record(f"strategy-{name}-{fault}-{seed}"), fault)
            detections.append(not accepts(record))
        matrix.append(
            {
                "fault": fault,
                "detected": int(sum(detections)),
                "injections": len(detections),
                "detected_class": bool(all(detections)),
            }
        )
    clean = [not accepts(base_record(f"strategy-{name}-clean-{seed}")) for seed in range(40)]
    detected = sum(row["detected"] for row in matrix)
    classes = sum(row["detected_class"] for row in matrix)
    false_rejections = int(sum(clean))
    return {
        "strategy": name,
        "declared_check_count": declared_checks,
        "fault_instances": 200,
        "detected_instances": detected,
        "instance_detection_rate": detected / 200,
        "instance_detection_wilson_95": wilson(detected, 200),
        "fault_classes": len(FAULTS),
        "fully_detected_fault_classes": classes,
        "fault_class_coverage": classes / len(FAULTS),
        "clean_records": len(clean),
        "false_rejections": false_rejections,
        "false_rejection_rate": false_rejections / len(clean),
        "false_rejection_wilson_95": wilson(false_rejections, len(clean)),
        "catalogue": matrix,
    }


def main() -> None:
    strategies = [evaluate_strategy(*strategy) for strategy in STRATEGIES]
    for previous, current in zip(strategies[:-1], strategies[1:]):
        current["increment_over_previous"] = {
            "previous_strategy": previous["strategy"],
            "additional_checks": current["declared_check_count"] - previous["declared_check_count"],
            "additional_fault_classes": current["fully_detected_fault_classes"] - previous["fully_detected_fault_classes"],
            "additional_detected_instances": current["detected_instances"] - previous["detected_instances"],
        }
    payload = {
        "schema": "DFSC-P4-Testing-Strategy-Comparison-v1",
        "design": "paired frozen fault catalogue with 20 implementation trials per class",
        "primary_unit": "fault class",
        "strategies": strategies,
        "claim_boundary": (
            "The comparison measures incremental coverage of the declared ten-class catalogue. "
            "It is not an estimate of field-defect prevalence or proof that the baselines exhaust "
            "unit, property-based, or metamorphic testing practice."
        ),
    }
    output = RESULTS / "p4_testing_strategy_comparison.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
