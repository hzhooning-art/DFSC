"""Validate the CS&I-oriented conformance and interoperability claims.

The experiment checks two distinct properties:
1. Python API, CLI, and key-order permutations produce the same canonical
   outcome; v1/v2 migration preserves evidence but requires v3 requalification.
2. A predeclared fault catalogue is detected without rejecting clean records.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from dfsc_protocol import CONFORMANCE_SCHEMA, canonical_json, evaluate_conformance, record_digest  # noqa: E402


def base_record(run_id: str, schema: str = CONFORMANCE_SCHEMA) -> dict:
    return {
        "schema": schema,
        "component": {"name": "reference-propagator", "version": "1.0.0"},
        "profile": "application",
        "operating_domain": {
            "identifier": "stable-negative-spectrum-v1",
            "units": "dimensionless",
        },
        "coverage": {
            "scope_frozen": True,
            "sample_count": 24,
            "anchors": [
                "nominal",
                "boundary",
                "heterogeneous_batch",
                "perturbation",
                "execution_policy",
                "long_horizon",
                "application_composition",
            ],
        },
        "requested_execution": {"dtype": "float64", "device": "cpu"},
        "observed_execution": {"dtype": "float64", "device": "cpu"},
        "evidence": {
            "value_accuracy": True,
            "gradient_accuracy": True,
            "batch_shape": True,
            "batch_independence": True,
            "repeatability": True,
            "ood_control": True,
            "long_horizon": True,
            "dtype_conformance": True,
            "device_local": True,
            "unit_consistency": True,
            "resource_reported": True,
            "calibration": True,
            "composition": True,
        },
        "provenance": {"implementation": "reference-python", "run_id": run_id},
    }


def shuffled(value, rng: random.Random):
    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        return {key: shuffled(item, rng) for key, item in items}
    if isinstance(value, list):
        return [shuffled(item, rng) for item in value]
    return value


def inject(record: dict, fault: str) -> dict:
    faulty = deepcopy(record)
    if fault == "inadequate_truncation":
        faulty["evidence"]["value_accuracy"] = False
    elif fault == "detached_gradient":
        faulty["evidence"]["gradient_accuracy"] = False
    elif fault == "silent_dtype_downgrade":
        faulty["observed_execution"]["dtype"] = "float32"
    elif fault == "batch_crosstalk":
        faulty["evidence"]["batch_independence"] = False
    elif fault == "silent_cpu_fallback":
        faulty["requested_execution"]["device"] = "cuda"
        faulty["observed_execution"]["device"] = "cpu"
    elif fault == "ood_scope_misuse":
        faulty["evidence"]["ood_control"] = False
    elif fault == "unit_mismatch":
        faulty["evidence"]["unit_consistency"] = False
    elif fault == "missing_provenance":
        del faulty["provenance"]
    elif fault == "unfrozen_scope":
        faulty["coverage"]["scope_frozen"] = False
    elif fault == "insufficient_scope_coverage":
        faulty["coverage"]["sample_count"] = 1
        faulty["coverage"]["anchors"] = ["nominal"]
    else:
        raise ValueError(f"unknown fault: {fault}")
    return faulty


def is_rejected(record: dict) -> bool:
    try:
        return evaluate_conformance(record)["conformance"]["status"] != "conformant"
    except ValueError:
        return True


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [center - radius, center + radius]


def interoperability_trials() -> dict:
    trials = []
    for seed in range(40):
        record = base_record(f"interop-{seed}")
        rng = random.Random(seed)
        permuted = shuffled(record, rng)
        api = canonical_json(permuted)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            target = Path(directory) / "output.json"
            source.write_text(json.dumps(permuted, indent=2), encoding="utf-8")
            subprocess.run(
                [sys.executable, "-m", "dfsc_protocol.cli", str(source), str(target)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            cli = target.read_text(encoding="utf-8").strip()
        legacy = deepcopy(record)
        legacy["schema"] = "DFSC-DNC-Conformance-v1"
        legacy.pop("profile")
        legacy.pop("coverage")
        migration_core = json.loads(canonical_json(legacy))
        migration_preserves_evidence = migration_core["evidence"] == record["evidence"]
        migration_requires_requalification = (
            migration_core["conformance"]["status"] == "nonconformant"
            and "migration_requires_requalification"
            in migration_core["conformance"]["coverage_failures"]
        )
        trials.append(
            {
                "seed": seed,
                "api_cli_equal": api == cli,
                "key_order_invariant": record_digest(permuted) == record_digest(record),
                "legacy_migration_preserves_evidence": migration_preserves_evidence,
                "legacy_migration_requires_requalification": migration_requires_requalification,
            }
        )
    return {
        "trials": len(trials),
        "api_cli_equivalence_rate": sum(row["api_cli_equal"] for row in trials) / len(trials),
        "key_order_invariance_rate": sum(row["key_order_invariant"] for row in trials) / len(trials),
        "migration_evidence_preservation_rate": sum(row["legacy_migration_preserves_evidence"] for row in trials) / len(trials),
        "migration_requalification_rate": sum(row["legacy_migration_requires_requalification"] for row in trials) / len(trials),
        "api_cli_wilson_95": wilson(sum(row["api_cli_equal"] for row in trials), len(trials)),
    }


def fault_trials() -> dict:
    faults = [
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
    ]
    per_fault = []
    total_detected = 0
    total_injected = 0
    for fault in faults:
        detected = 0
        for seed in range(20):
            detected += int(is_rejected(inject(base_record(f"{fault}-{seed}"), fault)))
        total_detected += detected
        total_injected += 20
        per_fault.append(
            {
                "fault": fault,
                "injections": 20,
                "detected": detected,
                "detection_rate": detected / 20,
                "wilson_95": wilson(detected, 20),
            }
        )
    clean_rejected = sum(is_rejected(base_record(f"clean-{seed}")) for seed in range(40))
    return {
        "catalogue": per_fault,
        "overall_injections": total_injected,
        "overall_detected": total_detected,
        "overall_detection_rate": total_detected / total_injected,
        "overall_detection_wilson_95": wilson(total_detected, total_injected),
        "clean_records": 40,
        "false_rejections": clean_rejected,
        "false_rejection_rate": clean_rejected / 40,
        "false_rejection_wilson_95": wilson(clean_rejected, 40),
    }


def main() -> None:
    result = {
        "schema": "DFSC-CSCI-Conformance-Validation-v1",
        "specification_status": "proposed_project_specification_not_formal_standard",
        "profiles": ["core", "extended", "application"],
        "interoperability": interoperability_trials(),
        "fault_injection": fault_trials(),
    }
    output = RESULTS / "p4_csi_conformance_validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
