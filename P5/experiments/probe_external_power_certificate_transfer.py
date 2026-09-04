"""Stage 72 retrospective transfer of the frozen Stage 69 certificate.

The public tasks predate Stage 69 and are not unopened confirmation data. A
fixed, outcome-independent adapter maps eligible groups to six curves on 24
uniform points over a dimensionless horizon of 16. Scope failures abstain.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p5_memory_protocol import CurveRecord  # noqa: E402
from probe_common_budget_order_detection import _selective_detection  # noqa: E402
import probe_public_kupferdigital_relaxation as kupfer  # noqa: E402
import probe_public_pva_relaxation as pva  # noqa: E402
import probe_public_uci_gas_recovery as gas  # noqa: E402
import probe_public_uci_hydraulic_transients as hydraulic  # noqa: E402


TARGET_HORIZON = 16.0
TARGET_SAMPLES = 24
TARGET_CHANNELS = 6
MAX_CALIBRATED_NOISE = 0.005
AR1_CLASSIFICATION_THRESHOLD = 0.30
MAX_MONOTONICITY_VIOLATION_FRACTION = 0.25
CERTIFICATE_FILE = ROOT / "results" / "power_certified_order_detection.json"


def _chunks(rows: list, size: int) -> list[list]:
    return [rows[index : index + size] for index in range(0, len(rows) - size + 1, size)]


def load_task_groups() -> dict[str, list[tuple[str, list]]]:
    pva_rows = sorted(pva.load_curves(), key=lambda row: (row.sample, row.cycle))
    kupfer_rows = sorted(kupfer.load_curves(), key=lambda row: row.unit)
    gas_rows = gas.load_curves()
    hydraulic_rows = hydraulic.load_curves()

    gas_by_unit: dict[str, list] = defaultdict(list)
    for row in gas_rows:
        gas_by_unit[row.unit].append(row)
    hydraulic_by_group: dict[str, list] = defaultdict(list)
    for row in hydraulic_rows:
        if row.channel == "CE":
            hydraulic_by_group[row.group].append(row)

    return {
        "pva": [("pva-first-six", pva_rows[:TARGET_CHANNELS])],
        "kupferdigital": [
            ("kupfer-block-%d" % (index + 1), block)
            for index, block in enumerate(_chunks(kupfer_rows, TARGET_CHANNELS))
        ],
        "uci_gas": [
            (unit, sorted(rows, key=lambda row: row.channel)[:TARGET_CHANNELS])
            for unit, rows in sorted(gas_by_unit.items())
        ],
        "uci_hydraulic": [
            (group, sorted(rows, key=lambda row: row.unit)[:TARGET_CHANNELS])
            for group, rows in sorted(hydraulic_by_group.items())
        ],
    }


def _source_arrays(row) -> tuple[np.ndarray, np.ndarray, str]:
    time = np.asarray(row.time, dtype=float)
    value = np.asarray(row.value, dtype=float)
    if hasattr(row, "channel"):
        channel = row.channel
    else:
        channel = "sample-%s-cycle-%s" % (row.sample, row.cycle)
    return time, value, str(channel)


def standardize_group(task: str, group_id: str, rows: list) -> tuple[list[CurveRecord] | None, dict]:
    if len(rows) != TARGET_CHANNELS:
        return None, {"eligible": False, "reason": "requires_exactly_six_curves"}
    if min(len(_source_arrays(row)[0]) for row in rows) < TARGET_SAMPLES:
        return None, {"eligible": False, "reason": "fewer_than_24_observed_samples"}

    target_fraction = np.linspace(0.0, 1.0, TARGET_SAMPLES)
    target_time = target_fraction * TARGET_HORIZON
    prepared = []
    normalized_values = []
    for index, row in enumerate(rows):
        source_time, source_value, channel = _source_arrays(row)
        duration = float(source_time[-1] - source_time[0])
        if duration <= 0 or not np.isfinite(source_value).all():
            return None, {"eligible": False, "reason": "invalid_time_or_value"}
        source_fraction = (source_time - source_time[0]) / duration
        value = np.interp(target_fraction, source_fraction, source_value)
        scale = float(value[0] - value[-1])
        if abs(scale) <= 1.0e-8:
            return None, {"eligible": False, "reason": "degenerate_endpoint_scale"}
        normalized = (value - value[-1]) / scale
        normalized_values.append(normalized)
        prepared.append(CurveRecord(
            unit="%s:%s:%d" % (task, group_id, index),
            group=group_id,
            channel=channel,
            time=target_time,
            value=normalized,
        ))

    array = np.stack(normalized_values)
    smooth = savgol_filter(array, window_length=7, polyorder=3, axis=1, mode="interp")
    residual = array - smooth
    centred = residual - np.median(residual)
    noise = float(1.4826 * np.median(np.abs(centred)))
    numerator = float(np.sum(residual[:, :-1] * residual[:, 1:]))
    denominator = float(np.sum(residual[:, :-1] ** 2))
    rho = float(np.clip(numerator / denominator, -0.99, 0.99)) if denominator > 0 else 0.0
    violation = float(np.mean(np.diff(array, axis=1) > 0.0))
    if violation > MAX_MONOTONICITY_VIOLATION_FRACTION:
        return None, {
            "eligible": False,
            "reason": "outside_monotone_decay_morphology",
            "noise_proxy": noise,
            "residual_lag1": rho,
            "monotonicity_violation_fraction": violation,
        }
    if noise > MAX_CALIBRATED_NOISE:
        return None, {
            "eligible": False,
            "reason": "noise_above_stage69_calibration",
            "noise_proxy": noise,
            "residual_lag1": rho,
            "monotonicity_violation_fraction": violation,
        }
    noise_bin = 0.001 if noise <= 0.001 else 0.005
    noise_model = "ar1" if rho >= AR1_CLASSIFICATION_THRESHOLD else "white"
    return prepared, {
        "eligible": True,
        "noise_proxy": noise,
        "declared_noise_bin": noise_bin,
        "residual_lag1": rho,
        "declared_noise_model": noise_model,
        "monotonicity_violation_fraction": violation,
    }


def load_certificates() -> dict[tuple, dict]:
    payload = json.loads(CERTIFICATE_FILE.read_text(encoding="utf-8"))
    return {
        (
            row["horizon"], row["samples_per_channel"], row["noise_std"], row["noise_model"]
        ): row
        for row in payload["certificates"]
    }


def evaluate_group(task: str, group_id: str, rows: list, certificates: dict) -> dict:
    prepared, scope = standardize_group(task, group_id, rows)
    record = {"task": task, "group": group_id, "source_curves": len(rows), "scope": scope}
    if prepared is None:
        return {**record, "decision": "INDETERMINATE_SCOPE"}
    key = (
        TARGET_HORIZON,
        TARGET_SAMPLES,
        scope["declared_noise_bin"],
        scope["declared_noise_model"],
    )
    certificate = certificates[key]
    evidence, _ = _selective_detection(prepared, scope["declared_noise_bin"])
    if evidence["decision"] == 2:
        decision = "EVIDENCE_AGAINST_RANK_1"
    elif (
        evidence["decision"] == 1
        and certificate["qualified_for_rank_one_claim"]
    ):
        decision = "SUPPORTED_RANK_1"
    else:
        decision = "INDETERMINATE_EVIDENCE"
    return {
        **record,
        "decision": decision,
        "certificate_qualified_for_rank_one": certificate["qualified_for_rank_one_claim"],
        "criterion_improvement_rank1_to_rank2": evidence["criterion_improvement_rank1_to_rank2"],
        "rank_two_checks": evidence["checks"],
    }


def run() -> dict:
    groups = load_task_groups()
    certificates = load_certificates()
    records = [
        evaluate_group(task, group_id, rows, certificates)
        for task, task_groups in groups.items()
        for group_id, rows in task_groups
    ]
    by_task = {}
    for task in groups:
        local = [row for row in records if row["task"] == task]
        by_task[task] = {
            "groups": len(local),
            "scope_eligible": sum(row["scope"]["eligible"] for row in local),
            "decisions": {
                decision: sum(row["decision"] == decision for row in local)
                for decision in sorted({row["decision"] for row in local})
            },
        }
    return {
        "schema": "P5-External-Power-Certificate-Transfer-v1",
        "design": {
            "target_channels": TARGET_CHANNELS,
            "target_samples": TARGET_SAMPLES,
            "dimensionless_horizon": TARGET_HORIZON,
            "maximum_calibrated_noise": MAX_CALIBRATED_NOISE,
            "ar1_classification_threshold": AR1_CLASSIFICATION_THRESHOLD,
            "maximum_monotonicity_violation_fraction": MAX_MONOTONICITY_VIOLATION_FRACTION,
            "thresholds_retuned_on_external_outcomes": False,
        },
        "summary_by_task": by_task,
        "records": records,
        "claim_boundary": (
            "This is retrospective frozen-rule transfer because all four public tasks predate Stage 69. "
            "Resampling standardizes representation but does not create new observations. Scope refusal "
            "is a result and must not be removed from the denominator."
        ),
        "record_storage": "Complete group-level transfer records included.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-summary", action="store_true")
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
