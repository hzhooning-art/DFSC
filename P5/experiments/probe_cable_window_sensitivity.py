"""Stage 76 post-result window sensitivity for the cable-ageing transfer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FILE = Path(__file__).with_name("stage76_cable_window_sensitivity_contract.json")

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p5_memory_protocol import CurveRecord  # noqa: E402
from probe_external_power_certificate_transfer import evaluate_group, load_certificates  # noqa: E402
from probe_preregistered_cable_ageing_transfer import CONTRACT_FILE as SOURCE_CONTRACT_FILE, load_curves  # noqa: E402


def crop(row: CurveRecord, start_fraction: float, end_fraction: float) -> CurveRecord:
    time = np.asarray(row.time, dtype=float)
    value = np.asarray(row.value, dtype=float)
    duration = float(time[-1] - time[0])
    lower = float(time[0] + start_fraction * duration)
    upper = float(time[0] + end_fraction * duration)
    mask = (time >= lower) & (time <= upper)
    return CurveRecord(row.unit, row.group, row.channel, time[mask], value[mask])


def run() -> dict:
    contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    source_contract = json.loads(SOURCE_CONTRACT_FILE.read_text(encoding="utf-8"))
    rows, _ = load_curves(source_contract)
    certificates = load_certificates()
    records = []
    for start in contract["start_fractions"]:
        for end in contract["end_fractions"]:
            cropped = [crop(row, float(start), float(end)) for row in rows]
            minimum = min(len(row.time) for row in cropped)
            if minimum < int(contract["minimum_retained_raw_points"]):
                raise RuntimeError("window violates frozen minimum raw point count")
            result = evaluate_group("cable_ageing_window", f"start-{start}-end-{end}", cropped, certificates)
            records.append({"start_fraction": start, "end_fraction": end, "minimum_raw_points": minimum, "result": result})
    decisions = [row["result"]["decision"] for row in records]
    eligible = [row["result"]["scope"]["eligible"] for row in records]
    parent_decision = json.loads((ROOT / contract["parent_result"]).read_text(encoding="utf-8"))["record"]["decision"]
    return {
        "schema": "P5-Cable-Window-Sensitivity-v1",
        "window_count": len(records),
        "scope_eligible_count": sum(eligible),
        "parent_decision": parent_decision,
        "matching_parent_decision_count": sum(decision == parent_decision for decision in decisions),
        "success_rule_passes": all(eligible) and all(decision == parent_decision for decision in decisions),
        "criterion_improvement_range": [
            min(row["result"].get("criterion_improvement_rank1_to_rank2", float("inf")) for row in records),
            max(row["result"].get("criterion_improvement_rank1_to_rank2", float("-inf")) for row in records),
        ],
        "records": records,
        "claim_boundary": contract["claim_boundary"],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
