"""Build frozen-record gate ablation and multiplicity audits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p5_memory_protocol import decide_transitions, holm_adjust, report

RESULTS = ROOT / "results"
GATE_OUTPUT = RESULTS / "gate_ablation_audit.json"
GATE_SUMMARY = RESULTS / "gate_ablation_audit.md"
MULT_OUTPUT = RESULTS / "multiple_comparison_audit.json"
MULT_SUMMARY = RESULTS / "multiple_comparison_audit.md"
ALL_GATES = {"information", "transfer", "stability", "separation"}


def _normalized_transition(row: dict, from_rank: int) -> dict:
    return {"from_rank": from_rank, "to_rank": int(row["to_rank"]), "gates": row["gates"]}


def _single_failure_transition(payload: dict, failed: str) -> tuple[dict, dict]:
    for boundary in payload["boundary"]:
        for index, row in enumerate(boundary["transitions"]):
            failed_here = [name for name in ("bic", "prediction", "stability", "separation") if not row["gates"][name]]
            if failed_here == [failed]:
                metadata = {"horizon_seconds": boundary["horizon_seconds"], "samples_per_curve": boundary["samples_per_curve"]}
                return _normalized_transition(row, index + 1), metadata
    raise RuntimeError(f"no single-{failed} failure found")


def build_gate_audit() -> dict:
    pva = json.loads((RESULTS / "public_pva_relaxation.json").read_text(encoding="utf-8"))
    gas = json.loads((RESULTS / "public_uci_gas_recovery.json").read_text(encoding="utf-8"))
    transfer, transfer_meta = _single_failure_transition(pva, "prediction")
    stability, stability_meta = _single_failure_transition(pva, "stability")
    separation = _normalized_transition(gas["decision"]["transitions"][0], 1)
    information = {
        "from_rank": 1,
        "to_rank": 2,
        "gates": {"bic": False, "prediction": True, "stability": True, "separation": True, "finite": True},
    }
    cases = {
        "information_control": (information, {"source": "controlled frozen transition", "claim": "logic-unit audit only"}),
        "transfer_pva": (transfer, {"source": "public PVA boundary", **transfer_meta}),
        "stability_pva": (stability, {"source": "public PVA boundary", **stability_meta}),
        "separation_gas": (separation, {"source": "public UCI gas recovery", "doi": gas["source"]["doi"]}),
    }
    records = []
    for omitted, (transition, metadata) in cases.items():
        full = decide_transitions([transition])
        ablated = decide_transitions([transition], active_gates=ALL_GATES - {omitted.split("_")[0]})
        records.append({
            "case": omitted,
            "omitted_gate": omitted.split("_")[0],
            "metadata": metadata,
            "frozen_transition": transition,
            "full_protocol": full,
            "leave_one_gate_out": ablated,
            "decision_changed": full["decision"] != ablated["decision"],
        })
    return {
        "experiment": "leave_one_gate_out_frozen_record_audit",
        "refitting_performed": False,
        "records": records,
        "all_four_gates_decision_relevant": all(row["decision_changed"] for row in records),
        "claim_boundary": "The audit establishes logical decision relevance on selected frozen stress records; it does not estimate population-level predictive importance.",
    }


def build_multiple_audit() -> dict:
    gas = json.loads((RESULTS / "public_uci_gas_recovery.json").read_text(encoding="utf-8"))
    hydraulic = json.loads((RESULTS / "public_uci_hydraulic_transients.json").read_text(encoding="utf-8"))
    gas_raw = {name: row["shared_rank3_comparison"]["wilcoxon_p_one_sided"] for name, row in gas["baselines"].items() if "shared_rank3_comparison" in row}
    hyd_raw = {name: row["wilcoxon_p_baseline_greater"] for name, row in hydraulic["model_family_baselines"].items() if isinstance(row, dict) and "wilcoxon_p_baseline_greater" in row}
    return {
        "experiment": "within_task_holm_familywise_audit",
        "method": "Holm step-down correction, applied separately within each public-task baseline family",
        "families": {
            "uci_gas_recovery": {"raw_p": gas_raw, "holm_adjusted_p": holm_adjust(gas_raw), "family_size": len(gas_raw)},
            "uci_hydraulic_transients": {"raw_p": hyd_raw, "holm_adjusted_p": holm_adjust(hyd_raw), "family_size": len(hyd_raw)},
        },
        "claim_boundary": "Adjusted p-values address multiplicity among matched baseline tests within each task; they do not correct across exploratory analyses elsewhere in the project.",
    }


def _write_markdown(gate: dict, multiple: dict) -> None:
    lines = ["# Leave-one-gate-out audit", "", "| Case | Omitted gate | Full | Ablated | Changed |", "|---|---|---|---|---|" ]
    for row in gate["records"]:
        lines.append(f"| {row['case']} | {row['omitted_gate']} | {row['full_protocol']['decision']} | {row['leave_one_gate_out']['decision']} | {row['decision_changed']} |")
    lines += ["", gate["claim_boundary"], ""]
    GATE_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    lines = ["# Holm-adjusted matched baseline tests", ""]
    for family, data in multiple["families"].items():
        lines += [f"## {family}", "", "| Baseline | Raw p | Holm-adjusted p |", "|---|---:|---:|" ]
        for name, raw in data["raw_p"].items():
            lines.append(f"| {name} | {raw:.6g} | {data['holm_adjusted_p'][name]:.6g} |")
        lines.append("")
    MULT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    gate = build_gate_audit()
    multiple = build_multiple_audit()
    report(gate, GATE_OUTPUT)
    report(multiple, MULT_OUTPUT)
    _write_markdown(gate, multiple)
    print(json.dumps({"gate_audit": gate["all_four_gates_decision_relevant"], "families": multiple["families"]}, indent=2))


if __name__ == "__main__":
    main()