"""Stage 75 new-to-P5 transfer under a frozen extraction and decision contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
CONTRACT_FILE = Path(__file__).with_name("stage75_cable_ageing_transfer_contract.json")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p5_memory_protocol import CurveRecord  # noqa: E402
from probe_external_power_certificate_transfer import (  # noqa: E402
    evaluate_group,
    load_certificates,
    standardize_group,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shared_strings(raw: bytes) -> list[str]:
    root = ET.fromstring(raw)
    return ["".join(node.text or "" for node in item.iter(f"{NS}t")) for item in root.findall(f"{NS}si")]


def _worksheet_cells(workbook: Path, contract: dict) -> dict[str, str | float]:
    with zipfile.ZipFile(workbook) as bundle:
        strings = _shared_strings(bundle.read(contract["source"]["shared_strings_member"]))
        root = ET.fromstring(bundle.read(contract["source"]["worksheet_member"]))
    output: dict[str, str | float] = {}
    for cell in root.iter(f"{NS}c"):
        value = cell.find(f"{NS}v")
        if value is None or value.text is None:
            continue
        output[cell.attrib["r"]] = strings[int(value.text)] if cell.attrib.get("t") == "s" else float(value.text)
    return output


def load_curves(contract: dict) -> tuple[list[CurveRecord], list[dict]]:
    workbook = REPOSITORY / contract["source"]["path"]
    observed_hash = file_sha256(workbook)
    if observed_hash != contract["source"]["sha256"]:
        raise RuntimeError("source workbook hash does not match the frozen contract")
    cells = _worksheet_cells(workbook, contract)
    records = []
    audit = []
    first_row = int(contract["extraction"]["first_data_row"])
    minimum = int(contract["extraction"]["minimum_raw_points_per_curve"])
    for item in contract["extraction"]["curves"]:
        if cells.get(f"{item['time_column']}2") != "Time" or cells.get(f"{item['response_column']}2") != item["expected_label"]:
            raise RuntimeError(f"header mismatch for {item['unit']}")
        pairs = []
        row = first_row
        while True:
            time = cells.get(f"{item['time_column']}{row}")
            value = cells.get(f"{item['response_column']}{row}")
            if time is None and value is None:
                break
            if isinstance(time, float) and isinstance(value, float):
                pairs.append((time, value))
            row += 1
        if len(pairs) < minimum:
            raise RuntimeError(f"insufficient raw observations for {item['unit']}")
        array = np.asarray(pairs, dtype=float)
        array = array[np.isfinite(array).all(axis=1)]
        order = np.argsort(array[:, 0], kind="stable")
        array = array[order]
        array = array[np.r_[True, np.diff(array[:, 0]) > 0.0]]
        time = array[:, 0] - array[0, 0]
        onset = float(array[0, 1])
        value = np.sign(onset) * array[:, 1] / abs(onset)
        records.append(CurveRecord(item["unit"], "cable-ageing-all-six", item["unit"], time, value))
        audit.append({"unit": item["unit"], "raw_points": len(pairs), "finite_unique_points": len(time)})
    return records, audit


def run() -> dict:
    contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    rows, extraction_audit = load_curves(contract)
    prepared, scope = standardize_group("cable_ageing", "cable-ageing-all-six", rows)
    result = evaluate_group("cable_ageing", "cable-ageing-all-six", rows, load_certificates())
    return {
        "schema": "P5-Preregistered-Cable-Ageing-Transfer-v1",
        "contract_sha256": file_sha256(CONTRACT_FILE),
        "runner_sha256": file_sha256(Path(__file__)),
        "source_sha256": contract["source"]["sha256"],
        "thresholds_retuned_after_observing_outcome": False,
        "source_curve_count": len(rows),
        "standardized_curve_count": 0 if prepared is None else len(prepared),
        "extraction_audit": extraction_audit,
        "record": result,
        "claim_boundary": contract["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run()
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
