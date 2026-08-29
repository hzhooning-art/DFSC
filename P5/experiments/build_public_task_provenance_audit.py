"""Build a machine-readable provenance and independence audit for public tasks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "external"
OUTPUT = RESULTS / "public_task_provenance_audit.json"
SUMMARY = ROOT / "PUBLIC_TASK_PROVENANCE_AUDIT.md"


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_result(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def zenodo_license(record_name: str) -> tuple[str, str]:
    record = json.loads((DATA / record_name).read_text(encoding="utf-8"))
    license_id = record["metadata"]["license"]["id"]
    if license_id.lower() != "cc-by-4.0":
        raise ValueError(f"unexpected Zenodo license {license_id!r} in {record_name}")
    return "CC-BY-4.0", "metadata.license.id"


def main() -> None:
    pva = load_result("public_pva_relaxation.json")
    gas = load_result("public_uci_gas_recovery.json")
    hydraulic = load_result("public_uci_hydraulic_transients.json")
    kupfer = load_result("public_kupferdigital_relaxation.json")
    pva_license, pva_field = zenodo_license("zenodo_21333840_api.json")
    kupfer_license, kupfer_field = zenodo_license("zenodo_10820438_api.json")
    hydrogel_license, hydrogel_field = zenodo_license("zenodo_19005159_api.json")

    pva_path = Path(pva["source"]["file"])
    gas_path = DATA / "uci_gas_flow_308" / "files" / "rawdata.csv.gz"
    hydraulic_path = DATA / "uci_hydraulic_447" / "uci_447.zip"
    kupfer_path = DATA / "zenodo_10820438" / "Files.zip"
    hydrogel_path = DATA / "zenodo_19005159" / "Rheology_2026.zip"

    sources = {
        "pva_gpe": {
            "included": True,
            "evidence_tier": "small-sample support",
            "title": pva["source"]["title"], "doi": pva["source"]["doi"],
            "url": pva["source"]["record_url"], "license": pva_license,
            "license_metadata_field": pva_field,
            "independent_unit": "physical specimen", "independent_unit_count": 3,
            "nested_measurements": "three loading cycles per specimen",
            "split_rule": "leave one specimen out; cycles never cross folds",
            "local_artifact": str(pva_path), "checksum_algorithm": "md5",
            "checksum": pva["source"]["md5"],
            "checksum_verified": digest(pva_path, "md5") == pva["source"]["md5"],
            "scope": "limited positive evidence; not a population-level validation",
        },
        "uci_gas": {
            "included": True, "evidence_tier": "public diagnostic task",
            "title": gas["source"]["title"], "doi": gas["source"]["doi"],
            "url": gas["source"]["url"], "license": gas["source"]["license"],
            "license_metadata_field": "UCI dataset metadata",
            "independent_unit": "non-air exposure experiment", "independent_unit_count": 50,
            "nested_measurements": "16 sensor channels within exposure",
            "split_rule": "held acquisition batch; channels remain clustered",
            "local_artifact": str(gas_path), "checksum_algorithm": "sha256",
            "checksum": gas["source"]["sha256"],
            "checksum_verified": digest(gas_path, "sha256") == gas["source"]["sha256"],
            "scope": "shared empirical recovery realization, not unique chemistry",
        },
        "uci_hydraulic": {
            "included": True, "evidence_tier": "public criterion-sensitivity task",
            "title": hydraulic["source"]["title"], "doi": hydraulic["source"]["doi"],
            "url": hydraulic["source"]["url"], "license": hydraulic["source"]["license"],
            "license_metadata_field": "UCI dataset metadata",
            "independent_unit": "load cycle", "independent_unit_count": 30,
            "nested_measurements": "four channels within cycle",
            "split_rule": "held cooler-condition group; channels remain clustered",
            "local_artifact": str(hydraulic_path), "checksum_algorithm": "sha256",
            "checksum": hydraulic["source"]["archive_sha256"],
            "checksum_verified": digest(hydraulic_path, "sha256") == hydraulic["source"]["archive_sha256"],
            "scope": "ordinary and AR(1)-profile BIC disagreement is retained",
        },
        "kupferdigital": {
            "included": True, "evidence_tier": "primary public positive task",
            "title": kupfer["source"]["title"], "doi": kupfer["source"]["doi"],
            "url": kupfer["source"]["record_url"], "license": kupfer_license,
            "license_metadata_field": kupfer_field,
            "independent_unit": "physical stress-relaxation test file/specimen", "independent_unit_count": 17,
            "material_group_count": 9,
            "split_rule": "leave one material-code group out; specimens from one material never cross folds",
            "local_artifact": str(kupfer_path), "checksum_algorithm": "md5",
            "checksum": kupfer["source"]["archive_md5"],
            "checksum_verified": digest(kupfer_path, "md5") == kupfer["source"]["archive_md5"],
            "scope": "supports a shared empirical rank-one relaxation under AR(1)-profile BIC; not a unique mechanism",
        },
        "hydrogel_candidate": {
            "included": False, "evidence_tier": "screened but excluded",
            "title": "Stress Relaxation Timescale and Hydrogel Network Connectivity Regulate Neural Progenitor Cell Stemness and Differentiation",
            "doi": "10.5281/zenodo.19005159", "url": "https://zenodo.org/records/19005159",
            "license": hydrogel_license, "license_metadata_field": hydrogel_field,
            "independent_unit": "sample replicate file", "independent_unit_count_per_condition": 2,
            "local_artifact": str(hydrogel_path), "checksum_algorithm": "md5",
            "checksum": "82535a017b20ad4f7324bd999f200af7",
            "checksum_verified": digest(hydrogel_path, "md5") == "82535a017b20ad4f7324bd999f200af7",
            "exclusion_reason": "only two independent sample files per condition; internal trial blocks are not independent evidence units",
        },
    }
    if not all(row.get("checksum_verified", True) for row in sources.values()):
        raise ValueError("at least one frozen public artifact failed checksum verification")
    payload = {
        "schema_version": "1.0.0",
        "inclusion_rule": {
            "required": ["public source or frozen official record", "verified local checksum", "declared independent unit", "group-aware split", "claim boundary"],
            "confirmatory_preference": "at least five independent units; smaller tasks are explicitly tiered as limited support",
            "nonindependence_rule": "channels, cycles, or internal trial blocks nested in one specimen/experiment are not counted as independent units",
        },
        "sources": sources,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Public-task provenance and independence audit", "",
        "This manifest freezes source identity, license provenance, local checksums, independent units, grouping, and claim boundaries.", "",
        "| Source | Included | Evidence tier | Independent units | License | Checksum |", "|---|---:|---|---:|---|---|",
    ]
    for name, row in sources.items():
        count = row.get("independent_unit_count", row.get("independent_unit_count_per_condition", "-"))
        lines.append(f"| {name} | {row['included']} | {row['evidence_tier']} | {count} | {row['license']} | {'verified' if row['checksum_verified'] else 'FAILED'} |")
    lines += ["", "The hydrogel candidate is not promoted because its internal trial blocks cannot substitute for independent samples.", "The PVA task remains explicitly small-sample support; KupferDigital provides the primary independent positive public task.", ""]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()