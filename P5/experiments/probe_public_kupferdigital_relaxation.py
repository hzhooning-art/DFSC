"""Independent public-data audit of KupferDigital stress-relaxation tests."""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_ROOT = ROOT / "data" / "external" / "zenodo_10820438"
ARCHIVE = DATA_ROOT / "Files.zip"
PRIMARY = DATA_ROOT / "extracted" / "Stress relaxation test" / "Primary data"
RESULTS = ROOT / "results"
OUTPUT_JSON = RESULTS / "public_kupferdigital_relaxation.json"
OUTPUT_MD = RESULTS / "public_kupferdigital_relaxation.md"
EXPECTED_MD5 = "11e68a094a0189a6706d6566d1f78d73"
DOI = "10.5281/zenodo.10820438"
GRID_SIZE = 96
RANKS = (1, 2, 3)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_numeric_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="cp1252", errors="replace").splitlines()
    header = next(index for index, line in enumerate(lines) if line.startswith("Zeit\t"))
    rows = []
    for line in lines[header + 1 :]:
        fields = line.strip().split("\t")
        if len(fields) < 5:
            continue
        try:
            rows.append([float(field.replace(",", ".")) for field in fields[:5]])
        except ValueError:
            continue
    array = np.asarray(rows, dtype=float)
    if array.ndim != 2 or array.shape[0] < 100 or array.shape[1] < 5:
        raise ValueError(f"insufficient numeric relaxation data in {path.name}")
    time = array[:, 0]
    stress = array[:, 4]
    order = np.argsort(time, kind="stable")
    time, stress = time[order], stress[order]
    unique = np.concatenate(([True], np.diff(time) > 0))
    return time[unique], stress[unique]


@lru_cache(maxsize=1)
def _load_curves_cached():
    from p5_memory_protocol import CurveRecord

    if file_md5(ARCHIVE) != EXPECTED_MD5:
        raise ValueError("KupferDigital archive checksum does not match the frozen source")
    files = sorted(PRIMARY.glob("Stress relaxation_*.lis"))
    if len(files) != 17:
        raise ValueError("expected 17 independent KupferDigital relaxation experiments")
    raw = []
    for path in files:
        time, stress = _parse_numeric_rows(path)
        raw.append((path, time, stress))
    common_end = min(float(time[-1]) for _, time, _ in raw)
    if common_end < 86000.0:
        raise ValueError("the declared 24-hour audit window is not covered by every experiment")
    grid_seconds = np.concatenate(([0.0], np.geomspace(1.0, common_end, GRID_SIZE - 1)))
    curves = []
    for path, time, stress in raw:
        registered = np.interp(grid_seconds, time, stress)
        if not np.isfinite(registered).all() or registered[0] <= 0:
            raise ValueError(f"invalid registered stress curve in {path.name}")
        normalized = registered / registered[0]
        unit = path.stem.removeprefix("Stress relaxation_")
        group = unit.split("_")[0]
        curves.append(CurveRecord(unit, group, "normalized_stress", grid_seconds / 3600.0, normalized))
    return tuple(curves)


def load_curves():
    return list(_load_curves_cached())


def build_payload() -> dict:
    from p5_memory_protocol import GateConfig, decide, evaluate, fit, identifiability_certificate

    curves = load_curves()
    evaluation_iid = evaluate(
        curves,
        RANKS,
        starts=8,
        rate_bounds=(1.0 / 240.0, 20.0),
        nonnegative_amplitudes=False,
    )
    evaluation_ar1 = evaluation_iid
    decision_iid = decide(evaluation_iid, GateConfig(use_ar1_bic=False))
    decision_ar1 = decide(evaluation_ar1, GateConfig(use_ar1_bic=True))
    primary_decision = decision_ar1
    certificate = None
    if primary_decision["selected_rank"] is not None:
        fitted = fit(
            curves,
            primary_decision["selected_rank"],
            starts=10,
            rate_bounds=(1.0 / 240.0, 20.0),
        )
        certificate = identifiability_certificate(curves, fitted["rates"])
    checks = {
        "archive_checksum": file_md5(ARCHIVE) == EXPECTED_MD5,
        "seventeen_independent_experiments": len(curves) == 17 and len({row.unit for row in curves}) == 17,
        "nine_material_groups": len({row.group for row in curves}) == 9,
        "common_registered_grid": all(np.allclose(row.time, curves[0].time) for row in curves),
        "finite_normalized_curves": all(np.isfinite(row.value).all() and abs(row.value[0] - 1.0) < 1e-12 for row in curves),
    }
    return {
        "experiment": "public_kupferdigital_relaxation",
        "protocol_frozen_before_fit": True,
        "source": {
            "title": "KupferDigital mechanical testing datasets: Stress relaxation and low-cycle fatigue (LCF) tests",
            "doi": DOI,
            "record_url": "https://zenodo.org/records/10820438",
            "access": "open",
            "license": "CC-BY-4.0",
            "license_metadata_field": "metadata.license.id",
            "archive": str(ARCHIVE),
            "archive_md5": file_md5(ARCHIVE),
            "independent_experiments": len(curves),
            "material_groups": sorted({row.group for row in curves}),
        },
        "independent_unit": "one physical stress-relaxation test file / specimen identifier",
        "grouping": "leave one material-code group out; repeated specimens from one material never cross a fold",
        "preprocessing": {
            "time_unit": "hours",
            "registered_samples": GRID_SIZE,
            "grid": "0 plus geometric spacing from 1 second to the shortest complete endpoint",
            "value": "measured engineering stress divided by the first registered stress",
            "smoothing": "none",
        },
        "decision": primary_decision,
        "decision_iid_bic": decision_iid,
        "decision_ar1_profile_bic": decision_ar1,
        "evaluation": evaluation_iid,
        "identifiability_certificate": certificate,
        "checks": checks,
        "route_pass": all(checks.values()),
        "claim_boundary": (
            "The decision concerns a shared empirical relaxation realization on the declared "
            "24-hour copper-alloy tests; it is not evidence for a unique microscopic mechanism."
        ),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# KupferDigital public stress-relaxation audit",
        "",
        f"Route pass: **{payload['route_pass']}**.",
        "",
        f"Primary AR(1)-profile decision: **{payload['decision']['decision']}**.",
        "",
        f"Ordinary-BIC sensitivity decision: **{payload['decision_iid_bic']['decision']}**.",
        "",
        "Independent evidence units: 17 physical test files in 9 material-code groups.",
        "",
        "The split is material-grouped, so repeated specimens from one material cannot leak across folds.",
        "",
        "## Rank diagnostics",
        "",
        "| Rank | Mean BIC | Mean AR(1) BIC | Median held-group NRMSE | Max log-rate std | Min rate ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rank in RANKS:
        row = payload["evaluation"]["rank_records"][str(rank)]
        lines.append(
            f"| {rank} | {row['mean_bic']:.2f} | {row['mean_ar1_bic']:.2f} | "
            f"{row['median_prediction_nrmse']:.4f} | {row['max_log_rate_std']:.3f} | "
            f"{row['minimum_rate_ratio']:.3f} |"
        )
    lines.extend(["", payload["claim_boundary"]])
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload()
    write_outputs(payload)
    print(json.dumps({"decision": payload["decision"]["decision"], "route_pass": payload["route_pass"]}, indent=2))


if __name__ == "__main__":
    main()


