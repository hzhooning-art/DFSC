"""Stage 61: freeze the confirmatory evidence chain and its claims."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARTIFACTS = (
    "morphology_calibrated_asymmetric_hierarchy.json",
    "external_multiphysics_confirmation.json",
    "cross_implementation_confirmation.json",
    "external_scope_boundary_map.json",
)
SCRIPTS = (
    "probe_morphology_calibrated_asymmetric_hierarchy.py",
    "probe_external_multiphysics_confirmation.py",
    "probe_cross_implementation_confirmation.py",
    "probe_external_scope_boundary_map.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> dict:
    payloads = {name: json.loads((RESULTS / name).read_text(encoding="utf-8")) for name in ARTIFACTS}
    stage57, stage58, stage59, stage60 = (payloads[name] for name in ARTIFACTS)
    files = []
    for path in [*(RESULTS / name for name in ARTIFACTS), *(ROOT / "experiments" / name for name in SCRIPTS)]:
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    claims = {
        "supported": [
            "The morphology-calibrated asymmetric rule passed its locked Stage 54 evaluation.",
            "The frozen rule passed a 42-pair external confirmation spanning three physical data sources.",
            "Independent SciPy and finite-difference implementations confirmed values and parameter gradients.",
            "A standalone decision implementation exactly replayed all Stage 57 and Stage 58 decisions.",
            "The normal external observation regime is supported under the preregistered checks.",
        ],
        "scope_limited": [
            "Doubled residual noise, 18-point training sampling, and a 45% observation window are scope-limited.",
            "Indeterminate outcomes are deliberate abstentions and must not be counted as mechanism recovery.",
            "The evidence supports the declared low-dimensional shared-spectrum setting, not arbitrary fractional systems.",
        ],
        "not_claimed": [
            "No independent external team has reproduced the artifact.",
            "No universal mechanism-identification guarantee or global theoretical error bound is claimed.",
            "Local cross-implementation agreement is not a substitute for a public archival release or DOI.",
        ],
    }
    checks = {
        "stage57_passed": bool(stage57["route_pass"]),
        "stage58_passed": bool(stage58["route_pass"]),
        "stage59_passed": bool(stage59["route_pass"]),
        "stage60_passed": bool(stage60["route_pass"]),
        "no_partial_checkpoint_remains": not any(RESULTS.glob("*.partial.json")),
        "all_evidence_files_hashed": len(files) == len(ARTIFACTS) + len(SCRIPTS),
    }
    return {
        "freeze": "stage61_confirmatory_evidence_freeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_thresholds": stage57["thresholds"],
        "headline_metrics": {
            "locked_stage57": stage57["evaluation_metrics"],
            "external_stage58": stage58["metrics"],
            "cross_implementation_stage59": {
                **stage59["maxima"],
                "decision_concordance": stage59["decision_replay"]["concordance"],
                "replayed_decisions": stage59["decision_replay"]["count"],
            },
            "scope_status_stage60": {name: row["status"] for name, row in stage60["regimes"].items()},
        },
        "claims": claims,
        "files": files,
        "checks": checks,
        "freeze_pass": all(checks.values()),
    }


def write_outputs(manifest: dict) -> None:
    (RESULTS / "stage61_evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metrics = manifest["headline_metrics"]
    lines = [
        "# Stage 61 evidence freeze", "", f"Freeze pass: **{manifest['freeze_pass']}**.", "",
        "## Confirmed evidence", "",
        f"- Stage 57 locked coverage: {metrics['locked_stage57']['coverage']:.4f}.",
        f"- Stage 58 external coverage / selective accuracy: {metrics['external_stage58']['coverage']:.4f} / {metrics['external_stage58']['selective_accuracy']:.4f}.",
        f"- Stage 59 value / gradient relative error: {metrics['cross_implementation_stage59']['value_relative_error']:.3e} / {metrics['cross_implementation_stage59']['gradient_relative_error']:.3e}.",
        f"- Stage 59 decision concordance: {metrics['cross_implementation_stage59']['decision_concordance']:.4f} over {metrics['cross_implementation_stage59']['replayed_decisions']} decisions.",
        "", "## Scope map", "",
    ]
    lines.extend(f"- {name}: **{status}**." for name, status in metrics["scope_status_stage60"].items())
    lines.extend(["", "## Claim boundary", ""])
    lines.extend(f"- {claim}" for claim in manifest["claims"]["not_claimed"])
    (ROOT / "STAGE61_FREEZE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest = build_manifest()
    write_outputs(manifest)
    print(json.dumps({"checks": manifest["checks"], "freeze_pass": manifest["freeze_pass"]}, indent=2))


if __name__ == "__main__":
    main()
