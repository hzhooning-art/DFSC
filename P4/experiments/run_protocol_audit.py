"""Run the reproducible P4 protocol evidence chain in one command."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "P4" / "results"


def run(script: str) -> dict:
    command = [sys.executable, str(ROOT / "P4" / "experiments" / script)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"{script} failed with exit code {completed.returncode}:\n{completed.stdout}\n{completed.stderr}"
        )
    return {"script": script, "returncode": completed.returncode}


def main() -> None:
    steps = [
        run("p4_mlsl_protocol_validation.py"),
        run("p4_protocol_registry.py"),
        run("p4_public_api_smoke.py"),
    ]
    registry = json.loads((RESULTS / "p4_primitive_protocol_registry.json").read_text(encoding="utf-8"))
    mlsl = next(row for row in registry["records"] if row["backend"] == "MLSL")
    result = {
        "schema": "DFSC-P4-Reproducibility-Manifest-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "steps": steps,
        "registry_schema": registry["schema"],
        "backend_count": len(registry["records"]),
        "mlsl_gate_status": {key: mlsl[key] for key in registry["required_dimensions"]},
        "status": "conformant" if mlsl["status"] == "conformant" else "nonconformant",
        "external_reproduction": "not required for this local evidence run; recommended before publication",
    }
    output = RESULTS / "p4_reproducibility_manifest.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
