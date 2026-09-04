"""Capture the complete SciPy #15620 buggy/fixed environment pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
RUNNER = Path(__file__).with_name("scipy_resample_poly_historical_pair_runner.py")
LEGACY_ROOT = REPOSITORY / ".codex_tmp" / "p4_scipy114_replay"
LEGACY_PYTHON = LEGACY_ROOT / "python310" / "python.exe"
DOWNLOADS = LEGACY_ROOT / "downloads"
DEFAULT_OUTPUT = ROOT / "results" / "p4_scipy_resample_poly_pair.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute(python: Path, role: str) -> dict:
    command = [str(python), str(RUNNER), "--expected-role", role]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{role} replay failed: {completed.stderr}")
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stderr": completed.stderr.strip(),
        "result": json.loads(completed.stdout),
    }


def run() -> dict:
    buggy = execute(LEGACY_PYTHON, "buggy")
    fixed = execute(Path(sys.executable), "fixed")
    complete = buggy["result"]["role_confirmed"] is True and fixed["result"]["role_confirmed"] is True
    scipy_wheel = DOWNLOADS / "scipy-1.14.1-cp310-cp310-win_amd64.whl"
    numpy_wheel = DOWNLOADS / "numpy-2.1.3-cp310-cp310-win_amd64.whl"
    return {
        "schema": "P4-SciPy-Resample-Poly-Complete-Historical-Pair-v1",
        "case_id": "scipy_15620",
        "upstream_issue": "https://github.com/scipy/scipy/issues/15620",
        "fix_pull_request": "https://github.com/scipy/scipy/pull/21686",
        "fix_commit": "fec5d2012691729c0b49bb26277464891ac4f189",
        "runner_sha256": sha256(RUNNER),
        "runner_unchanged_between_sides": True,
        "legacy_environment": {
            "python": "3.10.11",
            "scipy_wheel_sha256": sha256(scipy_wheel),
            "numpy_wheel_sha256": sha256(numpy_wheel),
        },
        "buggy_side": buggy,
        "fixed_side": fixed,
        "complete_pair": complete,
        "complete_pair_count": int(complete),
        "historical_family_ordinal": 3,
        "claim_boundary": "This is the third complete defect family overall and the second from SciPy. It adds a silent integer-dtype output failure, but the three families still span only two projects.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
