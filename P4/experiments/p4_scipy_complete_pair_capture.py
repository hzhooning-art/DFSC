"""Capture the complete SciPy #8906 buggy/fixed environment pair."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
RUNNER = Path(__file__).with_name("scipy_historical_pair_runner.py")
LEGACY_ROOT = REPOSITORY / ".codex_tmp" / "p4_scipy114_replay"
LEGACY_PYTHON = LEGACY_ROOT / "python310" / "python.exe"
DOWNLOADS = LEGACY_ROOT / "downloads"
TEMP_ROOT = REPOSITORY / ".codex_tmp"


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
    return {"command": command, "exit_code": completed.returncode, "stderr": completed.stderr.strip(), "result": json.loads(completed.stdout)}


def run() -> dict:
    buggy = execute(LEGACY_PYTHON, "buggy")
    fixed = execute(Path(sys.executable), "fixed")
    complete = buggy["result"]["role_confirmed"] is True and fixed["result"]["role_confirmed"] is True
    scipy_wheel = DOWNLOADS / "scipy-1.14.1-cp310-cp310-win_amd64.whl"
    numpy_wheel = DOWNLOADS / "numpy-2.1.3-cp310-cp310-win_amd64.whl"
    return {
        "schema": "P4-SciPy-Complete-Historical-Pair-v1",
        "case_id": "scipy_8906",
        "upstream_issue": "https://github.com/scipy/scipy/issues/8906",
        "fix_commit": "2d20569f42a0ec1d20ce6b396c12a2b636bd15f4",
        "runner_sha256": sha256(RUNNER),
        "runner_unchanged_between_sides": True,
        "legacy_environment": {
            "python": "3.10.11",
            "scipy_wheel_sha256": sha256(scipy_wheel),
            "numpy_wheel_sha256": sha256(numpy_wheel),
            "buggy_tag_source_sha256": sha256(TEMP_ROOT / "scipy_basic_1_14_1.py"),
            "fixed_tag_source_sha256": sha256(TEMP_ROOT / "scipy_basic_1_15_0.py"),
        },
        "buggy_side": buggy,
        "fixed_side": fixed,
        "complete_pair": complete,
        "complete_pair_count": int(complete),
        "claim_boundary": "This is the second complete defect pair overall and the first from SciPy. It adds project diversity but remains one narrow 1x1 solve_banded defect family.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
