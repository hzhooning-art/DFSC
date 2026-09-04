"""Capture an unchanged-runner buggy/fixed pair for PyTorch #80770."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
RUNNER = Path(__file__).with_name("pytorch_historical_pair_runner.py")
DEFAULT_LEGACY = REPOSITORY / ".codex_tmp" / "p4_torch111_replay" / "python310" / "python.exe"
DOWNLOADS = REPOSITORY / ".codex_tmp" / "p4_torch111_replay" / "downloads"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute(python: Path, role: str) -> dict:
    command = [str(python), str(RUNNER), "--case", "pytorch_80770", "--expected-role", role]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{role} replay failed: {completed.stderr}")
    payload = json.loads(completed.stdout)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stderr": completed.stderr.strip(),
        "result": payload,
    }


def run(legacy_python: Path) -> dict:
    wheel = DOWNLOADS / "torch-1.11.0+cpu-cp310-cp310-win_amd64.bits.whl"
    python_zip = DOWNLOADS / "python-3.10.11-embed-amd64.zip"
    dependency = legacy_python.parent / "Lib" / "site-packages" / "typing_extensions.py"
    buggy = execute(legacy_python, "buggy")
    fixed = execute(Path(sys.executable), "fixed")
    complete = buggy["result"]["role_confirmed"] is True and fixed["result"]["role_confirmed"] is True
    return {
        "schema": "P4-Complete-Historical-Pair-v1",
        "case_id": "pytorch_80770",
        "upstream_issue": "https://github.com/pytorch/pytorch/issues/80770",
        "runner_sha256": file_sha256(RUNNER),
        "runner_unchanged_between_sides": True,
        "legacy_environment": {
            "python_archive": {"name": python_zip.name, "sha256": file_sha256(python_zip)},
            "torch_wheel": {"name": wheel.name, "sha256": file_sha256(wheel)},
            "typing_extensions_sha256": file_sha256(dependency),
            "numpy_absent_and_not_required_by_runner": True,
        },
        "buggy_side": buggy,
        "fixed_side": fixed,
        "complete_pair": complete,
        "complete_pair_count": int(complete),
        "claim_boundary": (
            "One upstream-reported CPU defect is reproduced in the official PyTorch 1.11.0 CPU wheel "
            "and rejected by the unchanged runner in the current fixed environment. This is one defect "
            "family and does not estimate field prevalence or establish broad external effectiveness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-python", type=Path, default=DEFAULT_LEGACY)
    args = parser.parse_args()
    print(json.dumps(run(args.legacy_python.resolve()), indent=2))


if __name__ == "__main__":
    main()
