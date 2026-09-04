"""Capture current fixed-side records from the portable historical runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


RUNNER = Path(__file__).with_name("pytorch_historical_pair_runner.py")
sys.path.insert(0, str(RUNNER.parent))

from pytorch_historical_pair_runner import CASES, run_case  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run() -> dict:
    records = [run_case(case_id, "fixed") for case_id in sorted(CASES)]
    return {
        "schema": "P4-Historical-Fixed-Replays-v1",
        "runner": RUNNER.name,
        "runner_sha256": sha256(RUNNER),
        "records": records,
        "summary": {
            "attempted": len(records),
            "executed": sum(row["observation"]["executed"] for row in records),
            "fixed_roles_confirmed": sum(row["role_confirmed"] for row in records),
            "buggy_roles_confirmed": 0,
            "complete_pairs": 0,
        },
        "claim_boundary": (
            "These records confirm only the current fixed side. A historical pair remains incomplete "
            "until the identical runner hash confirms the reported bug in the old environment."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-summary", action="store_true")
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
