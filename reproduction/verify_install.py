"""Verify an installed dfsc release without importing from the source tree."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

import dfsc


EXPECTED_VERSION = "0.1.0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dfsc_install_report.json"))
    parser.add_argument("--allow-version", default=EXPECTED_VERSION)
    args = parser.parse_args()

    smoke_tests = [dfsc.smoke_test(device="cpu", dtype=torch.float64)]
    if torch.cuda.is_available():
        smoke_tests.extend(
            [
                dfsc.smoke_test(device="cuda", dtype=torch.float64),
                dfsc.smoke_test(device="cuda", dtype=torch.float32),
            ]
        )

    passed = dfsc.__version__ == args.allow_version and all(
        row["finite_output"] and row["finite_alpha_grad"] and row["finite_beta_grad"]
        for row in smoke_tests
    )
    report = {
        "passed": passed,
        "expected_dfsc_version": args.allow_version,
        "dfsc_version": dfsc.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "smoke_tests": smoke_tests,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
