"""Run the core maturity checks for MLSL."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print("$", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    py = sys.executable
    run([py, "validate.py"])
    run([py, "experiments/exp21_primitive_generality.py"])
    run([py, "experiments/exp24_boundary_generality.py"])
    run([py, "experiments/exp26_2d_nonlinear_extensions.py"])
    run([py, "experiments/exp30_primitive_contract_audit.py"])
    run([py, "experiments/exp48_real_geotes_cross_cycle.py"])
    run([py, "tools/mlsl_doctor.py"])
    run([py, "tools/maturity_report.py"])


if __name__ == "__main__":
    main()
