"""Run the minimal validation gate for the MLSL primitive."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(args: list[str]) -> None:
    print("$", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run([python, "-m", "unittest", "discover", "-s", "tests"])
    run([python, "examples/quickstart_layer.py"])
    run([python, "examples/batched_layer.py"])
    run([python, "examples/inverse_alpha_beta.py"])
    run([python, "examples/complex_arnoldi.py"])
    run([python, "examples/application_domains.py"])


if __name__ == "__main__":
    main()
