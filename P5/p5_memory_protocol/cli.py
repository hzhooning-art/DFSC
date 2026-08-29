"""One-command reproduction entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="p5-memory-protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    reproduce = sub.add_parser("reproduce")
    reproduce.add_argument("--task", choices=("uci-gas", "pva", "public"), default="public")
    args = parser.parse_args()
    scripts = []
    if args.task in ("pva", "public"):
        scripts.append(ROOT / "experiments" / "probe_public_pva_relaxation.py")
    if args.task in ("uci-gas", "public"):
        scripts.append(ROOT / "experiments" / "probe_public_uci_gas_recovery.py")
    for script in scripts:
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
