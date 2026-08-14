"""Command-line adapter for the DFSC differentiable-component specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .conformance import canonical_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a differentiable-component conformance record")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

