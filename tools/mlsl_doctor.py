"""Print a JSON capability report for the active dfsc environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import capability_report, component_summary


def main() -> None:
    report = capability_report()
    report["dfsc"] = component_summary()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
