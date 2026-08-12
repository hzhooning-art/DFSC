"""Smoke test for the public protocol API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4"))
from dfsc_protocol import load_profile, load_registry, summarize_registry  # noqa: E402


def main():
    registry = load_registry(ROOT / "P4" / "results" / "p4_primitive_protocol_registry.json")
    profile = load_profile(ROOT / "P4" / "results" / "p4_primitive_profile.json")
    result = {
        "registry_backends": len(registry["records"]),
        "profile_rows": len(profile["rows"]),
        "summary": summarize_registry(registry),
        "status": "pass",
    }
    out = ROOT / "P4" / "results" / "p4_public_api_smoke.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
