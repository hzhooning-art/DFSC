"""Pre-registered execution manifest for the independent baseline study."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_manifest(manifest: Mapping[str, Any]) -> bool:
    required = {"schema", "benchmark_digest", "case_ids", "strategies", "max_invocations", "wall_seconds", "seed", "analysis_version"}
    if set(manifest) != required or manifest["schema"] != "P4-Independent-Baseline-Preregistration-v1":
        return False
    if not isinstance(manifest["benchmark_digest"], str) or len(manifest["benchmark_digest"]) != 64:
        return False
    if not isinstance(manifest["case_ids"], list) or not manifest["case_ids"] or len(set(manifest["case_ids"])) != len(manifest["case_ids"]):
        return False
    if not isinstance(manifest["strategies"], list) or not manifest["strategies"] or len(set(manifest["strategies"])) != len(manifest["strategies"]):
        return False
    if isinstance(manifest["max_invocations"], bool) or not isinstance(manifest["max_invocations"], int) or manifest["max_invocations"] <= 0:
        return False
    if not isinstance(manifest["wall_seconds"], (int, float)) or manifest["wall_seconds"] <= 0:
        return False
    if isinstance(manifest["seed"], bool) or not isinstance(manifest["seed"], int):
        return False
    return isinstance(manifest["analysis_version"], str) and bool(manifest["analysis_version"])
