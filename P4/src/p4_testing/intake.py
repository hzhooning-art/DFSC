"""Non-destructive admission screening for historical defect candidates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class AdmissionState(str, Enum):
    CANDIDATE = "candidate"
    NEEDS_RUNNER = "needs_runner"
    NEEDS_REPLAY = "needs_replay"
    ADMITTED = "admitted"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class AdmissionRecord:
    state: AdmissionState
    reason: str
    upstream_url: str


def screen_metadata(candidate: Mapping[str, Any]) -> AdmissionRecord:
    url = candidate.get("upstream_url")
    if not isinstance(url, str) or not url.startswith("https://"):
        return AdmissionRecord(AdmissionState.EXCLUDED, "invalid_upstream_url", str(url or ""))
    if not candidate.get("buggy_version"):
        return AdmissionRecord(AdmissionState.EXCLUDED, "no_confirmed_buggy_version", url)
    if not candidate.get("fixed_version"):
        return AdmissionRecord(AdmissionState.EXCLUDED, "no_confirmed_fixed_version", url)
    digest = candidate.get("runner_sha256")
    if digest is None:
        return AdmissionRecord(AdmissionState.NEEDS_RUNNER, "runner_not_frozen", url)
    if not isinstance(digest, str) or len(digest) != 64:
        return AdmissionRecord(AdmissionState.EXCLUDED, "invalid_runner_digest", url)
    if candidate.get("clean_replays", 0) < 2:
        return AdmissionRecord(AdmissionState.NEEDS_REPLAY, "two_clean_replays_required", url)
    if candidate.get("buggy_confirmed") is not True or candidate.get("fixed_confirmed") is not True:
        return AdmissionRecord(AdmissionState.EXCLUDED, "paired_roles_not_confirmed", url)
    return AdmissionRecord(AdmissionState.ADMITTED, "paired_replay_confirmed_twice", url)


def screen_candidate(candidate: Mapping[str, Any], workspace: Path) -> AdmissionRecord:
    """Append a decision record; never edit a frozen manifest."""

    record = screen_metadata(candidate)
    workspace.mkdir(parents=True, exist_ok=True)
    audit_path = workspace / "admission_audit.jsonl"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {key: value for key, value in candidate.items() if "token" not in key.casefold() and "password" not in key.casefold()},
        "decision": {**asdict(record), "state": record.state.value},
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return record
