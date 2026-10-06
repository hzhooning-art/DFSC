"""Publication gates for the EMSE route."""

from __future__ import annotations

from typing import Any


def evaluate_gates(
    *,
    families: int,
    projects: int,
    independent_reproduction: bool,
    independent_logs: bool = False,
    lopo_complete: bool = False,
    meaningful_increment: bool = False,
    synchronized: bool = False,
) -> dict[str, Any]:
    checks = {
        "benchmark": families >= 25 and projects >= 7,
        "independent_logs": independent_logs,
        "transfer": lopo_complete,
        "claim": meaningful_increment,
        "reproduction": independent_reproduction,
        "synchronization": synchronized,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return {
        "claim_status": "OPEN" if not failed else "BLOCKED",
        "failed_gates": failed,
        "checks": checks,
        "families": families,
        "projects": projects,
    }
