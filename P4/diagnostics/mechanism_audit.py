"""Model-agnostic mechanism audit utilities retained from the P4 route.

This module does not fit a solver. It consumes candidate scores or validation
diagnostics and returns an auditable accept/abstain decision for DFSC/P3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class AuditDecision:
    selected: str
    confidence: float
    temporal_agreement: bool
    accepted: bool
    reason: str


def audit_candidates(
    candidates: Mapping[str, float],
    *,
    early_winner: str | None = None,
    late_winner: str | None = None,
    minimum_confidence: float = 0.75,
    require_temporal_agreement: bool = True,
) -> AuditDecision:
    """Create a deterministic accept/abstain decision from candidate scores.

    Scores are validation losses, so lower is better. The confidence is the
    relative gap between the two best candidates; it is intentionally an
    empirical diagnostic rather than a probability or error bound.
    """
    if len(candidates) < 2:
        raise ValueError("at least two candidate scores are required")
    ordered = sorted(candidates.items(), key=lambda item: item[1])
    best, second = ordered[0], ordered[1]
    denominator = max(abs(second[1]), 1e-12)
    confidence = max(0.0, min(1.0, (second[1] - best[1]) / denominator))
    agreement = early_winner is None or late_winner is None or early_winner == late_winner == best[0]
    accepted = confidence >= minimum_confidence and (agreement or not require_temporal_agreement)
    if accepted:
        reason = "confidence and temporal agreement passed"
    elif confidence < minimum_confidence:
        reason = "validation margin below threshold"
    else:
        reason = "validation winners disagree across time"
    return AuditDecision(best[0], confidence, agreement, accepted, reason)


def risk_coverage(
    records: Iterable[Mapping[str, object]],
    *,
    threshold: float,
    require_temporal_agreement: bool = True,
) -> dict[str, float | int | None]:
    """Summarize accepted coverage and regret for previously audited records."""
    rows = list(records)
    accepted = [
        row for row in rows
        if float(row["confidence"]) >= threshold
        and (not require_temporal_agreement or bool(row["agreement"]))
    ]
    regrets = [float(row["regret"]) for row in accepted]
    return {
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "accepted_count": len(accepted),
        "mean_regret": sum(regrets) / len(regrets) if regrets else None,
        "max_regret": max(regrets) if regrets else None,
    }
