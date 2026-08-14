"""Risk--coverage analysis for the completed P4 joint-stability probe."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def stats(rows: list[dict[str, object]], threshold: float, agreement: bool) -> dict[str, object]:
    accepted = [row for row in rows if float(row["confidence"]) >= threshold and (not agreement or bool(row["agreement"]))]
    regrets = torch.tensor([float(row["regret"]) for row in accepted]) if accepted else torch.tensor([])
    return {
        "threshold": threshold,
        "agreement_required": agreement,
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "accepted_count": len(accepted),
        "mean_regret": float(regrets.mean()) if len(regrets) else None,
        "max_regret": float(regrets.max()) if len(regrets) else None,
        "q90_regret": float(torch.quantile(regrets, 0.9)) if len(regrets) else None,
    }


def main() -> None:
    root = Path(__file__).parents[1]
    source = root / "results" / "p4_joint_stability_proxy.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload["rows"]
    thresholds = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)
    curves = {}
    for noise in (0.0, 0.05):
        group = [row for row in rows if float(row["noise"]) == noise]
        curves[f"noise={noise}|confidence_only"] = [stats(group, threshold, False) for threshold in thresholds]
        curves[f"noise={noise}|confidence_and_agreement"] = [stats(group, threshold, True) for threshold in thresholds]
    out = root / "results" / "p4_risk_coverage_curves.json"
    result = {
    "source": source.relative_to(root).as_posix(),
    "thresholds": thresholds,
    "curves": curves,
    "warning": "Small diagnostic sample inherited from the joint-stability probe; expand seeds before publication.",
}
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
