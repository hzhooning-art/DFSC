"""Expanded P4 joint-stability validation and risk--coverage sweep."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from joint_stability_proxy import one  # noqa: E402


def curve(rows: list[dict[str, object]], threshold: float, agreement: bool) -> dict[str, object]:
    accepted = [row for row in rows if float(row["confidence"]) >= threshold and (not agreement or bool(row["agreement"]))]
    regret = torch.tensor([float(row["regret"]) for row in accepted]) if accepted else torch.tensor([])
    return {"threshold": threshold, "agreement_required": agreement, "coverage": len(accepted) / len(rows), "accepted_count": len(accepted), "mean_regret": float(regret.mean()) if len(regret) else None, "max_regret": float(regret.max()) if len(regret) else None}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = (71, 73, 79, 83, 89)
    rows = [one(kind, seed, noise, end, device) for kind in ("tempered_power_law", "two_timescale") for noise in (0.0, 0.025, 0.05) for end in (10.0, 14.0) for seed in seeds]
    curves = {}
    for noise in (0.0, 0.025, 0.05):
        group = [row for row in rows if float(row["noise"]) == noise]
        for agreement in (False, True):
            curves[f"noise={noise}|agreement={agreement}"] = [curve(group, threshold, agreement) for threshold in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)]
    out = Path(__file__).parents[1] / "results" / "p4_expanded_joint_validation.json"
    payload = {"device": str(device), "seeds": seeds, "bootstrap_draws": 128, "rows": rows, "curves": curves, "warning": "Synthetic mechanism families; add physical datasets before final publication."}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "rows": len(rows), "curves": curves}, indent=2))


if __name__ == "__main__":
    main()
