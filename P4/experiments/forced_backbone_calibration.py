"""Calibrate the numerical envelope of P1's forced MLSL primitive."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P1" / "paper1_mlsl"))
from dfsc.factory import MLSLConfig, build_dirichlet_mlsl_1d  # noqa: E402
from dfsc.forced_layer import ForcedMittagLefflerSpectralLayer  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MLSLConfig(terms=40, dtype=torch.float64, device=device)
    _, base = build_dirichlet_mlsl_1d(num_points=16, num_modes=8, config=cfg)
    x = torch.linspace(0.0, 1.0, 16, dtype=torch.float64, device=device)
    u0 = torch.sin(torch.pi * x)
    rows = []
    for alpha in [0.65, 0.75, 0.85, 0.95]:
        for t_value in [0.05, 0.10, 0.20, 0.40, 0.80]:
            for q in [4, 8, 16, 32]:
                forcing_times = torch.linspace(0.05, 0.95, q, dtype=torch.float64, device=device)
                forcing = torch.sin(torch.pi * x)[None, :].repeat(q, 1)
                for terms in [8, 16, 32, 64]:
                    layer = ForcedMittagLefflerSpectralLayer(
                        base, forcing_terms=terms, ml_method="hybrid"
                    ).to(device)
                    with torch.no_grad():
                        value = layer(
                            u0,
                            torch.tensor([t_value], dtype=torch.float64, device=device),
                            torch.tensor(alpha, dtype=torch.float64, device=device),
                            forcing,
                            forcing_times,
                            beta=torch.tensor(2.0, dtype=torch.float64, device=device),
                        )
                    rows.append({
                        "alpha": alpha, "t": t_value, "q": q, "terms": terms,
                        "max_abs": float(value.abs().max().cpu()),
                        "finite": bool(torch.isfinite(value).all().item()),
                    })
    safe = [r for r in rows if r["finite"] and r["max_abs"] <= 1.0e6]
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_cases": len(rows),
        "safe_cases": len(safe),
        "safe_fraction": len(safe) / len(rows),
        "safe_max_abs": max((r["max_abs"] for r in safe), default=None),
        "unsafe_min_abs": min((r["max_abs"] for r in rows if r not in safe), default=None),
        "safe_region_summary": {
            "max_alpha": max((r["alpha"] for r in safe), default=None),
            "max_t": max((r["t"] for r in safe), default=None),
            "q_values": sorted(set(r["q"] for r in safe)),
            "terms_values": sorted(set(r["terms"] for r in safe)),
        },
        "rows": rows,
        "interpretation": "empirical numerical envelope; not a rigorous error bound",
    }
    out = ROOT / "P4" / "results" / "p4_forced_backbone_calibration.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
