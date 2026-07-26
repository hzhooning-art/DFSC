"""Threshold sensitivity for the hybrid Mittag-Leffler evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc.mittag_leffler import mittag_leffler_e_hybrid


def mp_mittag_leffler(alpha: float, z: float, *, dps: int = 80, tol: str = "1e-70") -> float:
    mp.mp.dps = dps
    total = mp.mpf("0")
    for k in range(10000):
        term = mp.mpf(z) ** k / mp.gamma(mp.mpf(alpha) * k + 1)
        total += term
        if abs(term) < mp.mpf(tol):
            break
    return float(total)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    z_values = [-8.0, -6.0, -4.0, -2.0, -1.0]
    print("alpha,threshold,max_abs_error_on_reference_points,tail_z_minus_80")
    for alpha_value in [0.65, 1.35]:
        alpha = torch.tensor(alpha_value)
        refs = torch.tensor([mp_mittag_leffler(alpha_value, z) for z in z_values])
        for threshold in [4.0, 6.0, 8.0, 10.0]:
            z = torch.tensor(z_values)
            pred = mittag_leffler_e_hybrid(alpha, z, series_terms=140, threshold=threshold)
            err = torch.max(torch.abs(pred - refs)).item()
            tail = mittag_leffler_e_hybrid(
                alpha,
                torch.tensor([-80.0]),
                series_terms=140,
                threshold=threshold,
            )[0].item()
            print(f"{alpha_value},{threshold},{err:.6e},{tail:.6e}")


if __name__ == "__main__":
    main()
