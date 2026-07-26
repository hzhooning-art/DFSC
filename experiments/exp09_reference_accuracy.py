"""High-precision reference check for Mittag-Leffler evaluators.

The reference is an mpmath power series on a controlled interval. This is not a
large-argument reference solver; it verifies the safe transition region used by
the current prototype.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import mittag_leffler_e


def mp_mittag_leffler(alpha: float, z: float, *, dps: int = 80, tol: str = "1e-70") -> float:
    mp.mp.dps = dps
    a = mp.mpf(alpha)
    zz = mp.mpf(z)
    total = mp.mpf("0")
    threshold = mp.mpf(tol)
    for k in range(10000):
        term = zz**k / mp.gamma(a * k + 1)
        total += term
        if abs(term) < threshold:
            break
    return float(total)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    z_values = [-8.0, -6.0, -4.0, -2.0, -1.0, 0.0]
    print("alpha,z,reference,series,hybrid,series_abs_error,hybrid_abs_error")
    for alpha_value in [0.65, 1.35]:
        alpha = torch.tensor(alpha_value)
        for z_value in z_values:
            z = torch.tensor(z_value)
            ref = mp_mittag_leffler(alpha_value, z_value)
            series = mittag_leffler_e(alpha, z, terms=140, method="series").item()
            hybrid = mittag_leffler_e(alpha, z, terms=140, method="hybrid").item()
            print(
                f"{alpha_value},{z_value},{ref:.16e},{series:.16e},{hybrid:.16e},"
                f"{abs(series - ref):.3e},{abs(hybrid - ref):.3e}"
            )


if __name__ == "__main__":
    main()
