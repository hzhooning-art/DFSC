"""Scan series and hybrid Mittag-Leffler evaluators on negative real inputs."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import mittag_leffler_e


def summarize(alpha_value: float) -> None:
    alpha = torch.tensor(alpha_value)
    z = -torch.linspace(0.0, 80.0, 81)

    series = mittag_leffler_e(alpha, z, terms=120, custom_backward=False, method="series")
    hybrid = mittag_leffler_e(alpha, z, terms=120, custom_backward=False, method="hybrid")

    series_finite = torch.isfinite(series)
    hybrid_finite = torch.isfinite(hybrid)
    overlap = torch.abs(z) <= 8.0
    overlap_error = torch.max(torch.abs(series[overlap] - hybrid[overlap])).item()

    print(f"alpha={alpha_value}")
    print("  series finite:", int(series_finite.sum()), "/", series.numel())
    print("  hybrid finite:", int(hybrid_finite.sum()), "/", hybrid.numel())
    print("  max overlap abs error:", f"{overlap_error:.3e}")
    print("  hybrid tail value z=-80:", f"{hybrid[-1].item():.6e}")
    if not bool(series_finite.all()):
        first_bad = int((~series_finite).nonzero()[0].item())
        print("  first non-finite series z:", z[first_bad].item())
    else:
        max_abs = torch.max(torch.abs(series)).item()
        print("  max abs series:", "inf" if math.isinf(max_abs) else f"{max_abs:.3e}")


def main() -> None:
    torch.set_default_dtype(torch.float64)
    summarize(0.65)
    summarize(1.35)


if __name__ == "__main__":
    main()
