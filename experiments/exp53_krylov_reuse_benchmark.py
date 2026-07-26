"""Accuracy and amortization benchmark for explicit prepared Lanczos bases."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dfsc


def synchronized_time(fn, device: torch.device, repeats: int = 5) -> float:
    values = []
    for _ in range(repeats):
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device.type == "cuda": torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return sorted(values)[len(values) // 2]


def run(device: torch.device) -> list[dict[str, object]]:
    rows = []
    dtype = torch.float64
    for size in (64, 128, 256):
        diagonal = torch.linspace(0.1, 8.0, size, dtype=dtype, device=device)
        operator = torch.diag(diagonal)
        u0 = torch.randn(4, size, dtype=dtype, device=device)
        prepared = dfsc.prepare_lanczos_basis(operator, u0, krylov_dimension=32)
        for queries in (4, 16, 64):
            times = torch.linspace(0.0, 0.4, 8, dtype=dtype, device=device)
            alphas = torch.linspace(0.55, 0.95, queries, dtype=dtype, device=device)

            def direct():
                return [dfsc.lanczos_mittag_leffler_action(operator, u0, times, alpha, krylov_dimension=32)[0] for alpha in alphas]

            def reused():
                return [dfsc.apply_prepared_lanczos_basis(prepared, times, alpha) for alpha in alphas]

            direct_values = direct()
            reused_values = reused()
            error = max(
                float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(1e-14))
                for a, b in zip(direct_values, reused_values, strict=True)
            )
            direct_seconds = synchronized_time(direct, device, repeats=3)
            reuse_seconds = synchronized_time(reused, device, repeats=3)
            rows.append(
                {
                    "device": str(device), "size": size, "batch": 4, "queries": queries,
                    "krylov_dimension": 32, "relative_error": error,
                    "direct_seconds": direct_seconds, "prepared_query_seconds": reuse_seconds,
                    "query_speedup_excluding_preparation": direct_seconds / reuse_seconds,
                }
            )
    return rows


def main() -> None:
    torch.set_default_dtype(torch.float64)
    output = ROOT / "revision_results"
    output.mkdir(parents=True, exist_ok=True)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available(): devices.append(torch.device("cuda"))
    rows = [row for device in devices for row in run(device)]
    payload = {
        "scope": "repeated alpha queries for a fixed operator and initial-state batch",
        "timing_boundary": "speedup excludes one-time basis preparation and is not an end-to-end solver comparison",
        "rows": rows,
    }
    (output / "krylov_reuse_benchmark.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"devices": [str(d) for d in devices], "cases": len(rows), "max_error": max(r["relative_error"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
