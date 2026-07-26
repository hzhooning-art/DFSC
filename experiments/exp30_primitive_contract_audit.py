"""Primitive-contract audit for MLSL.

This experiment is intentionally not a new solver benchmark. It checks whether
the proposed spectral layer behaves like a reusable SciML primitive: stable
shapes, trainable fractional orders, boundary constructors, accelerator
placement, and reduced-precision smoke tests.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import MLSLConfig, build_mlsl


RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_field(x: torch.Tensor, batch: int | None = None) -> torch.Tensor:
    if x.ndim == 1:
        field = torch.sin(torch.pi * x) + 0.25 * torch.sin(3.0 * torch.pi * x)
    else:
        field = torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])
    if batch is None:
        return field
    return torch.stack([(1.0 + 0.05 * i) * field for i in range(batch)])


def check_forward_contract(device: str, dtype: torch.dtype) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    x, layer = build_mlsl(
        dimension=1,
        boundary="dirichlet",
        num_points=96,
        num_modes=20,
        config=MLSLConfig.stable(terms=100, dtype=dtype, device=device),
    )
    times = torch.linspace(0.0, 0.05, 6, dtype=dtype, device=device)
    alpha = torch.tensor(0.85, dtype=dtype, device=device, requires_grad=True)
    beta = torch.tensor(1.45, dtype=dtype, device=device, requires_grad=True)

    for name, u0, expected_shape in [
        ("single_vector_time", make_field(x), (6, 96)),
        ("batch_vector_time", make_field(x, batch=4), (4, 6, 96)),
    ]:
        if alpha.grad is not None:
            alpha.grad.zero_()
        if beta.grad is not None:
            beta.grad.zero_()
        out = layer(u0, times, alpha, beta=beta)
        loss = out.square().mean()
        loss.backward()
        rows.append(
            {
                "axis": "shape_autograd",
                "case": name,
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "observed": str(tuple(out.shape)),
                "expected": str(expected_shape),
                "passed": tuple(out.shape) == expected_shape
                and bool(torch.isfinite(out).all().item())
                and bool(torch.isfinite(alpha.grad).item())
                and bool(torch.isfinite(beta.grad).item()),
            }
        )
    return rows


def check_constructors() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dimension in [1, 2]:
        for boundary in ["dirichlet", "neumann", "periodic", "mixed_dn", "mixed_nd"]:
            kwargs: dict[str, object]
            if dimension == 1:
                kwargs = {"num_points": 48, "num_modes": 10}
            else:
                kwargs = {"num_points_1d": 10, "num_modes_1d": 4}
            x, layer = build_mlsl(
                dimension=dimension,
                boundary=boundary,
                config=MLSLConfig.stable(terms=80, dtype=torch.float64),
                **kwargs,
            )
            u0 = make_field(x)
            alpha = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
            beta = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
            out = layer(u0, torch.linspace(0.0, 0.04, 4), alpha, beta=beta)
            out.square().mean().backward()
            rows.append(
                {
                    "axis": "constructor",
                    "case": f"{dimension}d_{boundary}",
                    "device": "cpu",
                    "dtype": "float64",
                    "observed": str(tuple(out.shape)),
                    "expected": "finite output and alpha/beta grads",
                    "passed": bool(torch.isfinite(out).all().item())
                    and bool(torch.isfinite(alpha.grad).item())
                    and bool(torch.isfinite(beta.grad).item()),
                }
            )
    return rows


def check_gpu_precision() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not torch.cuda.is_available():
        return [
            {
                "axis": "gpu_precision",
                "case": "cuda_unavailable",
                "device": "cuda",
                "dtype": "n/a",
                "observed": "CUDA unavailable",
                "expected": "CUDA available",
                "passed": False,
            }
        ]

    for dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16]:
        try:
            x, layer = build_mlsl(
                dimension=1,
                boundary="periodic",
                num_points=64,
                num_modes=17,
                config=MLSLConfig.stable(terms=80, dtype=dtype, device="cuda"),
            )
            u0 = make_field(x, batch=8)
            alpha = torch.tensor(0.9, dtype=dtype, device="cuda")
            beta = torch.tensor(1.4, dtype=dtype, device="cuda")
            out = layer(u0, torch.linspace(0.0, 0.03, 5, dtype=dtype, device="cuda"), alpha, beta=beta)
            torch.cuda.synchronize()
            passed = bool(torch.isfinite(out).all().item()) and tuple(out.shape) == (8, 5, 64)
            observed = str(tuple(out.shape))
        except Exception as exc:  # noqa: BLE001 - records capability boundary.
            passed = False
            observed = type(exc).__name__ + ": " + str(exc)[:120]
        rows.append(
            {
                "axis": "gpu_precision",
                "case": "batched_periodic_forward",
                "device": "cuda",
                "dtype": str(dtype).replace("torch.", ""),
                "observed": observed,
                "expected": "(8, 5, 64), finite",
                "passed": passed,
            }
        )
    return rows


def main() -> None:
    torch.set_default_dtype(torch.float64)
    RESULTS.mkdir(exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    rows.extend(check_forward_contract("cpu", torch.float64))
    rows.extend(check_constructors())
    rows.extend(check_gpu_precision())

    write_csv(TABLES / "primitive_contract_audit.csv", rows)
    total = len(rows)
    passed = sum(1 for row in rows if bool(row["passed"]))
    summary = {
        "total_checks": total,
        "passed_checks": passed,
        "failed_checks": total - passed,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    (RESULTS / "primitive_contract_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
