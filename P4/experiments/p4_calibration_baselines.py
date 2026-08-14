"""Matched calibration baselines for a differentiable matrix-exponential action.

The experiment uses the same two-parameter forward model for Adam and SciPy
L-BFGS-B.  The latter is a classical derivative-free-from-the-framework
baseline here: SciPy obtains objective evaluations through a NumPy/SciPy
matrix exponential, while Adam uses the PyTorch matrix exponential and
autodiff.  The comparison is deliberately limited to this declared model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "P4" / "results" / "p4_calibration_baselines.json"


def matrix_torch(theta: torch.Tensor, times: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    A = torch.stack((-theta[0], torch.tensor(0.2, dtype=theta.dtype, device=theta.device)))
    A = torch.stack((A, torch.stack((torch.tensor(0.0, dtype=theta.dtype, device=theta.device), -theta[1]))))
    return torch.stack([torch.matrix_exp(A * t) @ x0 for t in times])


def matrix_numpy(theta: np.ndarray, times: np.ndarray, x0: np.ndarray) -> np.ndarray:
    A = np.array([[-theta[0], 0.2], [0.0, -theta[1]]], dtype=np.float64)
    return np.stack([expm(A * float(t)) @ x0 for t in times])


def main() -> None:
    times = torch.linspace(0.05, 1.5, 32, dtype=torch.float64)
    x0 = torch.tensor([1.0, -0.35], dtype=torch.float64)
    truth = np.array([0.72, 0.41], dtype=np.float64)
    rows = []

    for seed in range(5):
        rng = np.random.default_rng(20260820 + seed)
        target_np = matrix_numpy(truth, times.numpy(), x0.numpy())
        target_np += 0.002 * rng.normal(size=target_np.shape)
        target = torch.tensor(target_np, dtype=torch.float64)

        raw = torch.tensor([0.95, 0.80], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([raw], lr=0.08)
        start = time.perf_counter()
        for _ in range(140):
            optimizer.zero_grad(set_to_none=True)
            pred = matrix_torch(raw, times, x0)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                raw.clamp_(0.05, 2.0)
        adam_elapsed = time.perf_counter() - start
        adam_theta = raw.detach().numpy().copy()
        adam_loss = float(torch.mean((matrix_torch(raw.detach(), times, x0) - target) ** 2))

        calls = {"n": 0}

        def objective(theta: np.ndarray) -> float:
            calls["n"] += 1
            return float(np.mean((matrix_numpy(theta, times.numpy(), x0.numpy()) - target_np) ** 2))

        start = time.perf_counter()
        result = minimize(objective, np.array([0.95, 0.80]), method="L-BFGS-B", bounds=[(0.05, 2.0), (0.05, 2.0)], options={"maxiter": 140, "ftol": 1e-14})
        lbfgs_elapsed = time.perf_counter() - start
        lbfgs_theta = np.asarray(result.x, dtype=np.float64)
        rows.append({
            "seed": seed,
            "adam": {"theta": adam_theta.tolist(), "parameter_l2_error": float(np.linalg.norm(adam_theta - truth)), "final_loss": adam_loss, "elapsed_seconds": adam_elapsed, "iterations": 140},
            "lbfgs_b": {"theta": lbfgs_theta.tolist(), "parameter_l2_error": float(np.linalg.norm(lbfgs_theta - truth)), "final_loss": float(result.fun), "elapsed_seconds": lbfgs_elapsed, "function_evaluations": calls["n"], "success": bool(result.success)},
        })

    def aggregate(name: str) -> dict:
        return {
            "parameter_l2_error_mean": float(np.mean([r[name]["parameter_l2_error"] for r in rows])),
            "parameter_l2_error_std": float(np.std([r[name]["parameter_l2_error"] for r in rows], ddof=1)),
            "final_loss_mean": float(np.mean([r[name]["final_loss"] for r in rows])),
            "elapsed_seconds_mean": float(np.mean([r[name]["elapsed_seconds"] for r in rows])),
        }

    payload = {"model": "2x2 stable matrix-exponential action", "seeds": 5, "noise_sigma": 0.002, "truth": truth.tolist(), "adam": aggregate("adam"), "lbfgs_b": aggregate("lbfgs_b"), "per_seed": rows, "scope": "matched low-dimensional calibration benchmark; not a universal optimizer ranking"}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
