"""Multi-start alpha/beta inverse recovery with local uncertainty diagnostics."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dfsc


def prediction(theta: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    alpha, beta = theta
    return dfsc.mittag_leffler_e_ab(alpha, beta, -1.4 * times.pow(alpha), terms=140, method="series")


def fit(observed: torch.Tensor, times: torch.Tensor, start: tuple[float, float]) -> tuple[torch.Tensor, float]:
    raw = torch.tensor(start, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=0.5, max_iter=120, line_search_fn="strong_wolfe")

    def bounded() -> torch.Tensor:
        return torch.stack([0.25 + 1.25 * torch.sigmoid(raw[0]), 0.45 + 1.55 * torch.sigmoid(raw[1])])

    def closure():
        optimizer.zero_grad()
        loss = torch.mean((prediction(bounded(), times) - observed) ** 2)
        loss.backward()
        return loss

    optimizer.step(closure)
    estimate = bounded().detach()
    return estimate, float(torch.mean((prediction(estimate, times) - observed) ** 2))


def main() -> None:
    torch.set_default_dtype(torch.float64)
    output = ROOT / "revision_results"
    output.mkdir(parents=True, exist_ok=True)
    truth = torch.tensor([0.78, 1.15])
    starts = ((-2.0, -2.0), (-2.0, 0.0), (-2.0, 2.0), (0.0, -2.0), (0.0, 0.0), (0.0, 2.0), (2.0, -2.0), (2.0, 0.0), (2.0, 2.0))
    rows = []
    for sensors in (12, 32, 64):
        times = torch.linspace(0.02, 1.0, sensors)
        clean = prediction(truth, times)
        for noise in (0.0, 0.005, 0.02):
            for seed in range(3):
                generator = torch.Generator().manual_seed(7000 + sensors * 10 + seed)
                observed = clean + noise * torch.randn(clean.shape, generator=generator)
                estimates = [fit(observed, times, start) for start in starts]
                best, loss = min(estimates, key=lambda item: item[1])
                report = dfsc.local_identifiability(
                    lambda theta: 0.5 * torch.sum((prediction(theta, times) - observed) ** 2),
                    best,
                    noise_variance=max(noise**2, 1e-12),
                )
                successful = [estimate for estimate, value in estimates if value <= loss + 1e-8]
                spread = torch.stack(successful).std(dim=0, unbiased=False)
                rows.append(
                    {
                        "sensors": sensors,
                        "noise_std": noise,
                        "seed": seed,
                        "alpha": float(best[0]),
                        "beta": float(best[1]),
                        "alpha_error": float(torch.abs(best[0] - truth[0])),
                        "beta_error": float(torch.abs(best[1] - truth[1])),
                        "loss": loss,
                        "condition_number": report.condition_number,
                        "locally_identifiable": report.locally_identifiable,
                        "alpha_standard_error": float(report.standard_errors[0]),
                        "beta_standard_error": float(report.standard_errors[1]),
                        "parameter_correlation": float(report.correlation[0, 1]),
                        "multistart_alpha_spread": float(spread[0]),
                        "multistart_beta_spread": float(spread[1]),
                    }
                )
    payload = {
        "truth": {"alpha": float(truth[0]), "beta": float(truth[1])},
        "scope": "local Hessian uncertainty plus bounded multi-start recovery; not global identifiability proof",
        "rows": rows,
    }
    (output / "inverse_identifiability.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"cases": len(rows), "max_alpha_error": max(r["alpha_error"] for r in rows), "max_beta_error": max(r["beta_error"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
