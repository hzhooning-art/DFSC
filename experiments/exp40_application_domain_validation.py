"""Validate the four domain templates and their application-level invariants."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dfsc


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(torch.finfo(reference.dtype).eps)
    return float((torch.linalg.vector_norm(actual - reference) / denominator).detach().cpu())


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(40)

    alpha = torch.tensor(0.8, requires_grad=True)
    beta = torch.tensor(1.6, requires_grad=True)
    diffusion = dfsc.anomalous_diffusion_case(
        initial=lambda x: torch.sin(torch.pi * x),
        times=torch.tensor([0.0, 0.02, 0.05]),
        alpha=alpha,
        beta=beta,
        diffusivity=0.1,
        num_points=32,
        num_modes=12,
    )
    diffusion_solution = diffusion.solve()
    diffusion_solution.values[-1].square().mean().backward()

    relaxation = dfsc.assembled_relaxation_case(
        stiffness=torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
        mass=torch.diag(torch.tensor([2.0, 1.0])),
        initial=torch.tensor([1.0, 0.0]),
        times=torch.tensor([0.0, 0.1]),
        alpha=0.9,
    ).solve()

    network = dfsc.network_diffusion_case(
        adjacency=torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        initial=torch.ones(3),
        times=torch.tensor([0.0, 0.2, 0.7]),
        alpha=0.75,
    ).solve()
    graph_constant_error = float(torch.max(torch.abs(network.values - 1.0)))

    transport_alpha = torch.tensor(1.0, requires_grad=True)
    diffusivity = torch.tensor(0.02, requires_grad=True)
    velocity = torch.tensor(0.15, requires_grad=True)
    times = torch.tensor([0.0, 0.01, 0.02])
    initial = torch.linspace(-0.7, 0.9, 8)
    transport_case = dfsc.advection_diffusion_case(
        initial=initial,
        times=times,
        alpha=transport_alpha,
        diffusivity=diffusivity,
        velocity=velocity,
        num_points=8,
        arnoldi_dimension=8,
    )
    transport = transport_case.solve()
    reference = torch.stack(
        [torch.matrix_exp(-time_value * transport_case.problem.operator) @ initial for time_value in times]
    )
    transport_relative_error = relative_error(transport.values, reference)
    weights = torch.linspace(0.2, 1.1, 8)
    (transport.values[-1] * weights).sum().backward()

    summary = {
        "catalog_size": len(dfsc.application_catalog()),
        "anomalous_diffusion_finite": bool(torch.isfinite(diffusion_solution.values).all()),
        "anomalous_diffusion_alpha_gradient_finite": bool(torch.isfinite(alpha.grad)),
        "anomalous_diffusion_beta_gradient_finite": bool(torch.isfinite(beta.grad)),
        "assembled_relaxation_mass_projection": relaxation.stats.get("mass_projection") is True,
        "graph_constant_mode_max_error": graph_constant_error,
        "advection_diffusion_exponential_relative_error": transport_relative_error,
        "advection_diffusion_alpha_gradient_finite": bool(torch.isfinite(transport_alpha.grad)),
        "advection_diffusion_diffusivity_gradient_finite": bool(torch.isfinite(diffusivity.grad)),
        "advection_diffusion_velocity_gradient_finite": bool(torch.isfinite(velocity.grad)),
        "scope": "four tested templates centered on differentiable fractional spectral propagation",
    }
    output = ROOT / "results" / "application_domain_validation_summary.json"
    try:
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except PermissionError:
        output = ROOT / "results" / "tables" / f"application_domain_validation_{int(time.time())}.json"
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["written_to"] = str(output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
