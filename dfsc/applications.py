"""Domain-oriented application templates built from dfsc primitives.

The templates translate common fractional-model parameters into the existing
problem--algorithm--solve interface.  They do not introduce separate solvers;
their purpose is to make the validated dfsc core usable without requiring each
application to reconstruct operators and metadata by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import torch

from .factory import MLSLConfig, build_mlsl

from .algorithms import MLSLArnoldi, MLSLGeneralizedOperator, MLSLGraph, MLSLStable
from .problems import (
    FractionalSpectralProblem,
    GeneralizedOperatorSpectralProblem,
    GeneralOperatorProblem,
    GraphSpectralProblem,
    Solution,
)


@dataclass(frozen=True)
class ApplicationProfile:
    """Machine-readable statement of a domain's fit to dfsc."""

    name: str
    domain: str
    fit: str
    entrypoint: str
    advantage: str
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "domain": self.domain,
            "fit": self.fit,
            "entrypoint": self.entrypoint,
            "advantage": self.advantage,
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
        }


@dataclass
class ApplicationCase:
    """A configured domain problem with assumptions and a recommended method."""

    name: str
    domain: str
    problem: Any
    recommended_algorithm: Any
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    differentiable_parameters: tuple[str, ...]
    coordinates: torch.Tensor | None = None

    def solve(self, algorithm: Any | None = None) -> Solution:
        """Solve the case with its recommended algorithm unless overridden."""

        from .solvers import solve

        selected = self.recommended_algorithm if algorithm is None else algorithm
        return solve(self.problem, selected)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "domain": self.domain,
            "problem_type": type(self.problem).__name__,
            "recommended_algorithm": self.recommended_algorithm.name,
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "differentiable_parameters": list(self.differentiable_parameters),
        }


_APPLICATION_PROFILES: tuple[ApplicationProfile, ...] = (
    ApplicationProfile(
        name="anomalous-diffusion",
        domain="time/space-fractional transport on regular domains",
        fit="native",
        entrypoint="dfsc.anomalous_diffusion_case",
        advantage="Direct batched time queries and trainable temporal/spatial orders without history storage.",
        assumptions=("Known Laplacian spectral basis", "Homogeneous evolution or a dfsc forcing wrapper"),
        limitations=("One- and two-dimensional tensor-product domains", "Constant fractional orders"),
    ),
    ApplicationProfile(
        name="assembled-fractional-relaxation",
        domain="finite-element diffusion and linear viscoelastic relaxation",
        fit="native-after-discretization",
        entrypoint="dfsc.assembled_relaxation_case",
        advantage="Mass-aware modal propagation with autograd-compatible fractional orders.",
        assumptions=("Symmetric positive-semidefinite stiffness", "Symmetric positive-definite mass"),
        limitations=("User supplies the assembled matrices", "Linear retained dynamics"),
    ),
    ApplicationProfile(
        name="network-memory-diffusion",
        domain="undirected network and graph-signal fractional diffusion",
        fit="native-after-graph-construction",
        entrypoint="dfsc.network_diffusion_case",
        advantage="Known graph diffusion is exposed as a differentiable layer for downstream neural models.",
        assumptions=("Undirected nonnegative adjacency", "Graph Laplacian propagation"),
        limitations=("Dense adjacency in the direct adapter", "Directed graphs require the Arnoldi path"),
    ),
    ApplicationProfile(
        name="fractional-advection-diffusion",
        domain="periodic non-self-adjoint transport",
        fit="controlled",
        entrypoint="dfsc.advection_diffusion_case",
        advantage="Differentiable Arnoldi actions avoid an eigenvector decomposition of the non-self-adjoint operator.",
        assumptions=("Periodic centered finite differences", "Moderate reduced Mittag-Leffler argument radius"),
        limitations=("Current validated reduced radius is at most four", "No upwind or nonlinear flux model"),
    ),
)


def application_catalog() -> list[dict[str, object]]:
    """Return the application domains currently represented by tested templates."""

    return [profile.to_dict() for profile in _APPLICATION_PROFILES]


def _initial_field(
    coordinates: torch.Tensor,
    initial: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    values = initial(coordinates) if callable(initial) else initial
    values = torch.as_tensor(values, dtype=coordinates.dtype, device=coordinates.device)
    expected = coordinates.shape[0]
    if values.shape[-1] != expected:
        raise ValueError(f"initial field must have trailing dimension {expected}")
    return values


def anomalous_diffusion_case(
    *,
    initial: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    times: torch.Tensor,
    alpha: torch.Tensor | float,
    beta: torch.Tensor | float = 2.0,
    diffusivity: float = 1.0,
    dimension: int = 1,
    boundary: str = "dirichlet",
    num_points: int = 64,
    num_modes: int = 24,
    length: float = 1.0,
    config: MLSLConfig | None = None,
) -> ApplicationCase:
    """Build a regular-domain anomalous-diffusion application case."""

    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")
    cfg = MLSLConfig.stable() if config is None else config
    cfg = replace(cfg, wave_speed=diffusivity**0.5)
    if dimension == 1:
        coordinates, layer = build_mlsl(
            dimension=1,
            boundary=boundary,
            num_points=num_points,
            num_modes=num_modes,
            length=length,
            config=cfg,
        )
    else:
        coordinates, layer = build_mlsl(
            dimension=dimension,
            boundary=boundary,
            num_points_1d=num_points,
            num_modes_1d=num_modes,
            length=length,
            config=cfg,
        )
    u0 = _initial_field(coordinates, initial)
    problem = FractionalSpectralProblem(
        layer=layer,
        u0=u0,
        times=times,
        alpha=alpha,
        beta=beta,
        metadata={"application": "anomalous-diffusion", "diffusivity": diffusivity, "boundary": boundary},
    )
    return ApplicationCase(
        name="anomalous-diffusion",
        domain="fractional transport",
        problem=problem,
        recommended_algorithm=MLSLStable(terms=cfg.terms, wave_speed=cfg.wave_speed, beta=cfg.beta),
        assumptions=("regular tensor-product domain", "homogeneous linear propagation"),
        limitations=("constant orders", "spectral truncation controls spatial resolution"),
        differentiable_parameters=("alpha", "beta"),
        coordinates=coordinates,
    )


def assembled_relaxation_case(
    *,
    stiffness: torch.Tensor,
    mass: torch.Tensor,
    initial: torch.Tensor,
    times: torch.Tensor,
    alpha: torch.Tensor | float,
    beta: torch.Tensor | float = 2.0,
    num_modes: int | None = None,
) -> ApplicationCase:
    """Build a mass-aware fractional relaxation case from assembled matrices."""

    problem = GeneralizedOperatorSpectralProblem(
        stiffness=stiffness,
        mass=mass,
        u0=initial,
        times=times,
        alpha=alpha,
        beta=beta,
        num_modes=num_modes,
        metadata={"application": "assembled-fractional-relaxation"},
    )
    return ApplicationCase(
        name="assembled-fractional-relaxation",
        domain="finite-element diffusion or linear viscoelastic relaxation",
        problem=problem,
        recommended_algorithm=MLSLGeneralizedOperator(num_modes=num_modes),
        assumptions=("symmetric PSD stiffness", "symmetric positive-definite mass"),
        limitations=("linear constitutive evolution", "assembled matrices supplied by the user"),
        differentiable_parameters=("alpha", "beta"),
    )


def network_diffusion_case(
    *,
    adjacency: torch.Tensor,
    initial: torch.Tensor,
    times: torch.Tensor,
    alpha: torch.Tensor | float,
    beta: torch.Tensor | float = 2.0,
    normalized: bool = False,
    num_modes: int | None = None,
) -> ApplicationCase:
    """Build an undirected graph fractional-diffusion case."""

    problem = GraphSpectralProblem(
        adjacency=adjacency,
        u0=initial,
        times=times,
        alpha=alpha,
        beta=beta,
        normalized=normalized,
        num_modes=num_modes,
        metadata={"application": "network-memory-diffusion"},
    )
    return ApplicationCase(
        name="network-memory-diffusion",
        domain="network dynamics and graph signal propagation",
        problem=problem,
        recommended_algorithm=MLSLGraph(num_modes=num_modes, normalized=normalized),
        assumptions=("undirected adjacency", "Laplacian-governed linear propagation"),
        limitations=("dense direct graph adapter", "directed networks require a general operator"),
        differentiable_parameters=("alpha", "beta"),
    )


def periodic_advection_diffusion_operator_1d(
    num_points: int,
    *,
    length: float = 1.0,
    diffusivity: torch.Tensor | float = 0.01,
    velocity: torch.Tensor | float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a centered periodic discretization of ``-kappa dxx + v dx``."""

    if num_points < 3:
        raise ValueError("num_points must be at least three")
    if length <= 0.0:
        raise ValueError("length must be positive")
    kappa = torch.as_tensor(diffusivity, dtype=dtype, device=device)
    speed = torch.as_tensor(velocity, dtype=dtype, device=device)
    if kappa.ndim != 0 or speed.ndim != 0:
        raise ValueError("diffusivity and velocity must be scalars")
    if float(kappa.detach().cpu()) < 0.0:
        raise ValueError("diffusivity must be nonnegative")
    dx = length / num_points
    identity = torch.eye(num_points, dtype=dtype, device=device)
    plus = torch.roll(identity, shifts=-1, dims=1)
    minus = torch.roll(identity, shifts=1, dims=1)
    first = (plus - minus) / (2.0 * dx)
    second = (plus - 2.0 * identity + minus) / (dx * dx)
    operator = -kappa * second + speed * first
    coordinates = torch.arange(num_points, dtype=dtype, device=device) * dx
    return coordinates, operator


def advection_diffusion_case(
    *,
    initial: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    times: torch.Tensor,
    alpha: torch.Tensor | float,
    diffusivity: torch.Tensor | float = 0.01,
    velocity: torch.Tensor | float = 1.0,
    num_points: int = 64,
    length: float = 1.0,
    arnoldi_dimension: int = 32,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ApplicationCase:
    """Build a controlled periodic fractional advection--diffusion case."""

    coordinates, operator = periodic_advection_diffusion_operator_1d(
        num_points,
        length=length,
        diffusivity=diffusivity,
        velocity=velocity,
        dtype=dtype,
        device=device,
    )
    u0 = _initial_field(coordinates, initial)
    problem = GeneralOperatorProblem(
        operator=operator,
        u0=u0,
        times=times,
        alpha=alpha,
        metadata={"application": "fractional-advection-diffusion", "boundary": "periodic"},
    )
    return ApplicationCase(
        name="fractional-advection-diffusion",
        domain="non-self-adjoint fractional transport",
        problem=problem,
        recommended_algorithm=MLSLArnoldi(arnoldi_dimension=min(arnoldi_dimension, num_points)),
        assumptions=("periodic centered discretization", "moderate reduced argument radius"),
        limitations=("validated reduced radius <= 4", "linear constant-coefficient transport"),
        differentiable_parameters=("alpha", "diffusivity", "velocity"),
        coordinates=coordinates,
    )
