"""Problem and solution objects for dfsc."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .linear_operators import GeneralLinearOperator, SelfAdjointLinearOperator
from .reliability import ReliabilityReport


@dataclass
class FractionalSpectralProblem:
    """Known-layer fractional spectral evolution problem."""

    layer: torch.nn.Module
    u0: torch.Tensor
    times: torch.Tensor
    alpha: torch.Tensor | float
    beta: torch.Tensor | float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorSpectralProblem:
    """Fractional spectral problem defined by a symmetric PSD operator."""

    operator: torch.Tensor
    u0: torch.Tensor
    times: torch.Tensor
    alpha: torch.Tensor | float
    beta: torch.Tensor | float | None = None
    num_modes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LinearOperatorSpectralProblem:
    """Fractional problem defined only through a self-adjoint matvec contract."""

    operator: SelfAdjointLinearOperator
    u0: torch.Tensor
    times: torch.Tensor
    alpha: torch.Tensor | float
    beta: torch.Tensor | float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneralOperatorProblem:
    """Fractional propagation problem for a non-self-adjoint or complex operator."""

    operator: torch.Tensor | GeneralLinearOperator
    u0: torch.Tensor
    times: torch.Tensor
    alpha: torch.Tensor | float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneralizedOperatorSpectralProblem:
    """Fractional spectral problem defined by an assembled ``K, M`` pair."""

    stiffness: torch.Tensor
    mass: torch.Tensor
    u0: torch.Tensor
    times: torch.Tensor
    alpha: torch.Tensor | float
    beta: torch.Tensor | float | None = None
    num_modes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphSpectralProblem:
    """Fractional spectral problem defined by an undirected graph adjacency."""

    adjacency: torch.Tensor
    u0: torch.Tensor
    times: torch.Tensor
    alpha: torch.Tensor | float
    beta: torch.Tensor | float | None = None
    num_modes: int | None = None
    normalized: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForcedSpectralProblem:
    """Forced fractional spectral evolution problem."""

    layer: torch.nn.Module
    u0: torch.Tensor
    times: torch.Tensor
    alpha: torch.Tensor | float
    forcing_values: torch.Tensor
    forcing_times: torch.Tensor
    beta: torch.Tensor | float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemilinearSpectralProblem:
    """Semilinear problem with an MLSL backbone and state nonlinearity."""

    layer: torch.nn.Module
    u0: torch.Tensor
    times: torch.Tensor
    alpha: torch.Tensor | float
    nonlinearity: Any
    beta: torch.Tensor | float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaputoL1Problem:
    """History-aware linear Caputo problem ``D_t^alpha u + A u = f``."""

    operator: torch.Tensor
    u0: torch.Tensor
    alpha: torch.Tensor | float
    final_time: float
    num_steps: int
    forcing: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaputoHistoryProblem:
    """Evaluate a Caputo-L1 derivative over a supplied uniform-grid trajectory."""

    values: torch.Tensor
    alpha: torch.Tensor | float
    final_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Solution:
    """Solution object returned by `dfsc.solve`."""

    values: torch.Tensor
    times: torch.Tensor
    algorithm: str
    problem_type: str
    retcode: str = "success"
    metadata: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    reliability: ReliabilityReport | None = None

    @property
    def u(self) -> torch.Tensor:
        return self.values

    @property
    def final(self) -> torch.Tensor:
        if self.times.ndim > 0 and self.values.ndim >= 1:
            time_dim = -1 if self.values.ndim == 1 else -2
            return self.values.select(dim=time_dim, index=self.values.shape[time_dim] - 1)
        return self.values

    @property
    def success(self) -> bool:
        return self.retcode == "success"

    @property
    def quality(self) -> str:
        return "unknown" if self.reliability is None else self.reliability.level

    def summary(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "problem_type": self.problem_type,
            "retcode": self.retcode,
            "success": self.success,
            "quality": self.quality,
            "shape": list(self.values.shape),
            "warnings": list(self.warnings),
            "stats": dict(self.stats),
            "reliability": None if self.reliability is None else self.reliability.to_dict(),
        }

    def relative_l2_error(self, target: torch.Tensor, *, eps: float = 1e-14) -> torch.Tensor:
        denom = torch.linalg.norm(target).clamp_min(eps)
        return torch.linalg.norm(self.values - target) / denom
