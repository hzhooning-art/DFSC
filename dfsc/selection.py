"""Algorithm selection policies for the dfsc problem interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .factory import MLSLConfig
from .spectral_layer import MittagLefflerSpectralLayer

from .algorithms import AutoDFSC, CaputoL1, CaputoL1HistoryDirect, CaputoL1HistoryFFT, MLSLArnoldi, MLSLDirect, MLSLForced, MLSLGeneralizedOperator, MLSLGraph, MLSLKrylov, MLSLOperator, MLSLPicard, MLSLStable
from .problems import CaputoHistoryProblem, CaputoL1Problem, ForcedSpectralProblem, FractionalSpectralProblem, GeneralizedOperatorSpectralProblem, GeneralOperatorProblem, GraphSpectralProblem, LinearOperatorSpectralProblem, OperatorSpectralProblem, SemilinearSpectralProblem


@dataclass(frozen=True)
class AlgorithmDecision:
    """Selected algorithm and the detached diagnostics behind the decision."""

    algorithm: object
    reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return getattr(self.algorithm, "name", self.algorithm.__class__.__name__)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.name,
            "reason": self.reason,
            "warnings": list(self.warnings),
            **self.diagnostics,
        }


def _max_spectral_argument(problem: FractionalSpectralProblem) -> tuple[float, float]:
    layer = problem.layer
    if not isinstance(layer, MittagLefflerSpectralLayer):
        raise TypeError("automatic selection requires a MittagLefflerSpectralLayer")
    eigenvalues = layer.eigenvalues.detach()
    beta = layer.beta if problem.beta is None else problem.beta
    beta_t = torch.as_tensor(beta, dtype=eigenvalues.dtype, device=eigenvalues.device).detach()
    alpha_t = torch.as_tensor(
        problem.alpha,
        dtype=problem.times.dtype,
        device=problem.times.device,
    ).detach()
    times = problem.times.detach().clamp_min(0.0)
    rates = (layer.wave_speed**2) * eigenvalues.clamp_min(0.0).pow(beta_t / 2.0)
    time_factor = times.clamp_min(torch.finfo(times.dtype).tiny).pow(alpha_t)
    time_factor = torch.where(times > 0, time_factor, torch.zeros_like(time_factor))
    radius = float((rates.max() * time_factor.max()).cpu()) if rates.numel() and times.numel() else 0.0
    alpha_min = float(torch.min(alpha_t).cpu())
    return radius, alpha_min


def choose_algorithm(problem: object, policy: AutoDFSC | None = None) -> AlgorithmDecision:
    """Choose an implemented algorithm without mutating the autograd graph."""

    policy = AutoDFSC() if policy is None else policy
    if isinstance(problem, FractionalSpectralProblem):
        radius, alpha_min = _max_spectral_argument(problem)
        dtype = problem.u0.dtype
        radius_limit = policy.direct_radius
        if alpha_min < 1.0:
            radius_limit = min(radius_limit, 8.0)
        if alpha_min < 0.60:
            radius_limit = min(radius_limit, 4.0)
        if dtype in (torch.float16, torch.bfloat16, torch.float32):
            radius_limit = min(radius_limit, 1.0)
        diagnostics = {
            "max_abs_argument": radius,
            "direct_radius_limit": radius_limit,
            "alpha_min": alpha_min,
            "dtype": str(dtype),
        }
        if radius <= radius_limit:
            return AlgorithmDecision(
                MLSLDirect(config=MLSLConfig(terms=policy.direct_terms, custom_backward=True)),
                "estimated spectral arguments remain inside the direct-series region",
                diagnostics,
            )
        return AlgorithmDecision(
            MLSLStable(terms=policy.stable_terms),
            "estimated spectral arguments require the stable hybrid evaluator",
            diagnostics,
        )
    if isinstance(problem, OperatorSpectralProblem):
        operator_size = int(problem.operator.shape[0])
        if problem.num_modes is None and operator_size > policy.dense_eigh_limit:
            return AlgorithmDecision(
                MLSLKrylov(
                    config=MLSLConfig.stable(terms=policy.stable_terms),
                    krylov_dimension=min(policy.krylov_dimension, operator_size),
                ),
                "operator size exceeds the configured full-eigendecomposition limit",
                {
                    "operator_size": operator_size,
                    "dense_eigh_limit": policy.dense_eigh_limit,
                    "krylov_dimension": min(policy.krylov_dimension, operator_size),
                },
            )
        return AlgorithmDecision(
            MLSLOperator(config=MLSLConfig.stable(terms=policy.stable_terms), num_modes=problem.num_modes),
            "operator problems use the stable symmetric-PSD spectral adapter",
            {"num_modes": problem.num_modes, "operator_size": operator_size},
        )
    if isinstance(problem, LinearOperatorSpectralProblem):
        return AlgorithmDecision(
            MLSLKrylov(
                config=MLSLConfig.stable(terms=policy.stable_terms),
                krylov_dimension=min(policy.krylov_dimension, problem.operator.size),
            ),
            "matrix-free operator problems require a Krylov matrix-function action",
            {
                "operator_size": problem.operator.size,
                "operator_representation": problem.operator.representation,
                "krylov_dimension": min(policy.krylov_dimension, problem.operator.size),
            },
        )
    if isinstance(problem, GeneralOperatorProblem):
        operator_size = problem.operator.shape[0] if torch.is_tensor(problem.operator) else problem.operator.size
        return AlgorithmDecision(
            MLSLArnoldi(arnoldi_dimension=min(policy.krylov_dimension, operator_size)),
            "general or complex operators require the controlled Arnoldi action",
            {
                "operator_size": operator_size,
                "arnoldi_dimension": min(policy.krylov_dimension, operator_size),
            },
        )
    if isinstance(problem, GeneralizedOperatorSpectralProblem):
        return AlgorithmDecision(
            MLSLGeneralizedOperator(
                config=MLSLConfig.stable(terms=policy.stable_terms),
                num_modes=problem.num_modes,
            ),
            "generalized operator problems use the stable stiffness/mass spectral adapter",
            {"num_modes": problem.num_modes},
        )
    if isinstance(problem, GraphSpectralProblem):
        return AlgorithmDecision(
            MLSLGraph(
                config=MLSLConfig.stable(terms=policy.stable_terms),
                num_modes=problem.num_modes,
                normalized=problem.normalized,
            ),
            "graph problems use the stable graph-Laplacian spectral adapter",
            {"num_modes": problem.num_modes, "normalized": problem.normalized},
        )
    if isinstance(problem, ForcedSpectralProblem):
        return AlgorithmDecision(
            MLSLForced(config=MLSLConfig.stable(terms=policy.stable_terms)),
            "forced problems require the two-parameter hybrid evaluator and quadrature wrapper",
        )
    if isinstance(problem, SemilinearSpectralProblem):
        return AlgorithmDecision(
            MLSLPicard(config=MLSLConfig.stable(terms=policy.stable_terms)),
            "semilinear problems use a mild-form Picard iteration on the supplied MLSL backbone",
        )
    if isinstance(problem, CaputoL1Problem):
        return AlgorithmDecision(
            CaputoL1(),
            "the problem requests a history-aware constant-order Caputo time discretization",
            {"num_steps": problem.num_steps, "final_time": problem.final_time},
        )
    if isinstance(problem, CaputoHistoryProblem):
        time_dimension = 0 if problem.values.ndim == 1 else problem.values.ndim - 2
        num_steps = int(problem.values.shape[time_dimension] - 1)
        diagnostics = {
            "num_steps": num_steps,
            "history_fft_threshold": policy.history_fft_threshold,
        }
        if num_steps <= policy.history_fft_threshold:
            return AlgorithmDecision(
                CaputoL1HistoryDirect(),
                "trajectory length remains below the direct-history threshold",
                diagnostics,
            )
        return AlgorithmDecision(
            CaputoL1HistoryFFT(),
            "trajectory length exceeds the direct-history threshold",
            diagnostics,
        )
    raise TypeError(f"unsupported dfsc problem type: {type(problem).__name__}")
