"""Problem--algorithm solve interface for dfsc."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from .forced_layer import ForcedMittagLefflerSpectralLayer
from .spectral_layer import MittagLefflerSpectralLayer

from .algorithms import AutoDFSC, CaputoL1, CaputoL1HistoryDirect, CaputoL1HistoryFFT, MLSLAdaptive, MLSLArnoldi, MLSLDirect, MLSLForced, MLSLGeneralizedOperator, MLSLGraph, MLSLKrylov, MLSLOperator, MLSLPicard, MLSLStable
from .arnoldi import arnoldi_mittag_leffler_action
from .fast_history import caputo_l1_derivative_direct, caputo_l1_derivative_fft
from .history import caputo_l1_linear_solve
from .krylov import adaptive_lanczos_mittag_leffler_action, lanczos_mittag_leffler_action
from .operators import build_generalized_operator_mlsl, build_graph_mlsl, build_operator_mlsl
from .problems import CaputoHistoryProblem, CaputoL1Problem, ForcedSpectralProblem, FractionalSpectralProblem, GeneralizedOperatorSpectralProblem, GeneralOperatorProblem, GraphSpectralProblem, LinearOperatorSpectralProblem, OperatorSpectralProblem, SemilinearSpectralProblem, Solution
from .reliability import assess_solution_reliability
from .semilinear import semilinear_mild_picard
from .selection import AlgorithmDecision, choose_algorithm


def _clone_layer(layer: MittagLefflerSpectralLayer, algorithm: MLSLDirect | MLSLStable) -> MittagLefflerSpectralLayer:
    config = algorithm.config
    return MittagLefflerSpectralLayer(
        layer.eigenvalues,
        layer.eigenvectors,
        projection_vectors=layer.projection_vectors,
        terms=config.terms,
        wave_speed=config.wave_speed,
        beta=config.beta,
        custom_backward=config.custom_backward,
        ml_method=config.ml_method,
    )


def _decision(problem: object, algorithm: object | str | None) -> AlgorithmDecision:
    if algorithm is None or algorithm == "auto" or isinstance(algorithm, AutoDFSC):
        return choose_algorithm(problem, algorithm if isinstance(algorithm, AutoDFSC) else None)
    aliases = {
        "direct": MLSLDirect,
        "caputo-l1": CaputoL1,
        "history-direct": CaputoL1HistoryDirect,
        "history-fft": CaputoL1HistoryFFT,
        "stable": MLSLStable,
        "forced": MLSLForced,
        "operator": MLSLOperator,
        "krylov": MLSLKrylov,
        "adaptive": MLSLAdaptive,
        "arnoldi": MLSLArnoldi,
        "generalized-operator": MLSLGeneralizedOperator,
        "graph": MLSLGraph,
        "picard": MLSLPicard,
    }
    if isinstance(algorithm, str):
        if algorithm not in aliases:
            raise ValueError(f"unknown dfsc algorithm alias: {algorithm}")
        algorithm = aliases[algorithm]()
    return AlgorithmDecision(algorithm, "algorithm selected explicitly")


def _solution(
    *,
    values: torch.Tensor,
    times: torch.Tensor,
    decision: AlgorithmDecision,
    problem_type: str,
    metadata: dict[str, Any],
    stats: dict[str, Any],
    started: float,
    retcode_override: str | None = None,
    extra_diagnostics: dict[str, Any] | None = None,
    extra_warnings: tuple[str, ...] = (),
) -> Solution:
    finite = bool(torch.isfinite(values.detach()).all().item())
    retcode = ("success" if finite else "nonfinite") if retcode_override is None else retcode_override
    if not finite:
        retcode = "nonfinite"
    warnings = [*decision.warnings, *extra_warnings]
    if not finite:
        warnings.append("solver output contains NaN or Inf values")
    diagnostics = decision.to_dict()
    diagnostics.update({"finite": finite, "output_shape": list(values.shape)})
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    warning_tuple = tuple(warnings)
    reliability = assess_solution_reliability(
        finite=finite,
        retcode=retcode,
        diagnostics=diagnostics,
        warnings=warning_tuple,
    )
    return Solution(
        values=values,
        times=times,
        algorithm=decision.name,
        problem_type=problem_type,
        retcode=retcode,
        metadata=metadata,
        stats={"elapsed_seconds": perf_counter() - started, "device": str(values.device), "dtype": str(values.dtype), **stats},
        diagnostics=diagnostics,
        warnings=warning_tuple,
        reliability=reliability,
    )


def _krylov_values_and_diagnostics(
    operator: object,
    u0: torch.Tensor,
    times: torch.Tensor,
    alpha: torch.Tensor | float,
    beta: torch.Tensor | float,
    algorithm: MLSLKrylov,
) -> tuple[torch.Tensor, dict[str, Any]]:
    values, diagnostics = lanczos_mittag_leffler_action(
        operator,
        u0,
        times,
        alpha,
        beta=beta,
        krylov_dimension=algorithm.krylov_dimension,
        breakdown_tol=algorithm.breakdown_tol,
        config=algorithm.config,
    )
    extra = diagnostics.to_dict()
    if algorithm.estimate_error and algorithm.krylov_dimension > 2:
        lower_dimension = max(2, algorithm.krylov_dimension - max(1, algorithm.error_dimension_step))
        with torch.no_grad():
            lower_values, _ = lanczos_mittag_leffler_action(
                operator,
                u0,
                times,
                alpha,
                beta=beta,
                krylov_dimension=lower_dimension,
                breakdown_tol=algorithm.breakdown_tol,
                config=algorithm.config,
            )
            denominator = torch.linalg.vector_norm(values.detach()).clamp_min(torch.finfo(values.dtype).eps)
            disagreement = torch.linalg.vector_norm(values.detach() - lower_values) / denominator
        extra.update(
            {
                "embedded_lower_krylov_dimension": lower_dimension,
                "embedded_relative_disagreement": float(disagreement.cpu()),
                "embedded_disagreement_is_error_bound": False,
            }
        )
    return values, extra


def _arnoldi_values_and_diagnostics(
    problem: GeneralOperatorProblem,
    algorithm: MLSLArnoldi,
) -> tuple[torch.Tensor, dict[str, Any]]:
    values, diagnostics = arnoldi_mittag_leffler_action(
        problem.operator,
        problem.u0,
        problem.times,
        problem.alpha,
        arnoldi_dimension=algorithm.arnoldi_dimension,
        terms=algorithm.terms,
        max_reduced_radius=algorithm.max_reduced_radius,
        breakdown_tol=algorithm.breakdown_tol,
        allow_unvalidated=algorithm.allow_unvalidated,
    )
    extra = diagnostics.to_dict()
    if algorithm.estimate_error and algorithm.arnoldi_dimension > 2:
        lower_dimension = max(2, algorithm.arnoldi_dimension - max(1, algorithm.error_dimension_step))
        with torch.no_grad():
            lower_values, _ = arnoldi_mittag_leffler_action(
                problem.operator,
                problem.u0,
                problem.times,
                problem.alpha,
                arnoldi_dimension=lower_dimension,
                terms=algorithm.terms,
                max_reduced_radius=algorithm.max_reduced_radius,
                breakdown_tol=algorithm.breakdown_tol,
                allow_unvalidated=algorithm.allow_unvalidated,
            )
            denominator = torch.linalg.vector_norm(values.detach()).clamp_min(
                torch.finfo(values.real.dtype).eps
            )
            disagreement = torch.linalg.vector_norm(values.detach() - lower_values) / denominator
        extra.update(
            {
                "embedded_lower_arnoldi_dimension": lower_dimension,
                "embedded_relative_disagreement": float(disagreement.cpu()),
                "embedded_disagreement_is_error_bound": False,
            }
        )
    return values, extra


def solve(problem: object, algorithm: object | str | None = None, **kwargs: Any) -> Solution:
    """Solve a supported dfsc problem with inspectable algorithm selection."""

    started = perf_counter()
    decision = _decision(problem, algorithm)
    alg = decision.algorithm

    if isinstance(problem, FractionalSpectralProblem):
        if not isinstance(alg, (MLSLDirect, MLSLStable)):
            raise TypeError("FractionalSpectralProblem requires MLSLDirect, MLSLStable, AutoDFSC, or algorithm=None")
        if not isinstance(problem.layer, MittagLefflerSpectralLayer):
            raise TypeError("algorithm-controlled solves require a MittagLefflerSpectralLayer")
        layer = _clone_layer(problem.layer, alg)
        values = layer(problem.u0, problem.times, problem.alpha, beta=problem.beta)
        diagnostics = dict(decision.diagnostics)
        diagnostics.update({"evaluator_method": layer.ml_method, "terms": layer.terms})
        decision = AlgorithmDecision(alg, decision.reason, diagnostics, decision.warnings)
        return _solution(
            values=values,
            times=problem.times,
            decision=decision,
            problem_type="FractionalSpectralProblem",
            metadata=dict(problem.metadata),
            stats={"direct_query": True, **kwargs},
            started=started,
        )

    if isinstance(problem, OperatorSpectralProblem):
        if isinstance(alg, MLSLAdaptive):
            beta = alg.config.beta if problem.beta is None else problem.beta
            values, adaptive_diagnostics = adaptive_lanczos_mittag_leffler_action(
                problem.operator,
                problem.u0,
                problem.times,
                problem.alpha,
                beta=beta,
                dimension_schedule=alg.dimension_schedule,
                rtol=alg.rtol,
                atol=alg.atol,
                breakdown_tol=alg.breakdown_tol,
                config=alg.config,
                strict=alg.strict,
            )
            converged = adaptive_diagnostics.converged
            return _solution(
                values=values,
                times=problem.times,
                decision=decision,
                problem_type="OperatorSpectralProblem",
                metadata={"num_modes": problem.num_modes, **problem.metadata},
                stats={"direct_query": True, "constructed_layer": False, "matrix_function_action": True, "adaptive": True, **kwargs},
                started=started,
                retcode_override="success" if converged else "maxiters",
                extra_diagnostics=adaptive_diagnostics.to_dict(),
                extra_warnings=() if converged else ("adaptive Krylov schedule exhausted before tolerance was met",),
            )
        if isinstance(alg, MLSLKrylov):
            beta = alg.config.beta if problem.beta is None else problem.beta
            values, extra_diagnostics = _krylov_values_and_diagnostics(
                problem.operator,
                problem.u0,
                problem.times,
                problem.alpha,
                beta,
                alg,
            )
            return _solution(
                values=values,
                times=problem.times,
                decision=decision,
                problem_type="OperatorSpectralProblem",
                metadata={"num_modes": problem.num_modes, **problem.metadata},
                stats={"direct_query": True, "constructed_layer": False, "matrix_function_action": True, **kwargs},
                started=started,
                extra_diagnostics=extra_diagnostics,
            )
        if not isinstance(alg, MLSLOperator):
            raise TypeError("OperatorSpectralProblem requires MLSLOperator, MLSLKrylov, AutoDFSC, or algorithm=None")
        layer = build_operator_mlsl(problem.operator, num_modes=problem.num_modes, config=alg.config)
        values = layer(problem.u0, problem.times, problem.alpha, beta=problem.beta)
        return _solution(
            values=values,
            times=problem.times,
            decision=decision,
            problem_type="OperatorSpectralProblem",
            metadata={"num_modes": problem.num_modes, **problem.metadata},
            stats={"direct_query": True, "constructed_layer": True, "evaluator_method": layer.ml_method, **kwargs},
            started=started,
        )

    if isinstance(problem, LinearOperatorSpectralProblem):
        if not isinstance(alg, MLSLKrylov):
            raise TypeError("LinearOperatorSpectralProblem requires MLSLKrylov, AutoDFSC, or algorithm=None")
        beta = alg.config.beta if problem.beta is None else problem.beta
        values, extra_diagnostics = _krylov_values_and_diagnostics(
            problem.operator,
            problem.u0,
            problem.times,
            problem.alpha,
            beta,
            alg,
        )
        return _solution(
            values=values,
            times=problem.times,
            decision=decision,
            problem_type="LinearOperatorSpectralProblem",
            metadata={"operator_name": problem.operator.name, **problem.metadata},
            stats={"direct_query": True, "matrix_function_action": True, "matrix_free": True, **kwargs},
            started=started,
            extra_diagnostics=extra_diagnostics,
        )

    if isinstance(problem, GeneralOperatorProblem):
        if not isinstance(alg, MLSLArnoldi):
            raise TypeError("GeneralOperatorProblem requires MLSLArnoldi, AutoDFSC, or algorithm=None")
        values, extra_diagnostics = _arnoldi_values_and_diagnostics(problem, alg)
        return _solution(
            values=values,
            times=problem.times,
            decision=decision,
            problem_type="GeneralOperatorProblem",
            metadata=dict(problem.metadata),
            stats={"direct_query": True, "matrix_function_action": True, "general_operator": True, **kwargs},
            started=started,
            extra_diagnostics=extra_diagnostics,
        )

    if isinstance(problem, GraphSpectralProblem):
        if not isinstance(alg, MLSLGraph):
            raise TypeError("GraphSpectralProblem requires MLSLGraph, AutoDFSC, or algorithm=None")
        layer = build_graph_mlsl(problem.adjacency, num_modes=problem.num_modes, normalized=problem.normalized, config=alg.config)
        values = layer(problem.u0, problem.times, problem.alpha, beta=problem.beta)
        return _solution(
            values=values,
            times=problem.times,
            decision=decision,
            problem_type="GraphSpectralProblem",
            metadata={"num_modes": problem.num_modes, "normalized": problem.normalized, **problem.metadata},
            stats={"direct_query": True, "constructed_layer": True, "evaluator_method": layer.ml_method, **kwargs},
            started=started,
        )

    if isinstance(problem, GeneralizedOperatorSpectralProblem):
        if not isinstance(alg, MLSLGeneralizedOperator):
            raise TypeError(
                "GeneralizedOperatorSpectralProblem requires MLSLGeneralizedOperator, AutoDFSC, or algorithm=None"
            )
        layer = build_generalized_operator_mlsl(
            problem.stiffness,
            problem.mass,
            num_modes=problem.num_modes,
            config=alg.config,
        )
        values = layer(problem.u0, problem.times, problem.alpha, beta=problem.beta)
        return _solution(
            values=values,
            times=problem.times,
            decision=decision,
            problem_type="GeneralizedOperatorSpectralProblem",
            metadata={"num_modes": problem.num_modes, **problem.metadata},
            stats={"direct_query": True, "constructed_layer": True, "mass_projection": True, "evaluator_method": layer.ml_method, **kwargs},
            started=started,
        )

    if isinstance(problem, ForcedSpectralProblem):
        if isinstance(alg, (MLSLDirect, MLSLStable)):
            alg = MLSLForced(config=alg.config)
            decision = AlgorithmDecision(alg, "explicit spectral algorithm adapted to the forced wrapper")
        if not isinstance(alg, MLSLForced):
            raise TypeError("ForcedSpectralProblem requires MLSLForced, AutoDFSC, or algorithm=None")
        if not isinstance(problem.layer, MittagLefflerSpectralLayer):
            raise TypeError("forced solves require a MittagLefflerSpectralLayer backbone")
        base_algorithm = MLSLStable(terms=alg.config.terms) if alg.config.ml_method == "hybrid" else MLSLDirect(alg.config)
        base_layer = _clone_layer(problem.layer, base_algorithm)
        forced_layer = ForcedMittagLefflerSpectralLayer(
            base_layer,
            forcing_terms=alg.forcing_terms,
            ml_method=alg.config.ml_method,
        )
        values = forced_layer(problem.u0, problem.times, problem.alpha, problem.forcing_values, problem.forcing_times, beta=problem.beta)
        return _solution(
            values=values,
            times=problem.times,
            decision=decision,
            problem_type="ForcedSpectralProblem",
            metadata=dict(problem.metadata),
            stats={"direct_query": True, "forcing_terms": alg.forcing_terms, "evaluator_method": alg.config.ml_method, **kwargs},
            started=started,
        )

    if isinstance(problem, SemilinearSpectralProblem):
        if not isinstance(alg, MLSLPicard):
            raise TypeError("SemilinearSpectralProblem requires MLSLPicard, AutoDFSC, or algorithm=None")
        if not isinstance(problem.layer, MittagLefflerSpectralLayer):
            raise TypeError("semilinear solves require a MittagLefflerSpectralLayer backbone")
        values, picard_diagnostics = semilinear_mild_picard(
            problem.layer,
            problem.u0,
            problem.times,
            problem.alpha,
            problem.nonlinearity,
            beta=problem.beta,
            max_iterations=alg.max_iterations,
            tolerance=alg.tolerance,
            relaxation=alg.relaxation,
            quadrature_points=alg.quadrature_points,
            forcing_terms=alg.forcing_terms,
            config=alg.config,
        )
        converged = picard_diagnostics.converged
        return _solution(
            values=values,
            times=problem.times,
            decision=decision,
            problem_type="SemilinearSpectralProblem",
            metadata=dict(problem.metadata),
            stats={"direct_query": False, "nonlinear_iteration": True, **kwargs},
            started=started,
            retcode_override="success" if converged else "maxiters",
            extra_diagnostics=picard_diagnostics.to_dict(),
            extra_warnings=() if converged else ("Picard iteration reached max_iterations before tolerance",),
        )

    if isinstance(problem, CaputoL1Problem):
        if not isinstance(alg, CaputoL1):
            raise TypeError("CaputoL1Problem requires CaputoL1, AutoDFSC, or algorithm=None")
        values, times = caputo_l1_linear_solve(
            problem.operator,
            problem.u0,
            alpha=problem.alpha,
            final_time=problem.final_time,
            num_steps=problem.num_steps,
            forcing=problem.forcing,
        )
        return _solution(
            values=values,
            times=times,
            decision=decision,
            problem_type="CaputoL1Problem",
            metadata=dict(problem.metadata),
            stats={"direct_query": False, "history_dependent": True, "num_steps": problem.num_steps, **kwargs},
            started=started,
        )

    if isinstance(problem, CaputoHistoryProblem):
        if not isinstance(alg, (CaputoL1HistoryDirect, CaputoL1HistoryFFT)):
            raise TypeError(
                "CaputoHistoryProblem requires CaputoL1HistoryDirect, CaputoL1HistoryFFT, AutoDFSC, or algorithm=None"
            )
        evaluator = caputo_l1_derivative_direct if isinstance(alg, CaputoL1HistoryDirect) else caputo_l1_derivative_fft
        values, history_diagnostics = evaluator(
            problem.values,
            alpha=problem.alpha,
            final_time=problem.final_time,
        )
        num_steps = history_diagnostics.num_steps
        times = torch.linspace(
            problem.final_time / num_steps,
            problem.final_time,
            num_steps,
            dtype=problem.values.dtype,
            device=problem.values.device,
        )
        return _solution(
            values=values,
            times=times,
            decision=decision,
            problem_type="CaputoHistoryProblem",
            metadata=dict(problem.metadata),
            stats={
                "direct_query": False,
                "history_dependent": True,
                "trajectory_operator": True,
                "online_time_stepper": False,
                **kwargs,
            },
            started=started,
            extra_diagnostics=history_diagnostics.to_dict(),
        )

    raise TypeError(f"unsupported dfsc problem type: {type(problem).__name__}")
