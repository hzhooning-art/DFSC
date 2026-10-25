"""Public opt-in route; reject unsupported problem contracts before computation."""
from dataclasses import dataclass
import math
from time import perf_counter

from .admission_scalar import NumericalRefusal, evaluate as evaluate_with_admission
from .admission_spectral import GuardedSpectral, SpectralRefusal
from .problems import OperatorSpectralProblem, Solution
from .reliability import ReliabilityReport


@dataclass(frozen=True)
class MLSLGuarded:
    scalar_rtol: float = 1e-9
    scalar_atol: float = 1e-12
    gradient_atol: float = 1e-10
    action_rtol: float = 1e-8
    action_atol: float = 1e-10

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in (
            self.scalar_rtol, self.scalar_atol, self.gradient_atol, self.action_rtol, self.action_atol)):
            raise ValueError('Require finite nonnegative tolerances')

    @property
    def name(self):
        return 'mlsl-guarded-spd'


def guarded_solve(problem, algorithm):
    if not isinstance(problem, OperatorSpectralProblem) or problem.num_modes is not None:
        raise TypeError('Guarded route accepts only full-spectrum OperatorSpectralProblem, without num_modes.')
    started = perf_counter()
    layer = GuardedSpectral(problem.operator, scalar_rtol=algorithm.scalar_rtol,
        scalar_atol=algorithm.scalar_atol, gradient_atol=algorithm.gradient_atol,
        action_rtol=algorithm.action_rtol, action_atol=algorithm.action_atol)
    result = layer(problem.u0, problem.times, problem.alpha,
        beta=2. if problem.beta is None else problem.beta, diagnostics=True)
    scalar = result.scalar_diagnostics
    diagnostics = {
        'acceptance_passed': True, 'acceptance_scope': 'local scalar value/first-gradient and reconstructed-value proxies',
        'all_modal_entries_admitted': bool(scalar.accepted.all()), 'fallback_used': False,
        'actual_scalar_terms_computed': scalar.actual_terms_computed,
        'selected_terms_min': int(scalar.selected_terms.min()), 'selected_terms_max': int(scalar.selected_terms.max()),
        'maximum_scalar_value_proxy': float(scalar.estimated_errors[..., 0].max()),
        'maximum_reconstructed_value_proxy': float(result.propagated_value_proxy.max()),
        'rigorous_floating_point_certificate': False, 'operator_or_time_gradients_supported': False,
    }
    message = 'Local heuristic admission is not rigorous floating-point, basis or full-solver certification.'
    reliability = ReliabilityReport(level='experimental-admission', within_validated_domain=False,
        finite=True, converged=True, gradient_reliability='scalar-proxies; composed-gradients-finitely-tested',
        error_estimate_kind='exact-arithmetic-tail-plus-heuristic-roundoff-proxy', rigorous_error_bound=False,
        messages=(message,), metadata={'acceptance_passed': True, 'global_validation_claim': False})
    return Solution(values=result.values, times=problem.times, algorithm=algorithm.name,
        problem_type=type(problem).__name__, metadata=dict(problem.metadata), retcode='success',
        stats={'elapsed_seconds': perf_counter()-started, 'basis_setup_included': True,
               'device': str(problem.u0.device), 'dtype': str(problem.u0.dtype)},
        diagnostics=diagnostics, warnings=(message,), reliability=reliability)
