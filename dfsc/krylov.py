"""Lanczos matrix-function actions for dfsc fractional propagators."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .factory import MLSLConfig
from .mittag_leffler import mittag_leffler_e

from .linear_operators import SelfAdjointLinearOperator, as_self_adjoint_operator


@dataclass(frozen=True)
class KrylovDiagnostics:
    """Detached diagnostics from a batched Lanczos propagation."""

    requested_dimension: int
    effective_dimensions: tuple[int, ...]
    breakdowns: tuple[bool, ...]
    symmetric: bool
    representation: str
    positive_semidefinite_contract: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_krylov_dimension": self.requested_dimension,
            "effective_krylov_dimensions": list(self.effective_dimensions),
            "min_effective_krylov_dimension": min(self.effective_dimensions),
            "max_effective_krylov_dimension": max(self.effective_dimensions),
            "lanczos_breakdown_count": sum(self.breakdowns),
            "operator_symmetric": self.symmetric,
            "operator_representation": self.representation,
            "positive_semidefinite_contract": self.positive_semidefinite_contract,
        }


@dataclass(frozen=True)
class AdaptiveKrylovDiagnostics:
    """Work and convergence diagnostics for adaptive Lanczos propagation."""

    converged: bool
    selected_dimension: int
    attempted_dimensions: tuple[int, ...]
    relative_disagreements: tuple[float, ...]
    absolute_disagreements: tuple[float, ...]
    final: KrylovDiagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            **self.final.to_dict(),
            "adaptive": True,
            "adaptive_converged": self.converged,
            "selected_krylov_dimension": self.selected_dimension,
            "attempted_krylov_dimensions": list(self.attempted_dimensions),
            "successive_relative_disagreements": list(self.relative_disagreements),
            "successive_absolute_disagreements": list(self.absolute_disagreements),
            "adaptive_estimate_is_error_bound": False,
        }


@dataclass(frozen=True)
class PreparedLanczosBasis:
    """Reusable reduced spaces for repeated parameter/time queries.

    A prepared basis is tied to the operator and initial states used to create
    it.  It remains differentiable with respect to query-time ``alpha`` and
    ``beta``.  Rebuild it after changing an operator parameter or ``u0``.
    """

    bases: tuple[torch.Tensor, ...]
    tridiagonals: tuple[torch.Tensor, ...]
    norms: tuple[torch.Tensor, ...]
    batch_shape: tuple[int, ...]
    state_size: int
    diagnostics: KrylovDiagnostics
    config: MLSLConfig


def prepare_lanczos_basis(
    operator: torch.Tensor | SelfAdjointLinearOperator,
    u0: torch.Tensor,
    *,
    krylov_dimension: int = 48,
    breakdown_tol: float = 1e-12,
    config: MLSLConfig | None = None,
) -> PreparedLanczosBasis:
    """Build batched Lanczos spaces once for repeated propagator queries."""

    config = MLSLConfig.stable() if config is None else config
    if not u0.is_floating_point():
        raise TypeError("u0 must be a floating-point tensor")
    normalized = as_self_adjoint_operator(operator, dtype=u0.dtype, device=u0.device)
    if u0.shape[-1] != normalized.size:
        raise ValueError("u0 last dimension must match the operator size")
    dimension = min(int(krylov_dimension), normalized.size)
    if dimension < 1:
        raise ValueError("krylov_dimension must be positive")
    bases: list[torch.Tensor] = []
    tridiagonals: list[torch.Tensor] = []
    norms: list[torch.Tensor] = []
    dimensions: list[int] = []
    breakdowns: list[bool] = []
    for vector in u0.reshape(-1, u0.shape[-1]):
        norm = torch.linalg.vector_norm(vector)
        basis, tridiagonal, breakdown = _lanczos_basis(
            normalized,
            vector,
            dimension=dimension,
            breakdown_tol=float(breakdown_tol),
        )
        bases.append(basis)
        tridiagonals.append(tridiagonal)
        norms.append(norm)
        dimensions.append(basis.shape[1])
        breakdowns.append(breakdown)
    diagnostics = KrylovDiagnostics(
        dimension,
        tuple(dimensions),
        tuple(breakdowns),
        normalized.symmetric,
        normalized.representation,
        normalized.positive_semidefinite,
    )
    return PreparedLanczosBasis(
        tuple(bases),
        tuple(tridiagonals),
        tuple(norms),
        tuple(u0.shape[:-1]),
        u0.shape[-1],
        diagnostics,
        config,
    )


def apply_prepared_lanczos_basis(
    prepared: PreparedLanczosBasis,
    times: torch.Tensor | float,
    alpha: torch.Tensor | float,
    *,
    beta: torch.Tensor | float = 2.0,
) -> torch.Tensor:
    """Apply a prepared basis while preserving query-parameter gradients."""

    prototype = prepared.norms[0]
    times_t = torch.as_tensor(times, dtype=prototype.dtype, device=prototype.device)
    alpha_t = torch.as_tensor(alpha, dtype=prototype.dtype, device=prototype.device)
    beta_t = torch.as_tensor(beta, dtype=prototype.dtype, device=prototype.device)
    outputs: list[torch.Tensor] = []
    for basis, tridiagonal, norm in zip(
        prepared.bases, prepared.tridiagonals, prepared.norms, strict=True
    ):
        if tridiagonal.numel() == 0:
            outputs.append(prototype.new_zeros(tuple(times_t.shape) + (prepared.state_size,)))
            continue
        ritz_values, ritz_vectors = torch.linalg.eigh(tridiagonal)
        scale = float(torch.abs(tridiagonal.detach()).max().cpu())
        tolerance = 1000.0 * torch.finfo(ritz_values.dtype).eps * max(1.0, scale)
        if float(ritz_values.min().detach().cpu()) < -tolerance:
            raise ValueError("operator is not positive semidefinite within numerical tolerance")
        ritz_values = ritz_values.clamp_min(0.0)
        safe = torch.where(ritz_values > 0, ritz_values, torch.ones_like(ritz_values))
        rates = (prepared.config.wave_speed**2) * torch.where(
            ritz_values > 0, safe.pow(beta_t / 2.0), torch.zeros_like(ritz_values)
        )
        tiny = torch.finfo(times_t.dtype).tiny
        time_factor = torch.where(times_t > 0, times_t.clamp_min(tiny).pow(alpha_t), torch.zeros_like(times_t))
        z = -rates * time_factor.unsqueeze(-1) if times_t.ndim else -rates * time_factor
        kernel = mittag_leffler_e(
            alpha_t,
            z,
            terms=prepared.config.terms,
            custom_backward=prepared.config.custom_backward,
            method=prepared.config.ml_method,
        )
        reduced = (kernel * ritz_vectors[0, :]) @ ritz_vectors.transpose(-1, -2)
        outputs.append(norm * (reduced @ basis.transpose(-1, -2)))
    stacked = torch.stack(outputs)
    output_shape = prepared.batch_shape + tuple(times_t.shape) + (prepared.state_size,)
    return stacked.reshape(output_shape)


def _lanczos_basis(
    operator: SelfAdjointLinearOperator,
    vector: torch.Tensor,
    *,
    dimension: int,
    breakdown_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Build a fully reorthogonalized Lanczos basis and tridiagonal matrix."""

    norm = torch.linalg.vector_norm(vector)
    if float(norm.detach().cpu()) == 0.0:
        return vector.new_zeros((vector.numel(), 0)), vector.new_zeros((0, 0)), True

    basis: list[torch.Tensor] = []
    diagonal: list[torch.Tensor] = []
    off_diagonal: list[torch.Tensor] = []
    q = vector / norm
    q_previous = torch.zeros_like(q)
    beta_previous = vector.new_zeros(())
    breakdown = False

    for index in range(dimension):
        basis.append(q)
        z = operator(q)
        alpha_j = torch.dot(q, z)
        diagonal.append(alpha_j)
        z = z - alpha_j * q
        if index > 0:
            z = z - beta_previous * q_previous

        # Full reorthogonalization is more reliable than the three-term
        # recurrence alone for the small and medium Krylov spaces used here.
        for basis_vector in basis:
            z = z - torch.dot(basis_vector, z) * basis_vector

        if index == dimension - 1:
            break
        beta_j = torch.linalg.vector_norm(z)
        scale = max(1.0, float(torch.abs(alpha_j.detach()).cpu()))
        if float(beta_j.detach().cpu()) <= breakdown_tol * scale:
            breakdown = True
            break
        off_diagonal.append(beta_j)
        q_previous, q = q, z / beta_j
        beta_previous = beta_j

    q_matrix = torch.stack(basis, dim=1)
    tridiagonal = torch.diag(torch.stack(diagonal))
    if off_diagonal:
        off = torch.stack(off_diagonal)
        tridiagonal = tridiagonal + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    return q_matrix, tridiagonal, breakdown


def _single_krylov_action(
    operator: SelfAdjointLinearOperator,
    vector: torch.Tensor,
    times: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    dimension: int,
    breakdown_tol: float,
    config: MLSLConfig,
) -> tuple[torch.Tensor, int, bool]:
    norm = torch.linalg.vector_norm(vector)
    if float(norm.detach().cpu()) == 0.0:
        shape = tuple(times.shape) + (vector.numel(),) if times.ndim else (vector.numel(),)
        return vector.new_zeros(shape), 0, True

    basis, tridiagonal, breakdown = _lanczos_basis(
        operator,
        vector,
        dimension=dimension,
        breakdown_tol=breakdown_tol,
    )
    ritz_values, ritz_vectors = torch.linalg.eigh(tridiagonal)
    reduced_scale = float(torch.abs(tridiagonal.detach()).max().cpu()) if tridiagonal.numel() else 1.0
    negative_tolerance = 1000.0 * torch.finfo(ritz_values.dtype).eps * max(1.0, reduced_scale)
    if float(ritz_values.min().detach().cpu()) < -negative_tolerance:
        raise ValueError("operator is not positive semidefinite within numerical tolerance")
    ritz_values = ritz_values.clamp_min(0.0)
    safe_ritz_values = torch.where(ritz_values > 0, ritz_values, torch.ones_like(ritz_values))
    positive_rates = safe_ritz_values.pow(beta / 2.0)
    rates = (config.wave_speed**2) * torch.where(
        ritz_values > 0,
        positive_rates,
        torch.zeros_like(positive_rates),
    )
    tiny = torch.finfo(times.dtype).tiny
    time_factor = torch.where(times > 0, times.clamp_min(tiny).pow(alpha), torch.zeros_like(times))
    z = -rates * time_factor.unsqueeze(-1) if times.ndim else -rates * time_factor
    kernel = mittag_leffler_e(
        alpha,
        z,
        terms=config.terms,
        custom_backward=config.custom_backward,
        method=config.ml_method,
    )
    first_row = ritz_vectors[0, :]
    reduced = (kernel * first_row) @ ritz_vectors.transpose(-1, -2)
    values = norm * (reduced @ basis.transpose(-1, -2))
    return values, basis.shape[1], breakdown


def lanczos_mittag_leffler_action(
    operator: torch.Tensor | SelfAdjointLinearOperator,
    u0: torch.Tensor,
    times: torch.Tensor | float,
    alpha: torch.Tensor | float,
    *,
    beta: torch.Tensor | float = 2.0,
    krylov_dimension: int = 48,
    breakdown_tol: float = 1e-12,
    config: MLSLConfig | None = None,
) -> tuple[torch.Tensor, KrylovDiagnostics]:
    """Approximate a Mittag-Leffler matrix-function action with Lanczos.

    The routine avoids a full eigendecomposition of a dense, sparse, or
    matrix-free symmetric positive-semidefinite operator. It supports leading batch
    dimensions on ``u0``; each initial state receives its own Krylov basis.
    Gradients are preserved through the reduced matrix function and with
    respect to ``u0``, ``alpha``, and ``beta`` inside a fixed Lanczos path.
    """

    config = MLSLConfig.stable() if config is None else config
    if not u0.is_floating_point():
        raise TypeError("u0 must be a floating-point tensor")
    normalized_operator = as_self_adjoint_operator(operator, dtype=u0.dtype, device=u0.device)
    if u0.shape[-1] != normalized_operator.size:
        raise ValueError("u0 last dimension must match the operator size")
    dimension = min(int(krylov_dimension), normalized_operator.size)
    if dimension < 1:
        raise ValueError("krylov_dimension must be positive")

    times_t = torch.as_tensor(times, dtype=u0.dtype, device=u0.device)
    alpha_t = torch.as_tensor(alpha, dtype=u0.dtype, device=u0.device)
    beta_t = torch.as_tensor(beta, dtype=u0.dtype, device=u0.device)
    flat = u0.reshape(-1, u0.shape[-1])
    outputs: list[torch.Tensor] = []
    dimensions: list[int] = []
    breakdowns: list[bool] = []
    for vector in flat:
        values, effective_dimension, breakdown = _single_krylov_action(
            normalized_operator,
            vector,
            times_t,
            alpha_t,
            beta_t,
            dimension=dimension,
            breakdown_tol=float(breakdown_tol),
            config=config,
        )
        outputs.append(values)
        dimensions.append(effective_dimension)
        breakdowns.append(breakdown)

    stacked = torch.stack(outputs, dim=0)
    output_shape = tuple(u0.shape[:-1]) + tuple(times_t.shape) + (u0.shape[-1],)
    values = stacked.reshape(output_shape)
    if u0.ndim == 1:
        values = values.reshape(tuple(times_t.shape) + (u0.shape[-1],))
    diagnostics = KrylovDiagnostics(
        dimension,
        tuple(dimensions),
        tuple(breakdowns),
        normalized_operator.symmetric,
        normalized_operator.representation,
        normalized_operator.positive_semidefinite,
    )
    return values, diagnostics


def adaptive_lanczos_mittag_leffler_action(
    operator: torch.Tensor | SelfAdjointLinearOperator,
    u0: torch.Tensor,
    times: torch.Tensor | float,
    alpha: torch.Tensor | float,
    *,
    beta: torch.Tensor | float = 2.0,
    dimension_schedule: tuple[int, ...] = (8, 16, 24, 32, 48, 64),
    rtol: float | None = None,
    atol: float | None = None,
    breakdown_tol: float = 1e-12,
    config: MLSLConfig | None = None,
    strict: bool = False,
) -> tuple[torch.Tensor, AdaptiveKrylovDiagnostics]:
    """Adapt the Lanczos dimension using successive action disagreement.

    The stop test is detached from autograd.  Gradients through the selected
    reduced action remain available, including gradients with respect to the
    initial state and fractional orders within a fixed selected path.
    """

    schedule = tuple(int(value) for value in dimension_schedule)
    if len(schedule) < 2 or any(value < 1 for value in schedule):
        raise ValueError("dimension_schedule must contain at least two positive entries")
    if any(right <= left for left, right in zip(schedule, schedule[1:])):
        raise ValueError("dimension_schedule must be strictly increasing")
    active_rtol = (1e-9 if u0.dtype == torch.float64 else 1e-4) if rtol is None else float(rtol)
    active_atol = (1e-11 if u0.dtype == torch.float64 else 1e-6) if atol is None else float(atol)

    previous: torch.Tensor | None = None
    selected_values: torch.Tensor | None = None
    selected_diagnostics: KrylovDiagnostics | None = None
    attempted: list[int] = []
    relative_history: list[float] = []
    absolute_history: list[float] = []
    converged = False
    operator_size = operator.shape[-1] if torch.is_tensor(operator) else operator.size

    for requested in schedule:
        dimension = min(requested, int(operator_size))
        if attempted and dimension == attempted[-1]:
            continue
        values, diagnostics = lanczos_mittag_leffler_action(
            operator,
            u0,
            times,
            alpha,
            beta=beta,
            krylov_dimension=dimension,
            breakdown_tol=breakdown_tol,
            config=config,
        )
        attempted.append(dimension)
        selected_values, selected_diagnostics = values, diagnostics
        if previous is not None:
            with torch.no_grad():
                delta = torch.linalg.vector_norm(values.detach() - previous.detach())
                scale = torch.linalg.vector_norm(values.detach())
                absolute = float(delta.cpu())
                relative = float((delta / scale.clamp_min(active_atol)).cpu())
                tolerance = active_atol + active_rtol * float(scale.cpu())
            absolute_history.append(absolute)
            relative_history.append(relative)
            if bool(torch.isfinite(values.detach()).all().item()) and absolute <= tolerance:
                converged = True
                break
        previous = values

    assert selected_values is not None and selected_diagnostics is not None
    if strict and not converged:
        raise RuntimeError("adaptive Lanczos dimension schedule exhausted before tolerance was met")
    diagnostics = AdaptiveKrylovDiagnostics(
        converged=converged,
        selected_dimension=attempted[-1],
        attempted_dimensions=tuple(attempted),
        relative_disagreements=tuple(relative_history),
        absolute_disagreements=tuple(absolute_history),
        final=selected_diagnostics,
    )
    return selected_values, diagnostics
