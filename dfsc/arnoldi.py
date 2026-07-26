"""Controlled Arnoldi Mittag-Leffler actions for non-self-adjoint operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .linear_operators import GeneralLinearOperator, as_general_operator


@dataclass(frozen=True)
class ArnoldiDiagnostics:
    requested_dimension: int
    effective_dimensions: tuple[int, ...]
    breakdowns: tuple[bool, ...]
    representation: str
    max_reduced_radius: float
    observed_reduced_radius: float
    max_reduced_nonnormality: float

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_arnoldi_dimension": self.requested_dimension,
            "effective_arnoldi_dimensions": list(self.effective_dimensions),
            "min_effective_arnoldi_dimension": min(self.effective_dimensions),
            "max_effective_arnoldi_dimension": max(self.effective_dimensions),
            "arnoldi_breakdown_count": sum(self.breakdowns),
            "operator_representation": self.representation,
            "max_reduced_radius": self.max_reduced_radius,
            "observed_reduced_radius": self.observed_reduced_radius,
            "max_reduced_nonnormality": self.max_reduced_nonnormality,
            "validated_domain": f"reduced ||-t^alpha H||_2 <= {self.max_reduced_radius}",
        }


def _matrix_mittag_leffler_series(
    alpha: torch.Tensor,
    matrix: torch.Tensor,
    *,
    terms: int,
) -> torch.Tensor:
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    result = identity
    power = identity
    for index in range(1, terms):
        power = power @ matrix
        coefficient = torch.exp(-torch.lgamma(alpha * index + 1.0))
        result = result + coefficient * power
    return result


def _arnoldi_basis(
    operator: GeneralLinearOperator,
    vector: torch.Tensor,
    *,
    dimension: int,
    breakdown_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    norm = torch.linalg.vector_norm(vector)
    if float(norm.detach().cpu()) == 0.0:
        return vector.new_zeros((vector.numel(), 0)), vector.new_zeros((0, 0)), True
    basis = [vector / norm]
    columns: list[torch.Tensor] = []
    breakdown = False
    for column_index in range(dimension):
        candidate = operator(basis[column_index])
        entries: list[torch.Tensor] = []
        for basis_vector in basis:
            coefficient = torch.vdot(basis_vector, candidate)
            entries.append(coefficient)
            candidate = candidate - coefficient * basis_vector
        # A second modified Gram-Schmidt pass improves orthogonality for
        # non-normal reduced problems without changing the public contract.
        for index, basis_vector in enumerate(basis):
            correction = torch.vdot(basis_vector, candidate)
            entries[index] = entries[index] + correction
            candidate = candidate - correction * basis_vector
        if column_index < dimension - 1:
            next_norm = torch.linalg.vector_norm(candidate)
            scale = max(1.0, float(torch.abs(entries[column_index].detach()).cpu()))
            if float(next_norm.detach().cpu()) <= breakdown_tol * scale:
                breakdown = True
                columns.append(torch.stack(entries))
                break
            entries.append(next_norm.to(dtype=vector.dtype))
            basis.append(candidate / next_norm)
        columns.append(torch.stack(entries))

    effective = len(columns)
    hessenberg = vector.new_zeros((effective, effective))
    for column_index, column in enumerate(columns):
        active = min(column.numel(), effective)
        hessenberg[:active, column_index] = column[:active]
    return torch.stack(basis[:effective], dim=1), hessenberg, breakdown


def _single_action(
    operator: GeneralLinearOperator,
    vector: torch.Tensor,
    times: torch.Tensor,
    alpha: torch.Tensor,
    *,
    dimension: int,
    terms: int,
    max_reduced_radius: float,
    breakdown_tol: float,
    allow_unvalidated: bool,
) -> tuple[torch.Tensor, int, bool, float, float]:
    norm = torch.linalg.vector_norm(vector)
    if float(norm.detach().cpu()) == 0.0:
        shape = tuple(times.shape) + (vector.numel(),) if times.ndim else (vector.numel(),)
        return vector.new_zeros(shape), 0, True, 0.0, 0.0
    basis, hessenberg, breakdown = _arnoldi_basis(
        operator, vector, dimension=dimension, breakdown_tol=breakdown_tol
    )
    tiny = torch.finfo(times.dtype).tiny
    factors = torch.where(times > 0, times.clamp_min(tiny).pow(alpha), torch.zeros_like(times))
    radius = float(torch.linalg.matrix_norm(hessenberg.detach(), ord=2).cpu())
    max_factor = float(torch.max(factors.detach()).cpu()) if factors.numel() else float(factors.detach().cpu())
    observed_radius = radius * max_factor
    if observed_radius > max_reduced_radius and not allow_unvalidated:
        raise ValueError(
            f"Arnoldi reduced radius {observed_radius:.6g} exceeds validated limit {max_reduced_radius:.6g}"
        )
    adjoint = hessenberg.transpose(-1, -2).conj()
    denominator = torch.linalg.matrix_norm(hessenberg.detach()).square().clamp_min(
        torch.finfo(times.dtype).eps
    )
    nonnormality = float(
        (torch.linalg.matrix_norm(hessenberg @ adjoint - adjoint @ hessenberg).detach() / denominator).cpu()
    )
    states = []
    first = torch.zeros(hessenberg.shape[0], dtype=vector.dtype, device=vector.device)
    first[0] = 1.0
    for factor in factors.reshape(-1):
        reduced_function = _matrix_mittag_leffler_series(
            alpha, -factor * hessenberg, terms=terms
        )
        states.append(norm * (basis @ (reduced_function @ first)))
    output = torch.stack(states).reshape(tuple(times.shape) + (vector.numel(),))
    return output, basis.shape[1], breakdown, observed_radius, nonnormality


def arnoldi_mittag_leffler_action(
    operator: torch.Tensor | GeneralLinearOperator,
    u0: torch.Tensor,
    times: torch.Tensor | float,
    alpha: torch.Tensor | float,
    *,
    arnoldi_dimension: int = 32,
    terms: int = 120,
    max_reduced_radius: float = 4.0,
    breakdown_tol: float = 1e-12,
    allow_unvalidated: bool = False,
) -> tuple[torch.Tensor, ArnoldiDiagnostics]:
    """Apply ``E_alpha(-t^alpha A)`` through a controlled Arnoldi projection."""

    if not torch.is_tensor(u0) or not (u0.is_floating_point() or torch.is_complex(u0)):
        raise TypeError("u0 must be a real or complex floating-point tensor")
    if torch.is_tensor(operator):
        working_dtype = torch.promote_types(operator.dtype, u0.dtype)
    else:
        working_dtype = operator.dtype
    working_u0 = u0.to(dtype=working_dtype)
    normalized_operator = as_general_operator(
        operator, dtype=working_dtype, device=working_u0.device
    )
    if working_u0.shape[-1] != normalized_operator.size:
        raise ValueError("u0 last dimension must match operator size")
    dimension = min(int(arnoldi_dimension), normalized_operator.size)
    if dimension < 1 or terms < 2:
        raise ValueError("arnoldi_dimension must be positive and terms must be >= 2")
    real_dtype = working_u0.real.dtype if torch.is_complex(working_u0) else working_u0.dtype
    times_t = torch.as_tensor(times, dtype=real_dtype, device=working_u0.device)
    if bool(torch.any(times_t < 0).item()):
        raise ValueError("times must be non-negative")
    alpha_t = torch.as_tensor(alpha, dtype=real_dtype, device=working_u0.device)
    if alpha_t.numel() != 1 or not bool(((alpha_t > 0) & (alpha_t <= 2)).detach().item()):
        raise ValueError("Arnoldi MLSL requires scalar 0 < alpha <= 2")

    flat = working_u0.reshape(-1, working_u0.shape[-1])
    outputs = []
    dimensions = []
    breakdowns = []
    radii = []
    nonnormalities = []
    for vector in flat:
        output, effective, breakdown, radius, nonnormality = _single_action(
            normalized_operator,
            vector,
            times_t,
            alpha_t,
            dimension=dimension,
            terms=terms,
            max_reduced_radius=max_reduced_radius,
            breakdown_tol=breakdown_tol,
            allow_unvalidated=allow_unvalidated,
        )
        outputs.append(output)
        dimensions.append(effective)
        breakdowns.append(breakdown)
        radii.append(radius)
        nonnormalities.append(nonnormality)
    values = torch.stack(outputs).reshape(
        tuple(working_u0.shape[:-1]) + tuple(times_t.shape) + (working_u0.shape[-1],)
    )
    if working_u0.ndim == 1:
        values = values.reshape(tuple(times_t.shape) + (working_u0.shape[-1],))
    diagnostics = ArnoldiDiagnostics(
        dimension,
        tuple(dimensions),
        tuple(breakdowns),
        normalized_operator.representation,
        max_reduced_radius,
        max(radii),
        max(nonnormalities),
    )
    return values, diagnostics
