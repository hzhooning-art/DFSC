"""Operator adapters for dfsc spectral primitives.

The routines here broaden dfsc beyond built-in tensor-product grids by accepting
finite-dimensional self-adjoint positive semidefinite operators. This covers
graph Laplacians, finite-element stiffness/mass-reduced operators, and other
retained discrete operators after the user has supplied the discretization.
"""

from __future__ import annotations

import torch

from .factory import MLSLConfig
from .spectral_layer import MittagLefflerSpectralLayer


def spectral_decomposition_from_operator(
    operator: torch.Tensor,
    *,
    num_modes: int | None = None,
    symmetry_tol: float = 1e-8,
    psd_tol: float = 1e-10,
    drop_zero_modes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return retained eigenpairs from a real symmetric PSD operator matrix."""

    if not torch.is_tensor(operator):
        operator = torch.as_tensor(operator)
    if not torch.is_floating_point(operator):
        operator = operator.to(dtype=torch.float64)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be a square matrix")
    if torch.is_complex(operator):
        raise ValueError("complex operators are not implemented in this adapter")
    if not torch.isfinite(operator).all().item():
        raise ValueError("operator entries must be finite")

    asymmetry = torch.linalg.norm(operator - operator.transpose(-1, -2))
    scale = torch.linalg.norm(operator).clamp_min(torch.finfo(operator.dtype).eps)
    if float((asymmetry / scale).detach().cpu()) > symmetry_tol:
        raise ValueError("operator must be symmetric within symmetry_tol")
    operator = 0.5 * (operator + operator.transpose(-1, -2))

    eigenvalues, eigenvectors = torch.linalg.eigh(operator)
    if float(eigenvalues.min().detach().cpu()) < -psd_tol:
        raise ValueError("operator must be positive semidefinite within psd_tol")
    eigenvalues = eigenvalues.clamp_min(0.0)

    if drop_zero_modes:
        keep = eigenvalues > psd_tol
        eigenvalues = eigenvalues[keep]
        eigenvectors = eigenvectors[:, keep]

    if num_modes is not None:
        if num_modes <= 0:
            raise ValueError("num_modes must be positive")
        eigenvalues = eigenvalues[:num_modes]
        eigenvectors = eigenvectors[:, :num_modes]

    return eigenvalues, eigenvectors


def build_operator_mlsl(
    operator: torch.Tensor,
    *,
    num_modes: int | None = None,
    config: MLSLConfig | None = None,
    symmetry_tol: float = 1e-8,
    psd_tol: float = 1e-10,
    drop_zero_modes: bool = False,
) -> MittagLefflerSpectralLayer:
    """Build an MLSL layer from a user-supplied symmetric PSD operator."""

    cfg = MLSLConfig() if config is None else config
    op = operator.to(dtype=cfg.dtype, device=cfg.device) if torch.is_tensor(operator) else torch.as_tensor(
        operator, dtype=cfg.dtype, device=cfg.device
    )
    eigenvalues, eigenvectors = spectral_decomposition_from_operator(
        op,
        num_modes=num_modes,
        symmetry_tol=symmetry_tol,
        psd_tol=psd_tol,
        drop_zero_modes=drop_zero_modes,
    )
    return MittagLefflerSpectralLayer(
        eigenvalues,
        eigenvectors,
        terms=cfg.terms,
        wave_speed=cfg.wave_speed,
        beta=cfg.beta,
        custom_backward=cfg.custom_backward,
        ml_method=cfg.ml_method,
    )


def generalized_spectral_decomposition(
    stiffness: torch.Tensor,
    mass: torch.Tensor,
    *,
    num_modes: int | None = None,
    symmetry_tol: float = 1e-8,
    psd_tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve ``K phi = lambda M phi`` for symmetric ``K`` and SPD ``M``.

    Returns eigenvalues, reconstruction vectors ``phi``, and projection
    vectors ``M phi``.  The latter are required because the modes are
    M-orthonormal rather than Euclidean-orthonormal.
    """

    if not torch.is_tensor(stiffness):
        stiffness = torch.as_tensor(stiffness)
    if not torch.is_tensor(mass):
        mass = torch.as_tensor(mass)
    if not torch.is_floating_point(stiffness):
        stiffness = stiffness.to(dtype=torch.float64)
    if not torch.is_floating_point(mass):
        mass = mass.to(dtype=torch.float64)
    mass = mass.to(dtype=stiffness.dtype, device=stiffness.device)
    if stiffness.ndim != 2 or stiffness.shape[0] != stiffness.shape[1]:
        raise ValueError("stiffness must be a square matrix")
    if mass.shape != stiffness.shape:
        raise ValueError("mass must have the same square shape as stiffness")
    if torch.is_complex(stiffness) or torch.is_complex(mass):
        raise ValueError("complex generalized operators are not implemented")
    if not torch.isfinite(stiffness).all().item() or not torch.isfinite(mass).all().item():
        raise ValueError("stiffness and mass entries must be finite")

    def relative_asymmetry(matrix: torch.Tensor) -> float:
        scale = torch.linalg.norm(matrix).clamp_min(torch.finfo(matrix.dtype).eps)
        return float((torch.linalg.norm(matrix - matrix.transpose(-1, -2)) / scale).detach().cpu())

    if relative_asymmetry(stiffness) > symmetry_tol or relative_asymmetry(mass) > symmetry_tol:
        raise ValueError("stiffness and mass must be symmetric within symmetry_tol")
    stiffness = 0.5 * (stiffness + stiffness.transpose(-1, -2))
    mass = 0.5 * (mass + mass.transpose(-1, -2))
    try:
        chol = torch.linalg.cholesky(mass)
    except RuntimeError as exc:
        raise ValueError("mass must be positive definite") from exc

    left = torch.linalg.solve_triangular(chol, stiffness, upper=False)
    transformed = torch.linalg.solve_triangular(chol, left.transpose(-1, -2), upper=False).transpose(-1, -2)
    transformed = 0.5 * (transformed + transformed.transpose(-1, -2))
    eigenvalues, transformed_vectors = spectral_decomposition_from_operator(
        transformed,
        num_modes=num_modes,
        symmetry_tol=symmetry_tol,
        psd_tol=psd_tol,
    )
    eigenvectors = torch.linalg.solve_triangular(
        chol.transpose(-1, -2),
        transformed_vectors,
        upper=True,
    )
    projection_vectors = mass @ eigenvectors
    return eigenvalues, eigenvectors, projection_vectors


def build_generalized_operator_mlsl(
    stiffness: torch.Tensor,
    mass: torch.Tensor,
    *,
    num_modes: int | None = None,
    config: MLSLConfig | None = None,
    symmetry_tol: float = 1e-8,
    psd_tol: float = 1e-10,
) -> MittagLefflerSpectralLayer:
    """Build MLSL from an assembled symmetric stiffness/mass pair."""

    cfg = MLSLConfig() if config is None else config
    stiffness_t = torch.as_tensor(stiffness, dtype=cfg.dtype, device=cfg.device)
    mass_t = torch.as_tensor(mass, dtype=cfg.dtype, device=cfg.device)
    eigenvalues, eigenvectors, projection_vectors = generalized_spectral_decomposition(
        stiffness_t,
        mass_t,
        num_modes=num_modes,
        symmetry_tol=symmetry_tol,
        psd_tol=psd_tol,
    )
    return MittagLefflerSpectralLayer(
        eigenvalues,
        eigenvectors,
        projection_vectors=projection_vectors,
        terms=cfg.terms,
        wave_speed=cfg.wave_speed,
        beta=cfg.beta,
        custom_backward=cfg.custom_backward,
        ml_method=cfg.ml_method,
    )


def graph_laplacian_from_adjacency(adjacency: torch.Tensor, *, normalized: bool = False) -> torch.Tensor:
    """Return an undirected graph Laplacian from a dense adjacency matrix."""

    if not torch.is_tensor(adjacency):
        adjacency = torch.as_tensor(adjacency)
    if not torch.is_floating_point(adjacency):
        adjacency = adjacency.to(dtype=torch.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    adjacency = 0.5 * (adjacency + adjacency.transpose(-1, -2))
    degree = adjacency.sum(dim=-1)
    laplacian = torch.diag(degree) - adjacency
    if not normalized:
        return laplacian
    inv_sqrt = degree.clamp_min(torch.finfo(adjacency.dtype).eps).rsqrt()
    return inv_sqrt[:, None] * laplacian * inv_sqrt[None, :]


def build_graph_mlsl(
    adjacency: torch.Tensor,
    *,
    num_modes: int | None = None,
    normalized: bool = False,
    config: MLSLConfig | None = None,
) -> MittagLefflerSpectralLayer:
    """Build an MLSL layer from a dense undirected graph adjacency matrix."""

    cfg = MLSLConfig() if config is None else config
    adj = adjacency.to(dtype=cfg.dtype, device=cfg.device) if torch.is_tensor(adjacency) else torch.as_tensor(
        adjacency, dtype=cfg.dtype, device=cfg.device
    )
    laplacian = graph_laplacian_from_adjacency(adj, normalized=normalized)
    return build_operator_mlsl(laplacian, num_modes=num_modes, config=cfg)
