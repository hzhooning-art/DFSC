"""Spectral bases for dfsc Mittag-Leffler layers."""

from __future__ import annotations

import math

import torch


def dirichlet_laplacian_1d(
    num_points: int,
    num_modes: int,
    *,
    length: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a sine basis for the 1D Dirichlet Laplacian on ``(0, length)``.

    The continuous eigenpairs are

    ``phi_n(x) = sqrt(2 / length) sin(n pi x / length)``
    ``lambda_n = (n pi / length)^2``.

    We return collocation values at interior grid points. The columns of
    ``phi`` are normalized in the discrete Euclidean norm so that
    ``phi.T @ phi ~= I``. This keeps the prototype layer simple and stable.
    """

    if num_modes > num_points:
        raise ValueError("num_modes must be <= num_points")

    device = torch.device("cpu") if device is None else torch.device(device)
    x = torch.linspace(
        0.0,
        length,
        num_points + 2,
        dtype=dtype,
        device=device,
    )[1:-1]

    n = torch.arange(1, num_modes + 1, dtype=dtype, device=device)
    phi = torch.sin(math.pi * x[:, None] * n[None, :] / length)
    phi = phi / torch.linalg.norm(phi, dim=0, keepdim=True).clamp_min(1e-30)
    eigenvalues = (math.pi * n / length) ** 2
    return x, eigenvalues, phi


def neumann_laplacian_1d(
    num_points: int,
    num_modes: int,
    *,
    length: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a cosine basis for the 1D Neumann Laplacian on ``[0, length]``."""

    if num_modes > num_points:
        raise ValueError("num_modes must be <= num_points")

    device = torch.device("cpu") if device is None else torch.device(device)
    x = (torch.arange(num_points, dtype=dtype, device=device) + 0.5) * length / num_points
    n = torch.arange(0, num_modes, dtype=dtype, device=device)
    phi = torch.cos(math.pi * x[:, None] * n[None, :] / length)
    phi = phi / torch.linalg.norm(phi, dim=0, keepdim=True).clamp_min(1e-30)
    eigenvalues = (math.pi * n / length) ** 2
    return x, eigenvalues, phi


def periodic_laplacian_1d(
    num_points: int,
    num_modes: int,
    *,
    length: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a real Fourier basis for the 1D periodic Laplacian."""

    if num_modes > num_points:
        raise ValueError("num_modes must be <= num_points")

    device = torch.device("cpu") if device is None else torch.device(device)
    x = torch.arange(num_points, dtype=dtype, device=device) * length / num_points
    columns = [torch.ones_like(x)]
    eigenvalues = [torch.zeros((), dtype=dtype, device=device)]
    k = 1
    while len(columns) < num_modes:
        lam = torch.as_tensor((2.0 * math.pi * k / length) ** 2, dtype=dtype, device=device)
        columns.append(torch.cos(2.0 * math.pi * k * x / length))
        eigenvalues.append(lam)
        if len(columns) < num_modes:
            columns.append(torch.sin(2.0 * math.pi * k * x / length))
            eigenvalues.append(lam)
        k += 1

    phi = torch.stack(columns, dim=1)
    phi = phi / torch.linalg.norm(phi, dim=0, keepdim=True).clamp_min(1e-30)
    return x, torch.stack(eigenvalues), phi


def mixed_laplacian_1d(
    num_points: int,
    num_modes: int,
    *,
    boundary: str = "dn",
    length: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a 1D mixed Dirichlet/Neumann Laplacian basis.

    ``boundary="dn"`` means Dirichlet at ``x=0`` and Neumann at ``x=L``.
    ``boundary="nd"`` means Neumann at ``x=0`` and Dirichlet at ``x=L``.
    """

    if boundary not in {"dn", "nd"}:
        raise ValueError("boundary must be 'dn' or 'nd'")
    if num_modes > num_points:
        raise ValueError("num_modes must be <= num_points")

    device = torch.device("cpu") if device is None else torch.device(device)
    x = torch.linspace(0.0, length, num_points + 2, dtype=dtype, device=device)[1:-1]
    n = torch.arange(0, num_modes, dtype=dtype, device=device)
    wavenumbers = (n + 0.5) * math.pi / length
    if boundary == "dn":
        phi = torch.sin(x[:, None] * wavenumbers[None, :])
    else:
        phi = torch.cos(x[:, None] * wavenumbers[None, :])
    phi = phi / torch.linalg.norm(phi, dim=0, keepdim=True).clamp_min(1e-30)
    return x, wavenumbers**2, phi


def tensor_product_laplacian_2d(
    x: torch.Tensor,
    lam_1d: torch.Tensor,
    phi_1d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a 2D tensor-product basis from a 1D spectral basis."""

    xx, yy = torch.meshgrid(x, x, indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    values = []
    modes = []
    for i in range(lam_1d.numel()):
        for j in range(lam_1d.numel()):
            values.append(lam_1d[i] + lam_1d[j])
            mode = torch.outer(phi_1d[:, i], phi_1d[:, j]).reshape(-1)
            modes.append(mode / torch.linalg.norm(mode).clamp_min(1e-30))
    return coords, torch.stack(values), torch.stack(modes, dim=1)


def dirichlet_laplacian_2d(
    num_points_1d: int,
    num_modes_1d: int,
    *,
    length: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a tensor-product sine basis for the 2D Dirichlet Laplacian."""

    x, lam_1d, phi_1d = dirichlet_laplacian_1d(
        num_points_1d,
        num_modes_1d,
        length=length,
        dtype=dtype,
        device=device,
    )
    return tensor_product_laplacian_2d(x, lam_1d, phi_1d)


def neumann_laplacian_2d(
    num_points_1d: int,
    num_modes_1d: int,
    *,
    length: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a tensor-product cosine basis for the 2D Neumann Laplacian."""

    x, lam_1d, phi_1d = neumann_laplacian_1d(
        num_points_1d,
        num_modes_1d,
        length=length,
        dtype=dtype,
        device=device,
    )
    return tensor_product_laplacian_2d(x, lam_1d, phi_1d)


def periodic_laplacian_2d(
    num_points_1d: int,
    num_modes_1d: int,
    *,
    length: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a tensor-product real Fourier basis for the 2D periodic Laplacian."""

    x, lam_1d, phi_1d = periodic_laplacian_1d(
        num_points_1d,
        num_modes_1d,
        length=length,
        dtype=dtype,
        device=device,
    )
    return tensor_product_laplacian_2d(x, lam_1d, phi_1d)


def mixed_laplacian_2d(
    num_points_1d: int,
    num_modes_1d: int,
    *,
    boundary: str = "dn",
    length: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a tensor-product mixed-boundary basis for the 2D Laplacian."""

    x, lam_1d, phi_1d = mixed_laplacian_1d(
        num_points_1d,
        num_modes_1d,
        boundary=boundary,
        length=length,
        dtype=dtype,
        device=device,
    )
    return tensor_product_laplacian_2d(x, lam_1d, phi_1d)
