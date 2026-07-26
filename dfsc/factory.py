"""Convenience constructors for reusable dfsc MLSL primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .basis import (
    dirichlet_laplacian_1d,
    dirichlet_laplacian_2d,
    mixed_laplacian_1d,
    mixed_laplacian_2d,
    neumann_laplacian_1d,
    neumann_laplacian_2d,
    periodic_laplacian_1d,
    periodic_laplacian_2d,
)
from .spectral_layer import MittagLefflerSpectralLayer


@dataclass(frozen=True)
class MLSLConfig:
    """Configuration for a Mittag-Leffler spectral layer.

    The config intentionally covers only numerical choices shared by the
    constructors. The fractional orders remain runtime inputs so they can be
    learned by gradient-based SciML workflows.
    """

    terms: int = 100
    wave_speed: float = 1.0
    beta: float = 2.0
    custom_backward: bool = True
    ml_method: str = "series"
    dtype: torch.dtype = torch.float64
    device: torch.device | str | None = None

    @classmethod
    def stable(
        cls,
        *,
        terms: int = 120,
        wave_speed: float = 1.0,
        beta: float = 2.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "MLSLConfig":
        """Return a configuration for broader real-negative regimes.

        The stable preset uses the hybrid Mittag-Leffler evaluator. Its backward
        pass is handled by PyTorch through differentiable series/asymptotic
        branches rather than the series-only custom backward.
        """

        return cls(
            terms=terms,
            wave_speed=wave_speed,
            beta=beta,
            custom_backward=False,
            ml_method="hybrid",
            dtype=dtype,
            device=device,
        )


def build_dirichlet_mlsl_1d(
    *,
    num_points: int,
    num_modes: int,
    length: float = 1.0,
    config: MLSLConfig | None = None,
) -> tuple[torch.Tensor, MittagLefflerSpectralLayer]:
    """Build a 1D Dirichlet MLSL layer and return ``(x, layer)``."""

    cfg = MLSLConfig() if config is None else config
    x, eigenvalues, phi = dirichlet_laplacian_1d(
        num_points,
        num_modes,
        length=length,
        dtype=cfg.dtype,
        device=cfg.device,
    )
    layer = MittagLefflerSpectralLayer(
        eigenvalues,
        phi,
        terms=cfg.terms,
        wave_speed=cfg.wave_speed,
        beta=cfg.beta,
        custom_backward=cfg.custom_backward,
        ml_method=cfg.ml_method,
    )
    return x, layer


def build_dirichlet_mlsl_2d(
    *,
    num_points_1d: int,
    num_modes_1d: int,
    length: float = 1.0,
    config: MLSLConfig | None = None,
) -> tuple[torch.Tensor, MittagLefflerSpectralLayer]:
    """Build a 2D tensor-product Dirichlet MLSL layer and return ``(coords, layer)``."""

    cfg = MLSLConfig() if config is None else config
    coords, eigenvalues, phi = dirichlet_laplacian_2d(
        num_points_1d,
        num_modes_1d,
        length=length,
        dtype=cfg.dtype,
        device=cfg.device,
    )
    layer = MittagLefflerSpectralLayer(
        eigenvalues,
        phi,
        terms=cfg.terms,
        wave_speed=cfg.wave_speed,
        beta=cfg.beta,
        custom_backward=cfg.custom_backward,
        ml_method=cfg.ml_method,
    )
    return coords, layer


def build_neumann_mlsl_1d(
    *,
    num_points: int,
    num_modes: int,
    length: float = 1.0,
    config: MLSLConfig | None = None,
) -> tuple[torch.Tensor, MittagLefflerSpectralLayer]:
    """Build a 1D Neumann MLSL layer and return ``(x, layer)``."""

    cfg = MLSLConfig() if config is None else config
    x, eigenvalues, phi = neumann_laplacian_1d(
        num_points,
        num_modes,
        length=length,
        dtype=cfg.dtype,
        device=cfg.device,
    )
    layer = MittagLefflerSpectralLayer(
        eigenvalues,
        phi,
        terms=cfg.terms,
        wave_speed=cfg.wave_speed,
        beta=cfg.beta,
        custom_backward=cfg.custom_backward,
        ml_method=cfg.ml_method,
    )
    return x, layer


def _build_layer_from_basis(
    eigenvalues: torch.Tensor,
    phi: torch.Tensor,
    cfg: MLSLConfig,
) -> MittagLefflerSpectralLayer:
    return MittagLefflerSpectralLayer(
        eigenvalues,
        phi,
        terms=cfg.terms,
        wave_speed=cfg.wave_speed,
        beta=cfg.beta,
        custom_backward=cfg.custom_backward,
        ml_method=cfg.ml_method,
    )


def build_periodic_mlsl_1d(
    *,
    num_points: int,
    num_modes: int,
    length: float = 1.0,
    config: MLSLConfig | None = None,
) -> tuple[torch.Tensor, MittagLefflerSpectralLayer]:
    """Build a 1D periodic MLSL layer and return ``(x, layer)``."""

    cfg = MLSLConfig() if config is None else config
    x, eigenvalues, phi = periodic_laplacian_1d(
        num_points,
        num_modes,
        length=length,
        dtype=cfg.dtype,
        device=cfg.device,
    )
    return x, _build_layer_from_basis(eigenvalues, phi, cfg)


def build_mixed_mlsl_1d(
    *,
    num_points: int,
    num_modes: int,
    boundary: str = "dn",
    length: float = 1.0,
    config: MLSLConfig | None = None,
) -> tuple[torch.Tensor, MittagLefflerSpectralLayer]:
    """Build a 1D mixed-boundary MLSL layer and return ``(x, layer)``."""

    cfg = MLSLConfig() if config is None else config
    x, eigenvalues, phi = mixed_laplacian_1d(
        num_points,
        num_modes,
        boundary=boundary,
        length=length,
        dtype=cfg.dtype,
        device=cfg.device,
    )
    return x, _build_layer_from_basis(eigenvalues, phi, cfg)


def build_neumann_mlsl_2d(
    *,
    num_points_1d: int,
    num_modes_1d: int,
    length: float = 1.0,
    config: MLSLConfig | None = None,
) -> tuple[torch.Tensor, MittagLefflerSpectralLayer]:
    """Build a 2D Neumann MLSL layer and return ``(coords, layer)``."""

    cfg = MLSLConfig() if config is None else config
    coords, eigenvalues, phi = neumann_laplacian_2d(
        num_points_1d,
        num_modes_1d,
        length=length,
        dtype=cfg.dtype,
        device=cfg.device,
    )
    return coords, _build_layer_from_basis(eigenvalues, phi, cfg)


def build_periodic_mlsl_2d(
    *,
    num_points_1d: int,
    num_modes_1d: int,
    length: float = 1.0,
    config: MLSLConfig | None = None,
) -> tuple[torch.Tensor, MittagLefflerSpectralLayer]:
    """Build a 2D periodic MLSL layer and return ``(coords, layer)``."""

    cfg = MLSLConfig() if config is None else config
    coords, eigenvalues, phi = periodic_laplacian_2d(
        num_points_1d,
        num_modes_1d,
        length=length,
        dtype=cfg.dtype,
        device=cfg.device,
    )
    return coords, _build_layer_from_basis(eigenvalues, phi, cfg)


def build_mixed_mlsl_2d(
    *,
    num_points_1d: int,
    num_modes_1d: int,
    boundary: str = "dn",
    length: float = 1.0,
    config: MLSLConfig | None = None,
) -> tuple[torch.Tensor, MittagLefflerSpectralLayer]:
    """Build a 2D mixed-boundary MLSL layer and return ``(coords, layer)``."""

    cfg = MLSLConfig() if config is None else config
    coords, eigenvalues, phi = mixed_laplacian_2d(
        num_points_1d,
        num_modes_1d,
        boundary=boundary,
        length=length,
        dtype=cfg.dtype,
        device=cfg.device,
    )
    return coords, _build_layer_from_basis(eigenvalues, phi, cfg)


def build_mlsl(
    *,
    dimension: int,
    boundary: str,
    num_points: int | None = None,
    num_modes: int | None = None,
    num_points_1d: int | None = None,
    num_modes_1d: int | None = None,
    length: float = 1.0,
    config: MLSLConfig | None = None,
) -> tuple[torch.Tensor, MittagLefflerSpectralLayer]:
    """Build an MLSL layer from a compact boundary/dimension specification.

    Parameters
    ----------
    dimension:
        ``1`` or ``2``.
    boundary:
        One of ``"dirichlet"``, ``"neumann"``, ``"periodic"``, ``"mixed_dn"``,
        or ``"mixed_nd"``.
    """

    if dimension not in {1, 2}:
        raise ValueError("dimension must be 1 or 2")
    if boundary not in {"dirichlet", "neumann", "periodic", "mixed_dn", "mixed_nd"}:
        raise ValueError("unknown boundary")

    if dimension == 1:
        if num_points is None or num_modes is None:
            raise ValueError("num_points and num_modes are required for dimension=1")
        builders_1d = {
            "dirichlet": build_dirichlet_mlsl_1d,
            "neumann": build_neumann_mlsl_1d,
            "periodic": build_periodic_mlsl_1d,
            "mixed_dn": lambda **kw: build_mixed_mlsl_1d(boundary="dn", **kw),
            "mixed_nd": lambda **kw: build_mixed_mlsl_1d(boundary="nd", **kw),
        }
        return builders_1d[boundary](
            num_points=num_points,
            num_modes=num_modes,
            length=length,
            config=config,
        )

    if num_points_1d is None or num_modes_1d is None:
        raise ValueError("num_points_1d and num_modes_1d are required for dimension=2")
    builders_2d = {
        "dirichlet": build_dirichlet_mlsl_2d,
        "neumann": build_neumann_mlsl_2d,
        "periodic": build_periodic_mlsl_2d,
        "mixed_dn": lambda **kw: build_mixed_mlsl_2d(boundary="dn", **kw),
        "mixed_nd": lambda **kw: build_mixed_mlsl_2d(boundary="nd", **kw),
    }
    return builders_2d[boundary](
        num_points_1d=num_points_1d,
        num_modes_1d=num_modes_1d,
        length=length,
        config=config,
    )
