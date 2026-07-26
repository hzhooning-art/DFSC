"""Self-adjoint linear-operator contracts for sparse and matrix-free dfsc paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


def _tensor_matvec(matrix: torch.Tensor) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    if matrix.layout == torch.strided:
        return lambda vector: matrix @ vector, "dense"
    sparse_matrix = matrix.to_sparse_coo().coalesce()
    return lambda vector: torch.sparse.mm(sparse_matrix, vector[:, None]).squeeze(1), "sparse"


@dataclass(frozen=True)
class GeneralLinearOperator:
    """Dense, sparse, or matrix-free linear operator without symmetry claims."""

    size: int
    matvec: Callable[[torch.Tensor], torch.Tensor]
    dtype: torch.dtype
    device: torch.device | str
    name: str = "general-matrix-free-operator"
    representation: str = "matrix-free"

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("operator size must be positive")
        if not callable(self.matvec):
            raise TypeError("matvec must be callable")

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim != 1 or vector.numel() != self.size:
            raise ValueError(f"matvec expects a vector with shape ({self.size},)")
        result = self.matvec(vector)
        if not torch.is_tensor(result) or result.shape != vector.shape:
            raise ValueError("matvec must return a tensor with the same shape as its input")
        if result.device != vector.device or result.dtype != vector.dtype:
            raise ValueError("matvec output must preserve input dtype and device")
        return result

    @classmethod
    def from_tensor(cls, matrix: torch.Tensor, *, name: str | None = None) -> "GeneralLinearOperator":
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        if not (matrix.is_floating_point() or torch.is_complex(matrix)):
            raise TypeError("matrix must be real or complex floating point")
        matvec, representation = _tensor_matvec(matrix)
        return cls(
            matrix.shape[0],
            matvec,
            matrix.dtype,
            matrix.device,
            name or f"general-{representation}-tensor-operator",
            representation,
        )


def _sparse_is_symmetric(matrix: torch.Tensor, *, rtol: float, atol: float) -> bool:
    coo = matrix.to_sparse_coo().coalesce()
    transposed = coo.transpose(0, 1).coalesce()
    return bool(
        torch.equal(coo.indices(), transposed.indices())
        and torch.allclose(coo.values(), transposed.values(), rtol=rtol, atol=atol)
    )


@dataclass(frozen=True)
class SelfAdjointLinearOperator:
    """Matrix-free contract used by the dfsc Lanczos implementation.

    ``symmetric`` and ``positive_semidefinite`` are user assertions for a
    callable operator. Lanczos checks dimensions and monitors reduced Ritz
    values, but a finite number of matvecs cannot prove either global property.
    """

    size: int
    matvec: Callable[[torch.Tensor], torch.Tensor]
    dtype: torch.dtype
    device: torch.device | str
    name: str = "matrix-free-operator"
    representation: str = "matrix-free"
    symmetric: bool = True
    positive_semidefinite: bool = True

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("operator size must be positive")
        if not callable(self.matvec):
            raise TypeError("matvec must be callable")

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim != 1 or vector.numel() != self.size:
            raise ValueError(f"matvec expects a vector with shape ({self.size},)")
        result = self.matvec(vector)
        if not torch.is_tensor(result) or result.shape != vector.shape:
            raise ValueError("matvec must return a tensor with the same shape as its input")
        if result.device != vector.device or result.dtype != vector.dtype:
            raise ValueError("matvec output must preserve the input dtype and device")
        return result

    @classmethod
    def from_tensor(
        cls,
        matrix: torch.Tensor,
        *,
        check_symmetry: bool = True,
        name: str | None = None,
    ) -> "SelfAdjointLinearOperator":
        """Wrap a dense or sparse square tensor as a linear operator."""

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        if not matrix.is_floating_point():
            raise TypeError("matrix must be floating point")
        tolerance = 100.0 * torch.finfo(matrix.dtype).eps
        if check_symmetry:
            if matrix.layout == torch.strided:
                symmetric = bool(
                    torch.allclose(matrix.detach(), matrix.detach().transpose(-1, -2), rtol=1e-7, atol=tolerance)
                )
            else:
                symmetric = _sparse_is_symmetric(matrix.detach(), rtol=1e-7, atol=tolerance)
            if not symmetric:
                raise ValueError("matrix must be symmetric")

        matvec, representation = _tensor_matvec(matrix)
        return cls(
            size=matrix.shape[0],
            matvec=matvec,
            dtype=matrix.dtype,
            device=matrix.device,
            name=name or f"{representation}-tensor-operator",
            representation=representation,
        )


def as_self_adjoint_operator(
    operator: torch.Tensor | SelfAdjointLinearOperator,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> SelfAdjointLinearOperator:
    """Normalize tensor and callable operator inputs for Lanczos."""

    if isinstance(operator, SelfAdjointLinearOperator):
        operator_device = torch.device(operator.device)
        same_device = operator_device.type == device.type and (
            operator_device.index is None or operator_device.index == device.index
        )
        if not same_device or operator.dtype != dtype:
            raise ValueError("matrix-free operator dtype/device must match u0")
        if not operator.symmetric:
            raise ValueError("Lanczos MLSL requires a self-adjoint operator contract")
        if not operator.positive_semidefinite:
            raise ValueError("Mittag-Leffler Lanczos currently requires a PSD operator contract")
        return operator
    if not torch.is_tensor(operator):
        raise TypeError("operator must be a tensor or SelfAdjointLinearOperator")
    return SelfAdjointLinearOperator.from_tensor(operator.to(dtype=dtype, device=device))


def as_general_operator(
    operator: torch.Tensor | GeneralLinearOperator,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> GeneralLinearOperator:
    """Normalize a tensor or general callable operator for Arnoldi."""

    if isinstance(operator, GeneralLinearOperator):
        operator_device = torch.device(operator.device)
        same_device = operator_device.type == device.type and (
            operator_device.index is None or operator_device.index == device.index
        )
        if not same_device or operator.dtype != dtype:
            raise ValueError("general operator dtype/device must match the working state")
        return operator
    if not torch.is_tensor(operator):
        raise TypeError("operator must be a tensor or GeneralLinearOperator")
    return GeneralLinearOperator.from_tensor(operator.to(dtype=dtype, device=device))
