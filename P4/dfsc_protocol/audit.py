"""Backend-independent primitive audit helpers exposed as public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

import torch


class PrimitiveBackend(Protocol):
    def __call__(self, inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class PrimitiveDomain:
    input_description: str
    parameter_ranges: Mapping[str, tuple[float, float]]
    output_description: str
    supports_batch: bool
    supports_gpu: bool
    supports_autograd: bool


@dataclass(frozen=True)
class QualificationCriteria:
    """Predeclared numerical gates for one implementation and test domain."""

    value_max_abs_error: float
    gradient_directional_relative_error: float

    def __post_init__(self) -> None:
        if self.value_max_abs_error < 0 or self.gradient_directional_relative_error < 0:
            raise ValueError("qualification tolerances must be non-negative")


@dataclass
class PrimitiveAudit:
    backend: str
    domain: PrimitiveDomain
    gates: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.gates:
            return "incomplete"
        return "conformant" if all(self.gates.values()) else "nonconformant"

    def to_dict(self) -> dict:
        return {"backend": self.backend, "domain": self.domain.__dict__, "status": self.status, "gates": self.gates, "metrics": self.metrics, "warnings": self.warnings}


def audit_value_and_gradient(backend: PrimitiveBackend, inputs: torch.Tensor, parameters: torch.Tensor, reference: torch.Tensor, direction: torch.Tensor, finite_difference_eps: float = 1e-5) -> dict[str, float | bool]:
    params = parameters.detach().clone().requires_grad_(True)
    values = backend(inputs, params)
    value_error = torch.max((values - reference).abs())
    (gradient,) = torch.autograd.grad(values.sum(), params)
    direction = direction.to(device=params.device, dtype=params.dtype)
    plus = backend(inputs, params.detach() + finite_difference_eps * direction)
    minus = backend(inputs, params.detach() - finite_difference_eps * direction)
    finite_directional = ((plus - minus).sum() / (2.0 * finite_difference_eps)).detach()
    autograd_directional = (gradient * direction).sum().detach()
    relative_gradient_error = (finite_directional - autograd_directional).abs() / (finite_directional.abs() + 1e-12)
    return {"value_max_abs_error": float(value_error.detach().cpu()), "gradient_finite": bool(torch.isfinite(gradient).all().item()), "gradient_directional_relative_error": float(relative_gradient_error.cpu()), "values_finite": bool(torch.isfinite(values).all().item())}


def audit_batch_and_device(backend: PrimitiveBackend, inputs: torch.Tensor, parameters: torch.Tensor) -> dict[str, bool]:
    outputs = backend(inputs, parameters)
    return {"batch_shape_preserved": bool(outputs.shape[0] == inputs.shape[0]), "device_local": bool(outputs.device == inputs.device == parameters.device), "outputs_finite": bool(torch.isfinite(outputs).all().item())}


def make_audit(backend_name: str, domain: PrimitiveDomain, value_gradient: Mapping[str, float | bool], batch_device: Mapping[str, bool], warnings: Sequence[str] = ()) -> PrimitiveAudit:
    gates = {"value_finite": bool(value_gradient["values_finite"]), "gradient_finite": bool(value_gradient["gradient_finite"]), "batch_shape": bool(batch_device["batch_shape_preserved"]), "device_local": bool(batch_device["device_local"])}
    metrics = {"value_max_abs_error": float(value_gradient["value_max_abs_error"]), "gradient_directional_relative_error": float(value_gradient["gradient_directional_relative_error"])}
    return PrimitiveAudit(backend_name, domain, gates, metrics, list(warnings))


def qualify_audit(audit: PrimitiveAudit, criteria: QualificationCriteria) -> PrimitiveAudit:
    """Apply predeclared error gates without discarding software-contract gates."""

    gates = dict(audit.gates)
    gates["value_accuracy"] = audit.metrics["value_max_abs_error"] <= criteria.value_max_abs_error
    gates["gradient_accuracy"] = (
        audit.metrics["gradient_directional_relative_error"]
        <= criteria.gradient_directional_relative_error
    )
    warnings = list(audit.warnings)
    for gate in ("value_accuracy", "gradient_accuracy"):
        if not gates[gate]:
            warnings.append(f"qualification gate failed: {gate}")
    return PrimitiveAudit(audit.backend, audit.domain, gates, dict(audit.metrics), warnings)
