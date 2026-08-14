"""Minimal executable qualification example for an engineering component."""

import json

import torch

from dfsc_protocol import (
    PrimitiveDomain,
    QualificationCriteria,
    audit_batch_and_device,
    audit_value_and_gradient,
    make_audit,
    qualify_audit,
)


class ExponentialResponse:
    def __call__(self, time: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        amplitude, rate = parameters.unbind()
        return amplitude * torch.exp(-rate * time)


def main() -> None:
    backend = ExponentialResponse()
    time = torch.linspace(0.0, 2.0, 64, dtype=torch.float64)
    parameters = torch.tensor([1.2, 0.7], dtype=torch.float64)
    reference = backend(time, parameters)
    value_gradient = audit_value_and_gradient(
        backend,
        time,
        parameters,
        reference,
        torch.tensor([0.6, -0.8], dtype=torch.float64),
    )
    batch_device = audit_batch_and_device(
        backend,
        time.expand(8, -1),
        parameters,
    )
    raw = make_audit(
        "exponential-response",
        PrimitiveDomain(
            input_description="time in [0, 2]",
            parameter_ranges={"amplitude": (0.5, 2.0), "rate": (0.1, 1.5)},
            output_description="batched scalar response",
            supports_batch=True,
            supports_gpu=True,
            supports_autograd=True,
        ),
        value_gradient,
        batch_device,
    )
    qualified = qualify_audit(raw, QualificationCriteria(1e-12, 1e-6))
    print(json.dumps(qualified.to_dict(), indent=2))


if __name__ == "__main__":
    main()
