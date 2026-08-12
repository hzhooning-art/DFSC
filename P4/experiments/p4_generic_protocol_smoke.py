"""Backend-independent smoke test using a toy differentiable propagator.

This is an interface test, not a scientific benchmark. It proves that the
protocol can audit a primitive that is not MLSL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4"))
from primitive_protocol import PrimitiveDomain, audit_batch_and_device, audit_value_and_gradient, make_audit


class ExponentialPropagator:
    def __call__(self, inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        amplitude, rate = parameters.unbind(dim=-1)
        return amplitude[..., None] * torch.exp(-rate[..., None] * inputs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    backend = ExponentialPropagator()
    inputs = torch.linspace(0.05, 1.5, 32, dtype=dtype, device=device)
    parameters = torch.tensor([1.2, 0.7], dtype=dtype, device=device)
    reference = backend(inputs, parameters)
    value_gradient = audit_value_and_gradient(
        backend,
        inputs,
        parameters,
        reference,
        direction=torch.tensor([0.4, -0.6], dtype=dtype, device=device),
    )
    batch_inputs = inputs.expand(16, -1)
    batch_parameters = parameters.expand(16, -1)
    batch_device = audit_batch_and_device(backend, batch_inputs, batch_parameters)
    audit = make_audit(
        "exponential_propagator_demo",
        PrimitiveDomain(
            input_description="batched scalar time points",
            parameter_ranges={"amplitude": (0.5, 2.0), "rate": (0.1, 1.5)},
            output_description="batched propagated scalar field",
            supports_batch=True,
            supports_gpu=torch.cuda.is_available(),
            supports_autograd=True,
        ),
        value_gradient,
        batch_device,
        warnings=["interface smoke test only; not a domain validation study"],
    )
    out = ROOT / "P4" / "results" / "p4_generic_protocol_smoke.json"
    out.write_text(json.dumps(audit.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(audit.to_dict(), indent=2))


if __name__ == "__main__":
    main()
