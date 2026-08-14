import unittest

import torch

from dfsc_protocol import (
    PrimitiveDomain,
    QualificationCriteria,
    audit_batch_and_device,
    audit_value_and_gradient,
    make_audit,
    qualify_audit,
)


class ExponentialBackend:
    def __call__(self, inputs, parameters):
        amplitude, rate = parameters.unbind(dim=-1) if parameters.ndim > 1 else parameters.unbind()
        return amplitude[..., None] * torch.exp(-rate[..., None] * inputs)


class AuditApiTests(unittest.TestCase):
    def test_value_gradient_and_batch_contract(self):
        backend = ExponentialBackend()
        inputs = torch.linspace(0.05, 1.0, 8, dtype=torch.float64)
        parameters = torch.tensor([1.2, 0.7], dtype=torch.float64)
        reference = backend(inputs, parameters)
        values = audit_value_and_gradient(backend, inputs, parameters, reference, torch.tensor([0.4, -0.6], dtype=torch.float64))
        batch = audit_batch_and_device(backend, inputs.expand(4, -1), parameters)
        audit = make_audit("exponential", PrimitiveDomain("time", {"rate": (0.1, 1.5)}, "scalar", True, False, True), values, batch)
        qualified = qualify_audit(audit, QualificationCriteria(1e-12, 1e-6))
        self.assertEqual(qualified.status, "conformant")
        self.assertLess(values["gradient_directional_relative_error"], 1e-6)

    def test_accuracy_gate_can_refuse_a_finite_backend(self):
        backend = ExponentialBackend()
        inputs = torch.linspace(0.05, 1.0, 8, dtype=torch.float64)
        parameters = torch.tensor([1.2, 0.7], dtype=torch.float64)
        biased_reference = backend(inputs, parameters) + 1e-3
        values = audit_value_and_gradient(
            backend,
            inputs,
            parameters,
            biased_reference,
            torch.tensor([0.4, -0.6], dtype=torch.float64),
        )
        batch = audit_batch_and_device(backend, inputs.expand(4, -1), parameters)
        audit = make_audit(
            "biased-exponential",
            PrimitiveDomain("time", {"rate": (0.1, 1.5)}, "scalar", True, False, True),
            values,
            batch,
        )
        qualified = qualify_audit(audit, QualificationCriteria(1e-6, 1e-6))
        self.assertEqual(qualified.status, "nonconformant")
        self.assertFalse(qualified.gates["value_accuracy"])
        self.assertIn("qualification gate failed: value_accuracy", qualified.warnings)


if __name__ == "__main__":
    unittest.main()
