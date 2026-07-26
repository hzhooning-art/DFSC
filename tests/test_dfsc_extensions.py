from __future__ import annotations

import unittest

import torch
from torch import nn

import dfsc


class DfscExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def test_applicability_report_accepts_retained_real_spectrum(self) -> None:
        x, layer = dfsc.build_mlsl(
            dimension=1,
            boundary="dirichlet",
            num_points=24,
            num_modes=6,
            config=dfsc.MLSLConfig.stable(terms=40),
        )
        report = dfsc.mlsl_applicability_report(layer.eigenvalues, layer.eigenvectors)
        self.assertTrue(report.supported)
        self.assertIn("spectral", " ".join(report.assumptions))
        self.assertEqual(x.shape[0], 24)

    def test_applicability_report_rejects_complex_spectrum(self) -> None:
        evals = torch.tensor([1.0 + 0.1j, 2.0 + 0.0j])
        report = dfsc.mlsl_applicability_report(evals)
        self.assertFalse(report.supported)
        self.assertTrue(any("complex" in reason for reason in report.unsupported_reasons))

    def test_variable_order_wrapper_is_differentiable(self) -> None:
        x, layer = dfsc.build_mlsl(
            dimension=1,
            boundary="dirichlet",
            num_points=24,
            num_modes=6,
            config=dfsc.MLSLConfig.stable(terms=40),
        )
        raw = torch.tensor(0.0, requires_grad=True)

        class OrderFn(nn.Module):
            def forward(self, times: torch.Tensor) -> torch.Tensor:
                return 0.8 + 0.2 * torch.sigmoid(raw) + 0.05 * times

        wrapper = dfsc.VariableOrderMLSL(layer, OrderFn())
        u0 = torch.sin(torch.pi * x)
        out = wrapper(u0, torch.linspace(0.0, 0.02, 4), beta=torch.tensor(1.5))
        loss = out.square().mean()
        loss.backward()
        self.assertEqual(tuple(out.shape), (4, 24))
        self.assertTrue(torch.isfinite(raw.grad).item())

    def test_distributed_order_wrapper_normalizes_weights(self) -> None:
        x, layer = dfsc.build_mlsl(
            dimension=1,
            boundary="dirichlet",
            num_points=24,
            num_modes=6,
            config=dfsc.MLSLConfig.stable(terms=40),
        )
        wrapper = dfsc.DistributedOrderMLSL(
            layer,
            torch.tensor([0.7, 1.0, 1.3]),
            trainable_weights=True,
        )
        u0 = torch.sin(torch.pi * x)
        out = wrapper(u0, torch.linspace(0.0, 0.02, 3), beta=torch.tensor(1.4))
        loss = out.square().mean()
        loss.backward()
        self.assertEqual(tuple(out.shape), (3, 24))
        self.assertAlmostEqual(float(wrapper.normalized_weights.sum().detach()), 1.0, places=12)
        self.assertTrue(torch.isfinite(wrapper.logits.grad).all().item())

    def test_gap_report_exposes_active_limitations(self) -> None:
        report = dfsc.ecosystem_gap_report()
        self.assertIn("active_limitations", report)
        self.assertTrue(any("PyPI" in item for item in report["active_limitations"]))


if __name__ == "__main__":
    unittest.main()
