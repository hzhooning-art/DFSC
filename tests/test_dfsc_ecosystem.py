from __future__ import annotations

import unittest

import torch
from torch import nn

import dfsc


class DfscEcosystemTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def test_canonical_library_name(self) -> None:
        self.assertEqual(dfsc.LIBRARY_NAME, "dfsc")
        summary = dfsc.component_summary()
        self.assertEqual(summary["library_name"], "dfsc")
        self.assertEqual(summary["python_package"], "dfsc")
        self.assertEqual(summary["compatibility_aliases"], [])

    def test_component_registry_has_implemented_entries(self) -> None:
        components = dfsc.implemented_components()
        names = {row["name"] for row in components}
        self.assertIn("Mittag-Leffler spectral layer", names)
        self.assertIn("Hybrid residual workflow", names)
        self.assertGreaterEqual(len(components), 5)

    def test_trainable_orders_are_bounded_and_differentiable(self) -> None:
        orders = dfsc.make_trainable_orders(alpha_init=0.9, beta_init=1.4)
        loss = orders.alpha.square() + orders.beta.square()
        loss.backward()
        self.assertGreater(float(orders.alpha.detach()), 0.2)
        self.assertLess(float(orders.alpha.detach()), 1.8)
        self.assertTrue(torch.isfinite(orders.raw_alpha.grad).item())
        self.assertTrue(torch.isfinite(orders.raw_beta.grad).item())

    def test_scalar_relaxation_hybrid_preserves_initial_value_and_gradients(self) -> None:
        head = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1)).double()
        model = dfsc.MittagLefflerResidualRegressor(head, alpha_init=0.8, rate_init=0.1).double()
        times = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
        prediction = model(times, torch.log1p(times).unsqueeze(-1))
        torch.testing.assert_close(prediction[0], torch.tensor(1.0, dtype=torch.float64))
        prediction[-1].backward()
        self.assertTrue(torch.isfinite(model.raw_alpha.grad))
        self.assertTrue(torch.isfinite(model.raw_rate.grad))

    def test_dfsc_builder_and_hybrid_model(self) -> None:
        x, layer = dfsc.build_mlsl(
            dimension=1,
            boundary="dirichlet",
            num_points=32,
            num_modes=8,
            config=dfsc.MLSLConfig.stable(terms=60),
        )
        head = nn.Linear(32, 32, bias=False, dtype=torch.float64)
        with torch.no_grad():
            head.weight.zero_()
        model = dfsc.HybridResidualModel(layer, head)
        u0 = torch.sin(torch.pi * x)
        out = model(u0, torch.linspace(0.0, 0.02, 3), torch.tensor(0.9), beta=torch.tensor(1.4))
        self.assertEqual(tuple(out.shape), (3, 32))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_operator_and_graph_adapters(self) -> None:
        n = 12
        adjacency = torch.zeros(n, n)
        for i in range(n - 1):
            adjacency[i, i + 1] = 1.0
            adjacency[i + 1, i] = 1.0
        layer = dfsc.build_graph_mlsl(
            adjacency,
            num_modes=8,
            config=dfsc.MLSLConfig.stable(terms=40),
        )
        u0 = torch.linspace(0.0, 1.0, n)
        out = layer(u0, torch.linspace(0.0, 0.01, 3), torch.tensor(0.9), beta=torch.tensor(1.4))
        self.assertEqual(tuple(out.shape), (3, n))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_generalized_operator_adapter_uses_mass_projection(self) -> None:
        stiffness = torch.diag(torch.tensor([0.0, 2.0, 9.0, 20.0]))
        mass = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        eigenvalues, eigenvectors, projection = dfsc.generalized_spectral_decomposition(stiffness, mass)
        residual = stiffness @ eigenvectors - mass @ eigenvectors * eigenvalues.unsqueeze(0)
        self.assertLess(torch.linalg.norm(residual).item(), 1e-12)
        self.assertTrue(torch.allclose(eigenvectors.transpose(0, 1) @ projection, torch.eye(4), atol=1e-12))
        layer = dfsc.build_generalized_operator_mlsl(
            stiffness,
            mass,
            config=dfsc.MLSLConfig.stable(terms=60),
        )
        u0 = torch.tensor([0.3, -0.4, 0.2, 0.7])
        out = layer(u0, torch.tensor([0.0]), torch.tensor(0.9))
        self.assertTrue(torch.allclose(out[0], u0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
