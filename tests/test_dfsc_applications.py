from __future__ import annotations

import unittest

import torch

import dfsc


class DfscApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def test_catalog_exposes_four_scoped_domains(self) -> None:
        catalog = dfsc.application_catalog()
        self.assertEqual(len(catalog), 4)
        self.assertEqual({row["fit"] for row in catalog}, {"native", "native-after-discretization", "native-after-graph-construction", "controlled"})
        self.assertTrue(all(row["assumptions"] and row["limitations"] for row in catalog))

    def test_anomalous_diffusion_supports_order_gradients(self) -> None:
        alpha = torch.tensor(0.8, requires_grad=True)
        beta = torch.tensor(1.7, requires_grad=True)
        case = dfsc.anomalous_diffusion_case(
            initial=lambda x: torch.sin(torch.pi * x),
            times=torch.tensor([0.0, 0.02, 0.05]),
            alpha=alpha,
            beta=beta,
            diffusivity=0.1,
            num_points=24,
            num_modes=10,
        )
        solution = case.solve()
        self.assertEqual(solution.values.shape, (3, 24))
        solution.values[-1].square().mean().backward()
        self.assertTrue(torch.isfinite(alpha.grad))
        self.assertTrue(torch.isfinite(beta.grad))
        self.assertEqual(case.summary()["recommended_algorithm"], "mlsl-stable")

    def test_assembled_relaxation_is_mass_aware(self) -> None:
        stiffness = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        mass = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        alpha = torch.tensor(0.9, requires_grad=True)
        case = dfsc.assembled_relaxation_case(
            stiffness=stiffness,
            mass=mass,
            initial=torch.tensor([1.0, 0.0]),
            times=torch.tensor([0.0, 0.1]),
            alpha=alpha,
        )
        solution = case.solve()
        self.assertTrue(solution.stats["mass_projection"])
        solution.values[-1].sum().backward()
        self.assertTrue(torch.isfinite(alpha.grad))

    def test_network_diffusion_preserves_constant_mode(self) -> None:
        adjacency = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        )
        case = dfsc.network_diffusion_case(
            adjacency=adjacency,
            initial=torch.ones(3),
            times=torch.tensor([0.0, 0.2, 0.7]),
            alpha=0.75,
        )
        solution = case.solve()
        torch.testing.assert_close(solution.values, torch.ones_like(solution.values), rtol=1e-11, atol=1e-11)

    def test_advection_diffusion_matches_exponential_and_preserves_gradients(self) -> None:
        alpha = torch.tensor(1.0, requires_grad=True)
        diffusivity = torch.tensor(0.02, requires_grad=True)
        velocity = torch.tensor(0.15, requires_grad=True)
        times = torch.tensor([0.0, 0.01, 0.02])
        initial = torch.linspace(-0.7, 0.9, 8)
        case = dfsc.advection_diffusion_case(
            initial=initial,
            times=times,
            alpha=alpha,
            diffusivity=diffusivity,
            velocity=velocity,
            num_points=8,
            arnoldi_dimension=8,
        )
        solution = case.solve()
        operator = case.problem.operator
        reference = torch.stack([torch.matrix_exp(-time * operator) @ initial for time in times])
        torch.testing.assert_close(solution.values, reference, rtol=1e-11, atol=1e-11)
        weights = torch.linspace(0.2, 1.1, 8)
        (solution.values[-1] * weights).sum().backward()
        for gradient in (alpha.grad, diffusivity.grad, velocity.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient))


if __name__ == "__main__":
    unittest.main()
