from __future__ import annotations

import unittest

import torch

from dfsc import (
    MLSLConfig,
    build_dirichlet_mlsl_1d,
    build_dirichlet_mlsl_2d,
    build_mlsl,
    build_mixed_mlsl_1d,
    build_mixed_mlsl_2d,
    build_neumann_mlsl_1d,
    build_neumann_mlsl_2d,
    build_periodic_mlsl_1d,
    build_periodic_mlsl_2d,
)


class SpectralLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def test_forward_shape_and_alpha_gradient(self) -> None:
        x, layer = build_dirichlet_mlsl_1d(
            num_points=48,
            num_modes=10,
            config=MLSLConfig(terms=80),
        )
        u0 = torch.sin(torch.pi * x)
        alpha = torch.tensor(1.35, requires_grad=True)
        out = layer(u0, torch.linspace(0.0, 0.03, 5), alpha)
        loss = out.square().mean()
        loss.backward()
        self.assertEqual(tuple(out.shape), (5, 48))
        self.assertTrue(torch.isfinite(alpha.grad).item())

    def test_batched_forward_shape(self) -> None:
        x, layer = build_dirichlet_mlsl_1d(
            num_points=40,
            num_modes=8,
            config=MLSLConfig(terms=70),
        )
        u0 = torch.stack([torch.sin(torch.pi * x), torch.sin(2.0 * torch.pi * x)])
        out = layer(u0, torch.linspace(0.0, 0.02, 4), torch.tensor(1.4))
        self.assertEqual(tuple(out.shape), (2, 4, 40))

    def test_beta_is_differentiable(self) -> None:
        x, layer = build_dirichlet_mlsl_1d(
            num_points=48,
            num_modes=10,
            config=MLSLConfig(terms=80),
        )
        u0 = torch.sin(torch.pi * x) + 0.1 * torch.sin(3.0 * torch.pi * x)
        beta = torch.tensor(1.45, requires_grad=True)
        out = layer(u0, torch.tensor(0.02), torch.tensor(1.35), beta=beta)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(beta.grad).item())

    def test_2d_constructor(self) -> None:
        coords, layer = build_dirichlet_mlsl_2d(
            num_points_1d=8,
            num_modes_1d=3,
            config=MLSLConfig(terms=70),
        )
        x = coords[:, 0]
        y = coords[:, 1]
        u0 = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        out = layer(u0, torch.linspace(0.0, 0.01, 3), torch.tensor(1.25))
        self.assertEqual(tuple(out.shape), (3, 64))

    def test_stable_config_handles_zero_time_gradient(self) -> None:
        x, layer = build_dirichlet_mlsl_1d(
            num_points=64,
            num_modes=14,
            config=MLSLConfig.stable(terms=100),
        )
        u0 = torch.sin(torch.pi * x) + 0.2 * torch.sin(4.0 * torch.pi * x)
        alpha = torch.tensor(0.65, requires_grad=True)
        beta = torch.tensor(2.0, requires_grad=True)
        out = layer(u0, torch.linspace(0.0, 0.035, 7), alpha, beta=beta)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertTrue(torch.isfinite(alpha.grad).item())
        self.assertTrue(torch.isfinite(beta.grad).item())

    def test_neumann_constructor_keeps_constant_mode(self) -> None:
        x, layer = build_neumann_mlsl_1d(
            num_points=48,
            num_modes=8,
            config=MLSLConfig.stable(terms=80),
        )
        u0 = torch.ones_like(x)
        out = layer(u0, torch.linspace(0.0, 0.05, 4), torch.tensor(0.85), beta=torch.tensor(1.4))
        self.assertEqual(tuple(out.shape), (4, 48))
        self.assertLess(torch.max(torch.abs(out - u0)).item(), 1e-10)

    def test_periodic_constructor_keeps_constant_mode(self) -> None:
        x, layer = build_periodic_mlsl_1d(
            num_points=64,
            num_modes=15,
            config=MLSLConfig.stable(terms=80),
        )
        u0 = torch.ones_like(x)
        out = layer(u0, torch.linspace(0.0, 0.05, 4), torch.tensor(0.9), beta=torch.tensor(1.2))
        self.assertLess(torch.max(torch.abs(out - u0)).item(), 1e-10)

    def test_mixed_boundary_constructors_are_differentiable(self) -> None:
        for boundary in ["dn", "nd"]:
            x, layer = build_mixed_mlsl_1d(
                num_points=48,
                num_modes=8,
                boundary=boundary,
                config=MLSLConfig.stable(terms=80),
            )
            alpha = torch.tensor(0.85, requires_grad=True)
            beta = torch.tensor(1.4, requires_grad=True)
            u0 = layer.eigenvectors[:, 0].to(dtype=x.dtype)
            out = layer(u0, torch.linspace(0.0, 0.03, 4), alpha, beta=beta)
            out.square().mean().backward()
            self.assertTrue(torch.isfinite(out).all().item())
            self.assertTrue(torch.isfinite(alpha.grad).item())
            self.assertTrue(torch.isfinite(beta.grad).item())

    def test_2d_boundary_constructors(self) -> None:
        builders = [
            build_neumann_mlsl_2d,
            build_periodic_mlsl_2d,
            lambda **kw: build_mixed_mlsl_2d(boundary="dn", **kw),
            lambda **kw: build_mixed_mlsl_2d(boundary="nd", **kw),
        ]
        for build in builders:
            coords, layer = build(
                num_points_1d=7,
                num_modes_1d=3,
                config=MLSLConfig.stable(terms=70),
            )
            alpha = torch.tensor(0.85, requires_grad=True)
            beta = torch.tensor(1.3, requires_grad=True)
            u0 = layer.eigenvectors[:, 0].to(dtype=coords.dtype)
            out = layer(u0, torch.linspace(0.0, 0.02, 3), alpha, beta=beta)
            out.square().mean().backward()
            self.assertEqual(tuple(out.shape), (3, 49))
            self.assertTrue(torch.isfinite(alpha.grad).item())
            self.assertTrue(torch.isfinite(beta.grad).item())

    def test_unified_builder(self) -> None:
        x, layer = build_mlsl(
            dimension=1,
            boundary="periodic",
            num_points=32,
            num_modes=9,
            config=MLSLConfig.stable(terms=60),
        )
        out = layer(torch.ones_like(x), torch.linspace(0.0, 0.02, 3), torch.tensor(0.8))
        self.assertEqual(tuple(out.shape), (3, 32))


if __name__ == "__main__":
    unittest.main()
