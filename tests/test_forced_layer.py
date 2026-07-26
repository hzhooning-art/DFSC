from __future__ import annotations

import unittest

import torch

from dfsc import ForcedMittagLefflerSpectralLayer, MLSLConfig, build_dirichlet_mlsl_1d


class ForcedLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def test_zero_forcing_matches_homogeneous_layer(self) -> None:
        x, base = build_dirichlet_mlsl_1d(
            num_points=32,
            num_modes=8,
            config=MLSLConfig(terms=80),
        )
        layer = ForcedMittagLefflerSpectralLayer(base, forcing_terms=80)
        u0 = torch.sin(torch.pi * x)
        times = torch.linspace(0.0, 0.02, 4)
        forcing_times = (torch.arange(16, dtype=x.dtype) + 0.5) / 16.0
        forcing = torch.zeros(16, x.numel())
        forced = layer(u0, times, torch.tensor(1.25), forcing, forcing_times)
        homogeneous = base(u0, times, torch.tensor(1.25))
        self.assertTrue(torch.allclose(forced, homogeneous, atol=1e-12))

    def test_time_specific_forcing_shape(self) -> None:
        x, base = build_dirichlet_mlsl_1d(
            num_points=32,
            num_modes=8,
            config=MLSLConfig(terms=80),
        )
        layer = ForcedMittagLefflerSpectralLayer(base, forcing_terms=80)
        times = torch.linspace(0.0, 0.02, 4)
        forcing_times = (torch.arange(12, dtype=x.dtype) + 0.5) / 12.0
        spatial = torch.sin(2.0 * torch.pi * x)
        physical_times = times[:, None] * forcing_times[None, :]
        forcing = torch.cos(physical_times)[:, :, None] * spatial[None, None, :]
        out = layer(torch.zeros_like(x), times, torch.tensor(0.75), forcing, forcing_times)
        self.assertEqual(tuple(out.shape), (4, 32))
        self.assertTrue(torch.isfinite(out).all().item())


if __name__ == "__main__":
    unittest.main()
