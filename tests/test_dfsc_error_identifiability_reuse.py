from __future__ import annotations

import unittest

import torch

import dfsc


class ErrorBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def test_alternating_bound_contains_richer_series_error(self) -> None:
        z = -torch.tensor([0.1, 0.5, 1.0])
        alpha = torch.tensor(0.8)
        terms = 24
        value = dfsc.mittag_leffler_e(alpha, z, terms=terms)
        reference = dfsc.mittag_leffler_e(alpha, z, terms=120)
        bound = dfsc.alternating_series_remainder_bound(alpha, z, terms=terms)
        self.assertIsNotNone(bound)
        assert bound is not None
        self.assertTrue(torch.all(torch.abs(value - reference) <= bound * (1 + 1e-10) + 1e-15))

    def test_unavailable_certificate_is_explicit(self) -> None:
        bound = dfsc.alternating_series_remainder_bound(1.2, -torch.tensor([1.0]), terms=20)
        self.assertIsNone(bound)
        report = dfsc.compose_error_budget_report(
            dfsc.ErrorBudget(),
            reference_norm=1.0,
            evaluator_estimate=1e-12,
            evaluator_rigorous=True,
        )
        self.assertFalse(report.assessed)
        self.assertFalse(report.rigorous_global_bound)


class IdentifiabilityTests(unittest.TestCase):
    def test_known_quadratic_hessian(self) -> None:
        torch.set_default_dtype(torch.float64)
        matrix = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
        report = dfsc.local_identifiability(
            lambda theta: 0.5 * theta @ matrix @ theta,
            torch.tensor([0.2, -0.1]),
            noise_variance=0.25,
        )
        self.assertTrue(report.locally_identifiable)
        self.assertEqual(report.rank, 2)
        self.assertTrue(torch.allclose(report.hessian, matrix))
        self.assertTrue(torch.allclose(report.covariance, 0.25 * torch.linalg.inv(matrix)))


class PreparedKrylovTests(unittest.TestCase):
    def test_prepared_batch_matches_direct_and_preserves_order_gradients(self) -> None:
        torch.set_default_dtype(torch.float64)
        size = 10
        operator = torch.diag(torch.linspace(0.2, 2.0, size))
        u0 = torch.randn(3, size)
        times = torch.linspace(0.0, 0.2, 5)
        alpha = torch.tensor(0.8, requires_grad=True)
        prepared = dfsc.prepare_lanczos_basis(operator, u0, krylov_dimension=size)
        reused = dfsc.apply_prepared_lanczos_basis(prepared, times, alpha)
        direct, _ = dfsc.lanczos_mittag_leffler_action(
            operator, u0, times, alpha.detach(), krylov_dimension=size
        )
        self.assertTrue(torch.allclose(reused, direct, rtol=1e-11, atol=1e-12))
        reused.square().mean().backward()
        self.assertTrue(torch.isfinite(alpha.grad).item())


if __name__ == "__main__":
    unittest.main()
