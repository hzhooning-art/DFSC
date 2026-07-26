import unittest

import torch

import dfsc


class AdaptiveMittagLefflerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def test_adaptive_evaluator_matches_richer_reference_and_backpropagates(self) -> None:
        alpha = torch.tensor(0.72, requires_grad=True)
        z = torch.linspace(0.0, -0.8, 17, requires_grad=True)
        result = dfsc.evaluate_mittag_leffler_adaptive(
            alpha,
            z,
            method="series",
            term_schedule=(12, 24, 48, 80),
            rtol=1e-11,
            atol=1e-13,
            strict=True,
        )
        reference = dfsc.mittag_leffler_e(alpha.detach(), z.detach(), terms=120, method="series")
        self.assertTrue(result.converged)
        self.assertLess(torch.max(torch.abs(result.values.detach() - reference)).item(), 1e-11)
        result.values.sum().backward()
        self.assertTrue(torch.isfinite(alpha.grad))
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_looser_tolerance_does_not_select_more_terms(self) -> None:
        z = -torch.linspace(0.0, 0.9, 32)
        loose = dfsc.evaluate_mittag_leffler_adaptive(
            0.8, z, method="series", term_schedule=(8, 12, 20, 36, 64), rtol=1e-5
        )
        tight = dfsc.evaluate_mittag_leffler_adaptive(
            0.8, z, method="series", term_schedule=(8, 12, 20, 36, 64), rtol=1e-11
        )
        self.assertLessEqual(loose.selected_terms, tight.selected_terms)

    def test_selected_branch_gradient_agrees_with_richer_series(self) -> None:
        z = -torch.linspace(0.0, 2.5, 31)
        for alpha_value in torch.linspace(0.58, 0.92, 9):
            alpha = alpha_value.clone().detach().requires_grad_(True)
            result = dfsc.evaluate_mittag_leffler_adaptive(
                alpha,
                z,
                method="series",
                term_schedule=(8, 12, 18, 26, 38, 56, 84, 126),
                rtol=2e-10,
                atol=2e-12,
                strict=True,
            )
            gradient = torch.autograd.grad(result.values.mean(), alpha)[0]

            alpha_reference = alpha_value.clone().detach().requires_grad_(True)
            reference = dfsc.mittag_leffler_e(
                alpha_reference, z, terms=180, method="series"
            )
            reference_gradient = torch.autograd.grad(reference.mean(), alpha_reference)[0]
            self.assertTrue(torch.isfinite(gradient))
            self.assertLess(
                (torch.abs(gradient - reference_gradient) / torch.abs(reference_gradient).clamp_min(1e-12)).item(),
                2e-7,
            )

    def test_adaptive_reliability_matches_controller_status(self) -> None:
        result = dfsc.evaluate_mittag_leffler_adaptive(
            0.8,
            -torch.linspace(0.0, 0.9, 16),
            method="series",
            term_schedule=(4, 6),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(result.converged, result.evaluation.converged)
        self.assertEqual(result.converged, result.evaluation.reliability.converged)
        self.assertEqual(
            result.evaluation.reliability.error_estimate_kind,
            "successive-truncation-disagreement-not-a-rigorous-bound",
        )

    def test_adaptive_krylov_matches_full_spectral_action_and_has_gradients(self) -> None:
        size = 32
        diagonal = torch.linspace(0.1, 3.0, size)
        operator = torch.diag(diagonal)
        u0 = torch.randn(size, requires_grad=True)
        alpha = torch.tensor(0.78, requires_grad=True)
        beta = torch.tensor(1.8, requires_grad=True)
        times = torch.linspace(0.0, 0.8, 9)
        values, diagnostics = dfsc.adaptive_lanczos_mittag_leffler_action(
            operator,
            u0,
            times,
            alpha,
            beta=beta,
            dimension_schedule=(6, 10, 16, 24, 32),
            rtol=2e-8,
            atol=1e-10,
            strict=True,
        )
        rates = diagonal.pow(beta.detach() / 2.0)
        reference = dfsc.mittag_leffler_e(
            alpha.detach(),
            -times[:, None].pow(alpha.detach()) * rates[None, :],
            terms=180,
            method="hybrid",
        ) * u0.detach()[None, :]
        relative_error = torch.linalg.vector_norm(values.detach() - reference) / torch.linalg.vector_norm(reference)
        self.assertTrue(diagnostics.converged)
        self.assertLess(relative_error.item(), 2e-7)
        values.square().mean().backward()
        self.assertTrue(torch.isfinite(u0.grad).all())
        self.assertTrue(torch.isfinite(alpha.grad))
        self.assertTrue(torch.isfinite(beta.grad))

    def test_problem_interface_reports_adaptive_budget(self) -> None:
        size = 24
        problem = dfsc.OperatorSpectralProblem(
            operator=torch.diag(torch.linspace(0.1, 2.0, size)),
            u0=torch.randn(size),
            times=torch.linspace(0.0, 0.5, 6),
            alpha=torch.tensor(0.8),
        )
        solution = dfsc.solve(
            problem,
            dfsc.MLSLAdaptive(
                dimension_schedule=(6, 12, 18, 24),
                rtol=1e-6,
                atol=1e-9,
                strict=True,
            ),
        )
        self.assertTrue(solution.success)
        self.assertEqual(solution.algorithm, "mlsl-adaptive")
        self.assertTrue(solution.diagnostics["adaptive_converged"])
        self.assertIn("selected_krylov_dimension", solution.diagnostics)


if __name__ == "__main__":
    unittest.main()
