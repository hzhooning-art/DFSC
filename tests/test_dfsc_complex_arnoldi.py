from __future__ import annotations

import unittest

import torch

import dfsc


class DfscComplexMittagLefflerTests(unittest.TestCase):
    def test_alpha_one_matches_complex_exponential(self) -> None:
        z = torch.tensor([-0.5 + 1.0j, 1.0 - 1.0j], dtype=torch.complex128)
        evaluation = dfsc.evaluate_complex_mittag_leffler(1.0, z)
        self.assertTrue(torch.allclose(evaluation.values, torch.exp(z), rtol=1e-13, atol=1e-13))
        self.assertTrue(evaluation.converged)
        self.assertEqual(evaluation.reliability.level, "high")
        self.assertTrue(evaluation.reliability.trusted)

    def test_alpha_two_matches_cosh_sqrt_identity(self) -> None:
        z = torch.tensor([-1.0 + 0.5j, 0.5 + 0.75j], dtype=torch.complex128)
        actual = dfsc.mittag_leffler_e_complex_series(2.0, z)
        reference = torch.cosh(torch.sqrt(z))
        self.assertTrue(torch.allclose(actual, reference, rtol=1e-13, atol=1e-13))

    def test_complex_alpha_gradient_is_finite(self) -> None:
        alpha = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
        z = torch.tensor([-0.5 + 0.4j, -1.0 - 0.25j], dtype=torch.complex128)
        values = dfsc.mittag_leffler_e_complex_series(alpha, z)
        values.abs().square().mean().backward()
        self.assertTrue(torch.isfinite(alpha.grad).item())

    def test_complex_radius_guard(self) -> None:
        with self.assertRaises(ValueError):
            dfsc.mittag_leffler_e_complex_series(
                0.8, torch.tensor([5.0j], dtype=torch.complex128)
            )


class DfscArnoldiTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def test_nonnormal_real_operator_matches_matrix_exponential(self) -> None:
        operator = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 2.0]])
        u0 = torch.tensor([1.0, -0.5, 0.25])
        times = torch.linspace(0.0, 0.2, 4)
        solution = dfsc.solve(dfsc.GeneralOperatorProblem(operator, u0, times, 1.0))
        reference = torch.stack([torch.matrix_exp(-time * operator) @ u0 for time in times])
        self.assertTrue(torch.allclose(solution.values, reference, rtol=1e-12, atol=1e-12))
        self.assertEqual(solution.algorithm, "mlsl-arnoldi")
        self.assertGreater(solution.diagnostics["max_reduced_nonnormality"], 0.0)

    def test_complex_operator_matches_matrix_exponential(self) -> None:
        operator = torch.tensor(
            [[1.0 + 0.4j, 0.3 - 0.2j], [0.0 + 0.0j, 1.5 - 0.1j]],
            dtype=torch.complex128,
        )
        u0 = torch.tensor([1.0 + 0.2j, -0.5j], dtype=torch.complex128)
        times = torch.linspace(0.0, 0.15, 4)
        values, diagnostics = dfsc.arnoldi_mittag_leffler_action(
            operator, u0, times, 1.0, arnoldi_dimension=2
        )
        reference = torch.stack([torch.matrix_exp(-time * operator) @ u0 for time in times])
        self.assertTrue(torch.allclose(values, reference, rtol=1e-12, atol=1e-12))
        self.assertEqual(diagnostics.representation, "dense")

    def test_matrix_free_parameter_and_alpha_gradients(self) -> None:
        base = torch.tensor(
            [[1.0 + 0.2j, 0.5j], [0.0 + 0.0j, 1.4 - 0.1j]],
            dtype=torch.complex128,
        )
        scale = torch.tensor(1.1, requires_grad=True)
        alpha = torch.tensor(0.85, requires_grad=True)
        operator = dfsc.GeneralLinearOperator(
            2,
            lambda vector: scale * (base @ vector),
            torch.complex128,
            "cpu",
        )
        problem = dfsc.GeneralOperatorProblem(
            operator,
            torch.tensor([1.0 + 0.0j, 0.25 - 0.2j], dtype=torch.complex128),
            torch.linspace(0.0, 0.1, 3),
            alpha,
        )
        solution = dfsc.solve(problem, dfsc.MLSLArnoldi(arnoldi_dimension=2))
        solution.values.abs().square().mean().backward()
        self.assertTrue(torch.isfinite(scale.grad).item())
        self.assertTrue(torch.isfinite(alpha.grad).item())
        self.assertTrue(solution.stats["general_operator"])

    def test_arnoldi_radius_guard(self) -> None:
        with self.assertRaises(ValueError):
            dfsc.arnoldi_mittag_leffler_action(
                20.0 * torch.eye(3),
                torch.ones(3),
                torch.tensor([1.0]),
                0.8,
                arnoldi_dimension=3,
            )


if __name__ == "__main__":
    unittest.main()
