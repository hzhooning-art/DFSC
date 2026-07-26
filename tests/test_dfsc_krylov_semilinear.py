from __future__ import annotations

import unittest

import torch

import dfsc


class DfscKrylovTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(17)

    @staticmethod
    def _operator(size: int) -> torch.Tensor:
        basis, _ = torch.linalg.qr(torch.randn(size, size))
        return basis @ torch.diag(torch.linspace(0.2, 4.0, size)) @ basis.transpose(-1, -2)

    def test_full_krylov_matches_full_eigendecomposition_and_alpha_gradient(self) -> None:
        size = 10
        operator = self._operator(size)
        u0 = torch.randn(size)
        times = torch.linspace(0.0, 0.2, 5)
        alpha_krylov = torch.tensor(0.8, requires_grad=True)
        alpha_full = alpha_krylov.detach().clone().requires_grad_(True)
        krylov_problem = dfsc.OperatorSpectralProblem(operator, u0, times, alpha_krylov)
        full_problem = dfsc.OperatorSpectralProblem(operator, u0, times, alpha_full)

        krylov = dfsc.solve(krylov_problem, dfsc.MLSLKrylov(krylov_dimension=size))
        full = dfsc.solve(full_problem, dfsc.MLSLOperator())
        self.assertTrue(torch.allclose(krylov.values, full.values, rtol=1e-11, atol=1e-12))
        krylov.values.square().sum().backward()
        full.values.square().sum().backward()
        self.assertTrue(torch.allclose(alpha_krylov.grad, alpha_full.grad, rtol=1e-9, atol=1e-10))
        self.assertEqual(krylov.diagnostics["max_effective_krylov_dimension"], size)
        self.assertIn("embedded_relative_disagreement", krylov.diagnostics)
        self.assertFalse(krylov.diagnostics["embedded_disagreement_is_error_bound"])
        self.assertTrue(krylov.stats["matrix_function_action"])

    def test_batched_low_rank_krylov_is_finite(self) -> None:
        size = 18
        values, diagnostics = dfsc.lanczos_mittag_leffler_action(
            self._operator(size),
            torch.randn(3, size),
            torch.linspace(0.0, 0.05, 4),
            torch.tensor(0.9),
            krylov_dimension=7,
        )
        self.assertEqual(tuple(values.shape), (3, 4, size))
        self.assertTrue(torch.isfinite(values).all().item())
        self.assertEqual(diagnostics.requested_dimension, 7)

    def test_auto_selects_krylov_above_policy_limit(self) -> None:
        size = 12
        problem = dfsc.OperatorSpectralProblem(
            self._operator(size),
            torch.randn(size),
            torch.linspace(0.0, 0.02, 3),
            torch.tensor(0.8),
        )
        policy = dfsc.AutoDFSC(dense_eigh_limit=8, krylov_dimension=6)
        decision = dfsc.choose_algorithm(problem, policy)
        solution = dfsc.solve(problem, policy)
        self.assertEqual(decision.name, "mlsl-krylov")
        self.assertEqual(solution.algorithm, "mlsl-krylov")
        self.assertEqual(solution.diagnostics["requested_krylov_dimension"], 6)

    def test_zero_mode_keeps_beta_gradient_finite(self) -> None:
        beta = torch.tensor(1.4, requires_grad=True)
        values, _ = dfsc.lanczos_mittag_leffler_action(
            torch.diag(torch.tensor([0.0, 1.0, 3.0])),
            torch.tensor([1.0, 0.5, -0.25]),
            torch.linspace(0.0, 0.1, 4),
            torch.tensor(0.8),
            beta=beta,
            krylov_dimension=3,
        )
        values.square().sum().backward()
        self.assertTrue(torch.isfinite(beta.grad).item())

    def test_sparse_tensor_matches_dense_action(self) -> None:
        size = 14
        dense = torch.diag(2.0 * torch.ones(size))
        dense = dense + torch.diag(-torch.ones(size - 1), 1) + torch.diag(-torch.ones(size - 1), -1)
        sparse = dense.to_sparse_coo()
        u0 = torch.randn(size)
        times = torch.linspace(0.0, 0.1, 5)
        dense_values, _ = dfsc.lanczos_mittag_leffler_action(
            dense, u0, times, 0.8, krylov_dimension=size
        )
        sparse_values, diagnostics = dfsc.lanczos_mittag_leffler_action(
            sparse, u0, times, 0.8, krylov_dimension=size
        )
        self.assertTrue(torch.allclose(sparse_values, dense_values, rtol=1e-11, atol=1e-12))
        self.assertEqual(diagnostics.representation, "sparse")

    def test_matrix_free_problem_preserves_operator_parameter_gradient(self) -> None:
        size = 10
        base = torch.diag(2.0 * torch.ones(size))
        base = base + torch.diag(-torch.ones(size - 1), 1) + torch.diag(-torch.ones(size - 1), -1)
        scale = torch.tensor(1.2, requires_grad=True)
        operator = dfsc.SelfAdjointLinearOperator(
            size=size,
            matvec=lambda vector: scale * (base @ vector),
            dtype=torch.float64,
            device="cpu",
            name="scaled-laplacian",
        )
        problem = dfsc.LinearOperatorSpectralProblem(
            operator,
            torch.randn(size),
            torch.linspace(0.0, 0.05, 4),
            torch.tensor(0.85),
        )
        solution = dfsc.solve(problem, dfsc.MLSLKrylov(krylov_dimension=size, estimate_error=False))
        self.assertEqual(solution.algorithm, "mlsl-krylov")
        self.assertTrue(solution.stats["matrix_free"])
        self.assertEqual(solution.diagnostics["operator_representation"], "matrix-free")
        solution.final.square().mean().backward()
        self.assertTrue(torch.isfinite(scale.grad).item())

    def test_matrix_free_auto_selection(self) -> None:
        size = 8
        diagonal = torch.linspace(0.5, 2.0, size)
        operator = dfsc.SelfAdjointLinearOperator(
            size,
            lambda vector: diagonal * vector,
            torch.float64,
            "cpu",
        )
        problem = dfsc.LinearOperatorSpectralProblem(
            operator,
            torch.ones(size),
            torch.linspace(0.0, 0.02, 3),
            torch.tensor(0.9),
        )
        decision = dfsc.choose_algorithm(problem)
        solution = dfsc.solve(problem)
        self.assertEqual(decision.name, "mlsl-krylov")
        self.assertTrue(solution.success)


class DfscSemilinearTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    @staticmethod
    def _layer() -> tuple[torch.Tensor, torch.nn.Module]:
        return dfsc.build_mlsl(
            dimension=1,
            boundary="dirichlet",
            num_points=16,
            num_modes=6,
            config=dfsc.MLSLConfig.stable(terms=80),
        )

    def test_zero_nonlinearity_matches_homogeneous_solution(self) -> None:
        x, layer = self._layer()
        u0 = torch.sin(torch.pi * x)
        times = torch.linspace(0.0, 0.02, 5)
        alpha = torch.tensor(0.9)
        problem = dfsc.SemilinearSpectralProblem(
            layer,
            u0,
            times,
            alpha,
            lambda state: torch.zeros_like(state),
        )
        solution = dfsc.solve(problem, dfsc.MLSLPicard(max_iterations=3, quadrature_points=8))
        reference = layer(u0, times, alpha)
        self.assertTrue(torch.allclose(solution.values, reference, rtol=1e-12, atol=1e-12))
        self.assertTrue(solution.success)
        self.assertTrue(solution.diagnostics["picard_converged"])

    def test_weak_cubic_problem_converges_and_preserves_gradients(self) -> None:
        x, layer = self._layer()
        alpha = torch.tensor(0.9, requires_grad=True)
        coefficient = torch.tensor(0.02, requires_grad=True)
        problem = dfsc.SemilinearSpectralProblem(
            layer,
            torch.sin(torch.pi * x),
            torch.linspace(0.0, 0.02, 5),
            alpha,
            lambda state: -coefficient * state.pow(3),
        )
        algorithm = dfsc.MLSLPicard(
            max_iterations=12,
            tolerance=1e-8,
            quadrature_points=12,
            forcing_terms=80,
        )
        solution = dfsc.solve(problem, algorithm)
        self.assertTrue(solution.success)
        self.assertLessEqual(solution.diagnostics["picard_residual"], algorithm.tolerance)
        solution.final.square().mean().backward()
        self.assertTrue(torch.isfinite(alpha.grad).item())
        self.assertTrue(torch.isfinite(coefficient.grad).item())

    def test_nonconvergence_is_reported(self) -> None:
        x, layer = self._layer()
        problem = dfsc.SemilinearSpectralProblem(
            layer,
            torch.sin(torch.pi * x),
            torch.linspace(0.0, 0.02, 4),
            torch.tensor(0.9),
            lambda state: 0.1 * state,
        )
        solution = dfsc.solve(
            problem,
            dfsc.MLSLPicard(max_iterations=1, tolerance=1e-16, quadrature_points=8),
        )
        self.assertEqual(solution.retcode, "maxiters")
        self.assertFalse(solution.diagnostics["picard_converged"])
        self.assertTrue(solution.warnings)


if __name__ == "__main__":
    unittest.main()
