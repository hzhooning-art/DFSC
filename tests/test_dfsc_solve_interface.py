from __future__ import annotations

import unittest

import torch

import dfsc


class DfscSolveInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def test_algorithm_registry_exposes_specialized_algorithms(self) -> None:
        names = {row["name"] for row in dfsc.algorithm_registry()}
        self.assertIn("auto-dfsc", names)
        self.assertIn("mlsl-direct", names)
        self.assertIn("mlsl-stable", names)
        self.assertIn("mlsl-operator", names)
        self.assertIn("mlsl-graph", names)
        self.assertIn("mlsl-krylov", names)
        self.assertIn("mlsl-adaptive", names)
        self.assertIn("mlsl-arnoldi", names)
        self.assertIn("mlsl-picard", names)
        self.assertIn("caputo-l1-history-direct", names)
        self.assertIn("caputo-l1-history-fft", names)

    def test_fractional_spectral_problem_solve(self) -> None:
        x, layer = dfsc.build_mlsl(
            dimension=1,
            boundary="dirichlet",
            num_points=32,
            num_modes=8,
            config=dfsc.MLSLConfig.stable(terms=50),
        )
        problem = dfsc.FractionalSpectralProblem(
            layer=layer,
            u0=torch.sin(torch.pi * x),
            times=torch.linspace(0.0, 0.02, 4),
            alpha=torch.tensor(0.9),
            beta=torch.tensor(1.4),
        )
        solution = dfsc.solve(problem, dfsc.MLSLStable(terms=50))
        self.assertEqual(solution.problem_type, "FractionalSpectralProblem")
        self.assertEqual(tuple(solution.u.shape), (4, 32))
        self.assertTrue(torch.isfinite(solution.final).all().item())
        self.assertTrue(solution.success)
        self.assertEqual(solution.diagnostics["evaluator_method"], "hybrid")
        self.assertEqual(solution.quality, "moderate")
        self.assertFalse(solution.reliability.rigorous_error_bound)
        self.assertEqual(solution.summary()["retcode"], "success")

    def test_explicit_algorithm_reconfigures_prebuilt_layer(self) -> None:
        x, layer = dfsc.build_mlsl(
            dimension=1,
            boundary="dirichlet",
            num_points=32,
            num_modes=8,
            config=dfsc.MLSLConfig.stable(terms=60),
        )
        problem = dfsc.FractionalSpectralProblem(
            layer=layer,
            u0=torch.sin(torch.pi * x),
            times=torch.linspace(0.0, 0.002, 3),
            alpha=torch.tensor(1.1),
            beta=torch.tensor(1.4),
        )
        solution = dfsc.solve(problem, dfsc.MLSLDirect(config=dfsc.MLSLConfig(terms=70)))
        self.assertEqual(solution.algorithm, "mlsl-direct")
        self.assertEqual(solution.diagnostics["evaluator_method"], "series")
        self.assertEqual(solution.diagnostics["terms"], 70)

    def test_auto_selection_reports_reason_and_regime(self) -> None:
        x, layer = dfsc.build_mlsl(
            dimension=1,
            boundary="dirichlet",
            num_points=32,
            num_modes=8,
            config=dfsc.MLSLConfig(),
        )
        small = dfsc.FractionalSpectralProblem(
            layer=layer,
            u0=torch.sin(torch.pi * x),
            times=torch.linspace(0.0, 1e-5, 3),
            alpha=torch.tensor(1.0),
        )
        large = dfsc.FractionalSpectralProblem(
            layer=layer,
            u0=torch.sin(torch.pi * x),
            times=torch.linspace(0.0, 0.2, 3),
            alpha=torch.tensor(0.7),
        )
        direct = dfsc.choose_algorithm(small)
        stable = dfsc.choose_algorithm(large)
        self.assertEqual(direct.name, "mlsl-direct")
        self.assertEqual(stable.name, "mlsl-stable")
        self.assertIn("max_abs_argument", stable.diagnostics)
        solution = dfsc.solve(large)
        self.assertEqual(solution.algorithm, "mlsl-stable")
        self.assertIn("reason", solution.diagnostics)

    def test_solution_final_uses_time_axis_for_batches(self) -> None:
        times = torch.linspace(0.0, 1.0, 4)
        values = torch.arange(2 * 4 * 3, dtype=torch.float64).reshape(2, 4, 3)
        solution = dfsc.Solution(values, times, "test", "test")
        self.assertTrue(torch.equal(solution.final, values[:, -1, :]))

    def test_graph_problem_solve(self) -> None:
        n = 10
        adjacency = torch.zeros(n, n)
        for i in range(n - 1):
            adjacency[i, i + 1] = 1.0
            adjacency[i + 1, i] = 1.0
        problem = dfsc.GraphSpectralProblem(
            adjacency=adjacency,
            u0=torch.linspace(0.0, 1.0, n),
            times=torch.linspace(0.0, 0.01, 3),
            alpha=torch.tensor(0.8),
            beta=torch.tensor(1.2),
            num_modes=6,
        )
        solution = dfsc.solve(problem)
        self.assertEqual(solution.algorithm, "mlsl-graph")
        self.assertEqual(tuple(solution.values.shape), (3, n))
        self.assertTrue(torch.isfinite(solution.values).all().item())
        self.assertTrue(solution.success)

    def test_operator_problem_rejects_wrong_algorithm(self) -> None:
        operator = torch.eye(4)
        problem = dfsc.OperatorSpectralProblem(
            operator=operator,
            u0=torch.ones(4),
            times=torch.linspace(0.0, 0.01, 2),
            alpha=torch.tensor(0.9),
        )
        with self.assertRaises(TypeError):
            dfsc.solve(problem, dfsc.MLSLGraph())

    def test_generalized_operator_problem(self) -> None:
        stiffness = torch.diag(torch.tensor([0.0, 2.0, 6.0]))
        mass = torch.diag(torch.tensor([1.0, 1.5, 2.0]))
        problem = dfsc.GeneralizedOperatorSpectralProblem(
            stiffness=stiffness,
            mass=mass,
            u0=torch.tensor([1.0, 0.5, -0.25]),
            times=torch.linspace(0.0, 0.02, 3),
            alpha=torch.tensor(0.9),
        )
        solution = dfsc.solve(problem)
        self.assertEqual(solution.algorithm, "mlsl-generalized-operator")
        self.assertEqual(solution.problem_type, "GeneralizedOperatorSpectralProblem")
        self.assertTrue(solution.success)
        self.assertTrue(solution.stats["mass_projection"])

    def test_caputo_l1_fallback_matches_scalar_baseline(self) -> None:
        alpha = torch.tensor(0.7, requires_grad=True)
        problem = dfsc.CaputoL1Problem(
            operator=torch.tensor([[2.0]]),
            u0=torch.tensor([1.0]),
            alpha=alpha,
            final_time=0.2,
            num_steps=12,
        )
        solution = dfsc.solve(problem)
        reference = dfsc.l1_caputo_relaxation(
            torch.tensor(1.0),
            alpha=alpha,
            mu=torch.tensor(2.0),
            final_time=0.2,
            num_steps=12,
        )
        self.assertEqual(solution.algorithm, "caputo-l1")
        self.assertTrue(torch.allclose(solution.values[:, 0], reference, atol=1e-12))
        self.assertTrue(solution.stats["history_dependent"])
        solution.final.square().sum().backward()
        self.assertTrue(torch.isfinite(alpha.grad).item())


if __name__ == "__main__":
    unittest.main()
