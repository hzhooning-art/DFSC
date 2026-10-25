"""Run outside the source tree with the installed candidate, no import-path edits."""
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import dfsc


class InstalledCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.A = torch.diag(torch.tensor([.2, .8, 1.4], dtype=torch.float64))
        self.u = torch.tensor([[1., 2., -1.], [.4, -.5, .9]], dtype=torch.float64)
        self.t = torch.tensor([0., .1, .4], dtype=torch.float64)

    def problem(self, **updates):
        args = dict(operator=self.A, u0=self.u, times=self.t, alpha=.75, beta=1.4)
        args.update(updates)
        return dfsc.OperatorSpectralProblem(**args)

    def test_public_alias_and_algorithm_object(self):
        p = self.problem()
        one = dfsc.solve(p, 'guarded-spd', require_admission=True)
        two = dfsc.solve(p, dfsc.MLSLGuarded())
        torch.testing.assert_close(one.require_accepted(), two.require_accepted(), atol=0, rtol=0)
        self.assertTrue(one.success and one.accepted)
        self.assertFalse(one.reliability.trusted)
        self.assertFalse(one.reliability.rigorous_error_bound)
        self.assertFalse(one.diagnostics['fallback_used'])
        json.dumps(one.summary(), allow_nan=False)

    def test_zero_time_and_output_shape(self):
        s = dfsc.solve(self.problem(), 'guarded-spd')
        self.assertEqual(s.values.shape, (2, 3, 3))
        torch.testing.assert_close(s.values[:, 0], self.u, atol=1e-14, rtol=0)
        # At alpha=1, spatial beta=2, compare against a different matrix-function algorithm.
        exponential = dfsc.solve(self.problem(alpha=1., beta=2.), 'guarded-spd').require_accepted()
        reference = torch.stack([self.u @ torch.matrix_exp(-time*self.A) for time in self.t], dim=1)
        torch.testing.assert_close(exponential, reference, atol=1e-11, rtol=1e-9)

    def test_joint_gradcheck_through_public_solve(self):
        args = (self.u.clone().requires_grad_(), torch.tensor(.75, dtype=torch.float64, requires_grad=True),
                torch.tensor(1.4, dtype=torch.float64, requires_grad=True))
        self.assertTrue(torch.autograd.gradcheck(lambda u, a, b: dfsc.solve(
            self.problem(u0=u, alpha=a, beta=b), 'guarded-spd').require_accepted(), args, atol=1e-7))

    def test_atomic_refusal_no_fallback_update_or_value(self):
        alpha = torch.nn.Parameter(torch.tensor(.45, dtype=torch.float64))
        optimizer = torch.optim.Adam([alpha], lr=.01)
        p = dfsc.OperatorSpectralProblem(torch.diag(torch.tensor([.2, 12.], dtype=torch.float64)),
            torch.ones((2, 2), dtype=torch.float64), torch.tensor([0., .2, 2.], dtype=torch.float64), alpha, beta=2.)
        before = alpha.detach().clone()
        with patch('dfsc.solvers.build_operator_mlsl', side_effect=AssertionError('No fallback allowed')):
            with self.assertRaises(dfsc.SpectralRefusal):
                dfsc.solve(p, 'guarded-spd', require_admission=True).values.square().sum().backward()
                optimizer.step()
        torch.testing.assert_close(alpha.detach(), before, atol=0, rtol=0)
        self.assertIsNone(alpha.grad)
        self.assertEqual(len(optimizer.state), 0)

    def test_legacy_execution_is_not_admitted(self):
        s = dfsc.solve(self.problem(), 'operator')
        self.assertTrue(s.success)
        self.assertFalse(s.accepted)
        self.assertFalse(s.reliability.trusted)
        with self.assertRaises(RuntimeError):
            s.require_accepted()

    def test_require_admission_rejects_legacy_before_solve(self):
        with patch('dfsc.solvers.build_operator_mlsl', side_effect=AssertionError('Must reject before numerical solve')):
            with self.assertRaises(RuntimeError):
                dfsc.solve(self.problem(), 'operator', require_admission=True)
        with self.assertRaises(TypeError):
            dfsc.solve(self.problem(), 'guarded-spd', require_admission='yes')

    def test_invalid_problem_contracts(self):
        invalid = [dict(num_modes=2), dict(operator=self.A.float()), dict(operator=-self.A),
                   dict(operator=self.A.clone().requires_grad_()), dict(alpha=1.2), dict(beta=3.),
                   dict(times=-self.t), dict(times=self.t.clone().requires_grad_()),
                   dict(times=self.t[:0]), dict(u0=self.u[:0]), dict(u0=self.u.float()),
                   dict(operator=torch.empty((0, 0), dtype=torch.float64))]
        for updates in invalid:
            with self.subTest(updates=updates.keys()):
                with self.assertRaises((ValueError, TypeError)):
                    dfsc.solve(self.problem(**updates), 'guarded-spd')

    def test_no_unknown_options_or_wrong_problem_fallback(self):
        with self.assertRaises(TypeError):
            dfsc.solve(object(), 'guarded-spd')
        with self.assertRaises(TypeError):
            dfsc.solve(self.problem(), 'guarded-spd', unrelated_option=True)

    def test_invalid_tolerances(self):
        for value in (-1., math.inf, math.nan):
            for name in ('scalar_rtol', 'scalar_atol', 'gradient_atol', 'action_rtol', 'action_atol'):
                with self.assertRaises(ValueError):
                    dfsc.MLSLGuarded(**{name: value})

    def test_legacy_strict_does_not_accept_agreement_as_accuracy(self):
        with self.assertRaises(RuntimeError):
            dfsc.evaluate_mittag_leffler_adaptive(.75, torch.tensor(-.2, dtype=torch.float64), strict=True)

    def test_scalar_public_gate_and_registry(self):
        one = dfsc.evaluate_with_admission(.75, torch.tensor(-.2, dtype=torch.float64))
        self.assertTrue(torch.isfinite(one.values))
        refused = dfsc.evaluate_with_admission(.45, torch.tensor(-24., dtype=torch.float64))
        with self.assertRaises(dfsc.NumericalRefusal):
            _ = refused.values
        self.assertIn('mlsl-guarded-spd', {r['name'] for r in dfsc.algorithm_registry()})


if __name__ == '__main__':
    unittest.main()
