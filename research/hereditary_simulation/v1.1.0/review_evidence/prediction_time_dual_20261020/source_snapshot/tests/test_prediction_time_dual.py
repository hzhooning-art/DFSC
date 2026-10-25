import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from prediction_time_dual import exponential_upper, correct_dual, enclose_dual
from prediction_time_search import Information, strict_witness, log_laplace


class ContinuousDualTests(unittest.TestCase):
    def test_signed_residual_including_a_remote_positive_tail(self):
        # The positive maximum lies far beyond the old rate horizon 128.
        d = np.array([0., .0001, .01, .3])
        a = np.array([.02, 1., -1.8, .2])
        cert = exponential_upper(d, a, residual_tolerance=1e-8)
        points = np.r_[0., np.geomspace(1e-8, 1e7, 10000)]
        actual = a@np.exp(-d[:, None]*points)
        self.assertGreaterEqual(cert['upper'], actual.max())
        self.assertGreater(cert['tail_extensions'], 0)
        self.assertLess(cert['gap'], 1e-8)

    def test_budget_exhaustion_keeps_a_valid_conservative_bound(self):
        d, a = np.array([.001, .1, 1.]), np.array([1., -2., 1.2])
        cert = exponential_upper(d, a, max_cells=2, residual_tolerance=1e-12)
        x = np.r_[0., np.geomspace(1e-6, 1e6, 2000)]
        self.assertGreaterEqual(cert['upper'], (a@np.exp(-d[:, None]*x)).max())
        self.assertEqual(cert['cells'], 2)
        self.assertFalse(cert['tolerance_attained'])

    def test_coincident_zero_rates_preserve_the_constant_tail(self):
        cert = exponential_upper([0.,0.,1.], [.8,-.1,-.5])
        self.assertAlmostEqual(cert['sampled_lower'], .7)
        self.assertGreaterEqual(cert['upper'], .7)
        self.assertLess(cert['upper'], .70000001)

    def test_arbitrary_dual_coefficients_are_repaired_on_the_continuum(self):
        info = Information(1., .1, (0.,.5), (.7,.6), (.15,.15))
        corr = correct_dual(info, 1., .11, [.1,-2.,-.3,.2], [-.2,.8,.3,-.6])
        rates = np.geomspace(1e-8, 1e5, 500)
        for rate in rates:
            for mass in (.55,.6,.7,.8):
                obs = mass*np.exp(-np.array(info.delays)*rate)
                if np.all(np.abs(obs-info.centers) <= info.bands):
                    q = mass*np.exp(-.01*rate)
                    self.assertLessEqual(corr['outer'][0], q)
                    self.assertGreaterEqual(corr['outer'][1], q)
        with self.assertRaises(ValueError):
            correct_dual(info, 0., .11, np.zeros(4), np.zeros(4))

    def test_target_aware_finite_atom_and_repaired_physical_witnesses(self):
        rates, masses = np.array([0., .15, .8]), np.array([.3,.4,.3])
        delays = (0., .5, 1., 2., 4.)
        centers = tuple(float(masses@np.exp(-(.1+d)*rates+log_laplace(rates,1.))) for d in delays)
        info = Information(1., .1, delays, centers, (.01,)*5)
        archive, logs = {}, []
        b = enclose_dual(info, 1., .101, 128, 'test', archive, logs,
                         feasible=strict_witness(info))
        self.assertAlmostEqual(b['fast_rate'], 20000.)
        self.assertLess(b['inner'][0], .82)
        self.assertEqual(b['programs'], 2)
        self.assertEqual(len(logs), 2)
        self.assertGreater(archive['test_il_log_physical_masses'].max(), 709.)
        for side in ('il','ih'):
            nodes, weights = archive[f'test_{side}_physical_nodes'], archive[f'test_{side}_physical_weights']
            self.assertTrue(np.isfinite(nodes).all())
            self.assertTrue((weights > 0).all())
            obs = np.exp(-np.array(delays)[:,None]*nodes)@weights
            self.assertTrue(np.all(np.abs(obs-centers) <= np.array(info.bands)+1e-12))
        truth = float(masses@np.exp(-.101*rates+log_laplace(rates,1.)))
        self.assertLessEqual(b['outer'][0], truth)
        self.assertGreaterEqual(b['outer'][1], truth)


if __name__ == '__main__':
    unittest.main()
