import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experiments"))
from prediction_time_search import (Information, bounded, decision, log_laplace,
                                    ratio, rate_witness, representation, search,
                                    strict_witness)


def fixture(beta=3.):
    rates = np.array([0., .3, 2.])
    mass = np.array([.2, .5, .3])
    delays = (0., .5, 1., 2., 4.)
    anchor = np.exp(-.1*rates+log_laplace(rates, beta))
    y = np.exp(-np.asarray(delays)[:, None]*rates)@(mass*anchor)
    return Information(beta, .1, delays, tuple(y), (.01,)*5)


class TimeSearchTests(unittest.TestCase):
    def test_boundary_and_changed_information_class(self):
        info = fixture()
        self.assertFalse(bounded(info, 0., .1))
        self.assertTrue(bounded(info, 3., .1))
        self.assertTrue(bounded(info, 0., np.nextafter(.1, 1.)))
        self.assertFalse(bounded(info, 3., np.nextafter(.1, 0.)))
        with self.assertRaises(ValueError):
            Information(3, .1, (0.,), (1.,), (.01,), common_mass_cap=1.)

    def test_cell_and_tail_envelopes_against_independent_quadrature(self):
        for beta, gamma, wait in [(1, 0, .12), (3, 1, .103), (1, 1, .1), (2, 3, .1)]:
            rep = representation(fixture(beta), gamma, wait, 32)
            for i, (l, u) in enumerate(zip(rep["edges"][:-1], rep["edges"][1:])):
                samples = np.linspace(l, u, 5) if np.isfinite(u) else l*np.array([1, 1.1, 2, 10])
                for x in samples:
                    # Independent loading-time quadrature on a rescaled domain
                    # avoids underflow in the original high-rate integrals.
                    def lap(order):
                        if order == 0:
                            return 1.
                        if x == 0:
                            return 1/order
                        return quad(lambda z: z**(order-1)*np.exp(-z),
                                    0, min(x, 150.), epsabs=1e-12)[0]/x**order
                    q = np.exp(-(wait-.1)*x)*lap(gamma)/lap(beta)
                    self.assertLessEqual(rep["q_low"][i], q+1e-10)
                    self.assertGreaterEqual(rep["q_high"][i]+1e-10, q)

    def test_witness_is_feasible_and_region_is_valid_at_earlier_waits(self):
        info = fixture()
        witness = strict_witness(info)
        self.assertIsNotNone(witness)
        pair = rate_witness(info, 0, .11, witness)
        rates, weights = witness["nodes"], witness["weighted_masses"]
        original = np.exp(-np.asarray(info.delays)[:, None]*rates)@weights
        changed = original+pair["added_weighted_mass"]*np.exp(-np.asarray(info.delays)*pair["rate"])
        self.assertTrue(np.all(np.abs(changed-info.centers) < info.bands))
        for wait in [.1001, .105, .11]:
            actual = pair["added_weighted_mass"]*ratio(pair["rate"], 3, 0, wait-.1)
            self.assertGreaterEqual(actual, pair["target_increment"]*(1-1e-12))

    def test_no_bisection_of_nonmonotone_diameters_or_early_unresolved(self):
        info = fixture(1)
        # Exact singleton brackets with deliberately nonmonotone radii.
        sizes = {.2: .4, .3: .01, .4: .5, .5: .01}
        def oracle(w, cells):
            z = sizes[w]
            return {"outer": [1., 1.+z], "inner": [1., 1.+z], "cells": cells}
        result = search(info, 1, list(sizes), .02, .05, 0., [32], oracle,
                        strict_witness(info), use_witness=False)
        self.assertEqual([r["state"] for r in result["rows"]],
                         ["insufficient", "sufficient", "insufficient", "sufficient"])
        self.assertEqual(result["first_sufficient_candidate"], .3)
        self.assertFalse(result["continuous_time_minimum_claimed"])
        failed = search(info, 0, [.1, .2], .02, .05, 0., [32], oracle, None)
        self.assertTrue(all(r["state"] == "unresolved" for r in failed["rows"]))

    def test_floor_and_numerical_failure_do_not_become_predictions(self):
        info = fixture()
        def bad(*args):
            raise RuntimeError("intentional LP failure")
        fail = search(info, 0, [1.], .02, .05, 0., [32], bad,
                      strict_witness(info), use_witness=False)
        self.assertEqual(fail["rows"][0]["state"], "unresolved")
        floor = search(info, 0, [1.], .02, .05, .06, [32], bad, strict_witness(info))
        self.assertEqual(floor["rows"][0]["route"], "input_floor")
        self.assertEqual(floor["lp_calls"], 0)
        tight = {"outer": [.4, .8], "inner": [.4001, .7999]}
        self.assertEqual(decision(tight, .01, .05)["state"], "insufficient")

    def test_separate_optimal_estimators_do_not_imply_a_joint_estimator(self):
        exact = {"outer": [.4, .8], "inner": [.4, .8]}
        # Arithmetic midpoint .6 meets absolute .2, harmonic .5333 meets
        # relative 1/3; neither proves that one estimate meets both.
        result = decision(exact, .2, 1/3, margin=0.)
        self.assertEqual(result["state"], "insufficient")
        self.assertIsNone(result["joint_prediction"])

    def test_identical_profile_tail_does_not_lose_its_nonzero_limit(self):
        rep = representation(fixture(1.), 1., .1, 32)
        np.testing.assert_allclose(rep["q_low"], 1., atol=2e-12)
        np.testing.assert_allclose(rep["q_high"], 1., atol=2e-12)


if __name__ == "__main__":
    unittest.main()
