import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experiments"))
from prediction_time_transport import transport_bracket, transferred_decision, adaptive_search
from prediction_time_search import Information
from prediction_time_curvature import curvature_bound, joint_cell_test, quadratic_max


class TransportTests(unittest.TestCase):
    def test_containment_for_enumerated_physical_spectra_at_interior_times(self):
        # A finite positive physical class supplies an independent exact example.
        rates = np.array([0., .2, 1., 4.])
        spectra = np.array([[.2, .5, .3, 0.], [.3, .1, .1, .8], [.1, .6, .2, .1]])
        def exact(t):
            values = spectra@np.exp(-t*rates)
            return {"outer": [values.min(), values.max()],
                    "inner": [values.min(), values.max()]}
        s, t = .2, 1.3
        for k in np.linspace(s, t, 31):
            q = spectra@np.exp(-k*rates)
            for at in [None, k]:
                bracket = transport_bracket(exact(s), exact(t), s, t, at)
                self.assertLessEqual(bracket["outer"][0], q.min()+1e-14)
                self.assertGreaterEqual(bracket["outer"][1], q.max()-1e-14)
                if bracket["inner"] is not None:
                    self.assertGreaterEqual(bracket["inner"][0], q.min()-1e-14)
                    self.assertLessEqual(bracket["inner"][1], q.max()+1e-14)

    def test_empty_inner_overlap_never_becomes_a_rejection(self):
        left = {"outer": [2., 3.], "inner": [2.1, 2.9]}
        right = {"outer": [.3, .8], "inner": [.4, .7]}
        b = transport_bracket(left, right, 1., 2.)
        self.assertIsNone(b["inner"])
        self.assertEqual(transferred_decision(b, .01, .05, 0.)["state"], "unresolved")
        self.assertEqual(transferred_decision(b, 3., .95, 0.)["state"], "sufficient")

    def test_secant_is_tighter_than_left_endpoint_but_never_extrapolates(self):
        left = {"outer": [.4, .8], "inner": [.45, .75]}
        right = {"outer": [.3, .6], "inner": [.35, .55]}
        box = transport_bracket(left, right, 1., 3.)
        mid = transport_bracket(left, right, 1., 3., 2.)
        self.assertAlmostEqual(mid["outer"][1], .7)
        self.assertLess(mid["outer"][1], box["outer"][1])
        with self.assertRaises(ValueError):
            transport_bracket(left, right, 1., 3., 4.)

    def test_search_reuses_region_and_retains_numerical_failures(self):
        info = Information(1., 0., (0.,), (1.,), (.01,))
        called = []
        def oracle(w, cells):
            called.append(w)
            return {"outer": [.99-.001*w, 1.01-.001*w], "inner": [.995-.001*w, 1.005-.001*w],
                    "programs": 4, "solver_calls": 4, "cells": cells,
                    "requested_cells": cells}
        feasible = {"alpha": .001}
        result = adaptive_search(info, 1., list(np.linspace(.1, 1., 17)), .025, .05, 0., [128], oracle, feasible)
        self.assertEqual(len(called), 2)
        self.assertEqual(result["lp_calls"], 8)
        self.assertTrue(all(r["state"] == "sufficient" for r in result["rows"]))
        self.assertTrue(result["time_cells"])
        def failure(*args):
            raise RuntimeError("intentional")
        failed = adaptive_search(info, 1., [.1, .2], .025, .05, 0., [128], failure, feasible)
        self.assertTrue(all(r["state"] == "unresolved" for r in failed["rows"]))

    def test_no_strict_witness_disables_both_exclusion_and_computation(self):
        info = Information(1., .1, (0.,), (1.,), (.01,))
        def forbidden(*args):
            raise AssertionError("oracle should not be called")
        r = adaptive_search(info, 0., [0., .1, .2], .025, .05, 0., [128], forbidden, None)
        self.assertTrue(all(v["state"] == "unresolved" for v in r["rows"]))

    def test_whole_halfline_curvature_bound(self):
        from prediction_time_search import ratio
        info = Information(3., .1, (0.,), (.4,), (.01,))
        self.assertIsNone(curvature_bound(info, 0., .1))
        for gamma in [0., 1.]:
            for start in [.101, .3, 1.]:
                bound = curvature_bound(info, gamma, start)
                x = np.r_[0., np.geomspace(1e-5, 1e6, 5000)]
                observed = .41*x*x*ratio(x, 3., gamma, start-.1)
                self.assertGreaterEqual(bound*(1+1e-12), observed.max())

    def test_quadratic_max_and_joint_cell_predictor(self):
        x = np.linspace(0., 2., 10001)
        actual = .1+(.5-.1)*x/2+3*x*(2-x)/2
        self.assertGreaterEqual(quadratic_max(.1, .5, 2., 3.)+1e-12, actual.max())
        left = {"outer": [.99, 1.01], "inner": [.995, 1.005]}
        right = {"outer": [.97, .99], "inner": [.975, .985]}
        cert = joint_cell_test(left, right, 1., 1.1, .2, .025, .05, 0.)
        self.assertIsNotNone(cert)
        for t in np.linspace(1., 1.1, 51):
            w = (1.1-t)/.1
            l = w*.99+(1-w)*.97-.2*(t-1)*(1.1-t)/2
            u = w*1.01+(1-w)*.99
            lo, hi = max(u-.025, .95*u), min(l+.025, 1.05*l)
            self.assertLessEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()
