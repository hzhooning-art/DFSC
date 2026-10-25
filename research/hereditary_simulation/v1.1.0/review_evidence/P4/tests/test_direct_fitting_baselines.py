"""Arithmetic checks before observing new target-error results."""
import inspect
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from direct_fitting_baselines import response_basis, simplex, split_pieces, fit_models, SPEC


class DirectFittingTests(unittest.TestCase):
    def test_step_ramp_closed_form_and_zero(self):
        t=np.array([0.,.2,1.,2.,8.,20.]); rates=np.array([0.,1e-12,.3,4.])
        a,d=response_basis([(t,[(0.,1.,1.)])],rates)
        np.testing.assert_allclose(a[:,1],a[:,0],atol=1e-14)
        expected=np.exp(-.3*np.maximum(t-1,0))*(-np.expm1(-.3*np.minimum(t,1)))/.3
        np.testing.assert_allclose(a[:,3],expected,atol=1e-14)
        self.assertTrue(np.isfinite(d).all())

    def test_rate_jacobian(self):
        t=np.array([.25,1.,3.,8.,16.]); pieces=[(t,[(0.,.5,2.),(2.,2.5,-2.)])]
        rates=np.array([1e-8,.3,3.]); _,d=response_basis(pieces,rates)
        step=1e-5
        plus=response_basis(pieces,rates+step)[0][:,1:]
        minus=response_basis(pieces,rates-step)[0][:,1:]
        np.testing.assert_allclose(d,(plus-minus)/(2*step),atol=2e-7,rtol=1e-5)

    def test_weight_jacobian(self):
        sticks=np.array([.2,.4,.7]); w,j=simplex(sticks)
        self.assertAlmostEqual(w.sum(),1.);self.assertGreaterEqual(w.min(),0)
        for i in range(3):
            step=np.eye(3)[i]*1e-6
            np.testing.assert_allclose(j[:,i],(simplex(sticks+step)[0]-simplex(sticks-step)[0])/2e-6,atol=1e-10)
        np.testing.assert_allclose(simplex(np.array([1.,0.,1.]))[0].sum(),1.)

    def test_validation_preserves_loading_blocks(self):
        pieces=[(np.arange(n),[(0.,1.,1.)]) for n in (16,32,32)]
        train,valid,ti,vi=split_pieces(pieces)
        self.assertEqual([len(t) for t,s in train],[12,24,24])
        self.assertEqual([len(t) for t,s in valid],[4,8,8])
        self.assertEqual(len(set(ti)&set(vi)),0)
        self.assertEqual(set(ti)|set(vi),set(range(80)))

    def test_no_truth_or_identity_in_fit_api(self):
        self.assertEqual(list(inspect.signature(fit_models).parameters),['pieces','y','protocol'])
        p=json.loads(SPEC.read_text())
        self.assertEqual(p['prony_rate_bounds'],[0.,4.])
        self.assertEqual(p['full_case_count'],256)
        self.assertEqual(p['absolute_point_tolerances'],{'G20':.1,'heldout_sigma20':.05})


if __name__=='__main__':unittest.main()
