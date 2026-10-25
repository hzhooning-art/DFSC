import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import references as r
from freeze_protocol import build_protocol


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.p=build_protocol()
        self.p['reference_digits']=[65,85]  # unit tests, not science protocol

    def test_exponential_and_origin_do_not_zero_order_derivative(self):
        answer=r.scalar_reference(1.,1.,self.p)
        self.assertEqual(answer['status'],'resolved')
        self.assertAlmostEqual(answer['values'][0],math.exp(-1),13)
        self.assertAlmostEqual(answer['values'][2],math.exp(-1),13)
        self.assertGreater(abs(answer['values'][1]),.01)
        origin=r.scalar_reference(.7,0.,self.p)
        self.assertEqual(origin['values'][:2],[1.,0.])
        self.assertAlmostEqual(origin['values'][2],1/math.gamma(1.7),13)

    def test_precision_disagreement_is_unresolved_not_candidate_failure(self):
        with patch.object(r,'defining_sum',side_effect=[(['1','0','1'],20),(['2','0','1'],21)]):
            answer=r.scalar_reference(1.,1.,self.p)
        self.assertEqual(answer['status'],'reference_unresolved')
        self.assertFalse(answer['precision_agreement'])

    def test_polynomial_refinement_and_nonfinite_failure(self):
        answer=r.refined_derivative(lambda x:x**5,.8,[.01,.005,.0025])
        self.assertEqual(answer['status'],'resolved')
        self.assertAlmostEqual(answer['value'],5*.8**4,10)
        failed=r.refined_derivative(lambda x:float('nan'),1.,[.01,.005,.0025])
        self.assertEqual(failed['status'],'reference_unresolved')
        self.assertIn('nonfinite',failed['error'].lower())

    def test_inverse_laplace_exception_and_zero_transform_are_retained(self):
        def broken(*args,**kwargs):
            raise ArithmeticError('injected inverse failure')
        ans=r.laplace_reference(.7,1.,self.p,inverter=broken)
        self.assertEqual(ans['status'],'reference_unresolved')
        self.assertEqual(len(ans['runs']),2)
        self.assertIn('injected inverse failure',ans['runs'][0]['error'])
        zero=r.laplace_reference(.7,0.,self.p,inverter=broken)
        self.assertEqual(zero['status'],'analytical_origin_not_laplace_success')
        self.assertEqual(zero['runs'][0]['status'],'error')

    def test_independent_initial_state_gradient_matches_coordinate_fd(self):
        operator=[[1.,.2],[.2,2.]]; u0=np.array([[.6,-.4],[.1,.3]])
        ans=r.action_reference(operator,u0.tolist(),[0.,.2,.7],.7,self.p)
        self.assertEqual(ans['status'],'resolved')
        for i in range(2):
            for j in range(2):
                plus=u0.copy(); minus=u0.copy()
                plus[i,j]+=1e-5; minus[i,j]-=1e-5
                fp=np.mean(r.independent_action(operator,plus,[0.,.2,.7],.7)**2)
                fm=np.mean(r.independent_action(operator,minus,[0.,.2,.7],.7)**2)
                self.assertAlmostEqual(ans['u0_gradient'][i][j],(fp-fm)/2e-5,9)


if __name__=='__main__':
    unittest.main()
