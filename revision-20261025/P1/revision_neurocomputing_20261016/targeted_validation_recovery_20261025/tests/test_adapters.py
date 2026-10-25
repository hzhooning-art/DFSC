from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import adapters as a


class AdapterTests(unittest.TestCase):
    def test_unresolved_u0_fd_preserves_valid_native_gradient(self):
        p={'operator':[[1.]],'u0':[[.8]],'times':[0.,.17,.41],'alpha':.73}
        with patch.object(a,'refined_derivative',side_effect=[
                {'status':'resolved','value':.1},
                {'status':'reference_unresolved','error':'injected cancellation'}]):
            row=a.loss_gradient('selective',p,'physical',None)
        self.assertEqual(row['status'],'returned')
        self.assertTrue(np.isfinite(row['alpha_gradient']))
        self.assertEqual(row['discrete_u0_comparison_status'][0][0],'reference_unresolved')
        self.assertIsNone(row['native_vs_discrete_u0_errors'][0][0])

    def test_pycaputo_unresolved_fd_is_reference_failure_not_method_error(self):
        p={'operator':[[1.]],'u0':[[.8]],'times':[0.,.17,.41],'alpha':.73}
        with patch.object(a,'refined_derivative',return_value={
                'status':'reference_unresolved','error':'injected cancellation'}):
            row=a.loss_gradient('pycaputo',p,'physical',8)
        self.assertEqual(row['status'],'reference_unresolved')

    def test_shared_matrix_and_observations_have_expected_spectrum(self):
        p=a.problem('rotated2',.7)
        np.testing.assert_allclose(np.linalg.eigvalsh(p['operator']),[.3,6],atol=1e-14)
        self.assertEqual(p['times'],[0,.125,.25,.5,.75,1])
        self.assertEqual(p['u0'],[[1.,-.25]])

    def test_frozen_backend_identity_and_repair_forward_graph(self):
        backend=a.load_frozen_backends()
        self.assertTrue(backend['source_snapshot_execution'])
        self.assertEqual(len(backend['module_hashes']),4)
        alpha=torch.nn.Parameter(torch.tensor(.73,dtype=torch.float64))
        u=torch.tensor([[.8]],dtype=torch.float64,requires_grad=True)
        row=a.forward('selective',[[1.]],u,[0.,.17,.41],alpha,None,True)
        self.assertEqual(row['status'],'returned')
        self.assertEqual(tuple(row['values'].shape),(1,3,1))
        grads=a.connected_gradients(row['values'].square().mean(),(alpha,u))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_disconnected_upstream_gradient_cannot_pass(self):
        upstream=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
        detached=torch.nn.Parameter(upstream.detach().clone())
        with self.assertRaisesRegex(RuntimeError,'not.connected'):
            a.connected_gradients(detached.square(),(upstream,))

    def test_source_drift_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'module.py'; src.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                a.verify_source_bytes(src,b'original')

    def test_external_paths_keep_native_or_fd_identity(self):
        p={'operator':[[1.]],'u0':[[.8]],'times':[0.,.17,.41],'alpha':.73}
        native=a.loss_gradient('fdeint',p,'physical',8)
        self.assertEqual(native['gradient_label'],'native_torch_AD')
        self.assertEqual(native['status'],'returned')
        self.assertIn('native_vs_discrete_alpha_error',native)
        transformed=a.loss_gradient('fdeint',p,'transformed',8)
        self.assertEqual(transformed['alpha_path'],'transformed')
        self.assertIn(transformed['status'],('returned','error'))
        if transformed['status']=='error':
            self.assertIn('error',transformed)
        fd=a.loss_gradient('pycaputo',p,'physical',8)
        self.assertEqual(fd['gradient_label'],'explicit_discrete_FD')
        self.assertEqual(fd['status'],'returned')
        self.assertFalse(fd['native_torch_u0'])

    def test_external_solver_uses_zero_initial_time_for_heldout_observations(self):
        row=a.forward('fdeint',[[1.]],[[1.]],[.17,.41],.73,8,False)
        self.assertEqual(row['status'],'returned')
        baseline=a.forward('fdeint',[[1.]],[[1.]],[0.,.17,.41],.73,8,False)
        torch.testing.assert_close(row['values'],baseline['values'][:,1:,:])


if __name__=='__main__': unittest.main()
