import unittest
from unittest.mock import patch

import torch

from dfsc_cpu_repair.selective_spectral import SelectiveSpectral,ActionRefusal


def tensor(value,grad=False):
    return torch.tensor(value,dtype=torch.float64,requires_grad=grad)


class SelectiveSpectralTests(unittest.TestCase):
    def test_mixed_routing_and_gradients_match_full_precision(self):
        op=torch.diag(tensor([.2,6.]))
        a,b,u=tensor(.75,True),tensor(2.,True),tensor([[1.,.7]],True)
        t=tensor([0.,1.])
        selective=SelectiveSpectral(op)(u,t,a,beta=b)
        full=SelectiveSpectral(op,mode='all_high_precision')(u,t,a,beta=b)
        self.assertGreater(selective.fast_entries,0)
        self.assertGreater(selective.repaired_entries,0)
        torch.testing.assert_close(selective.values,full.values,rtol=1e-8,atol=1e-10)
        g=torch.autograd.grad(selective.values.square().mean(),(a,b,u))
        h=torch.autograd.grad(full.values.square().mean(),(a,b,u))
        for x,y in zip(g,h):
            torch.testing.assert_close(x,y,rtol=2e-6,atol=2e-8)

    def test_rejected_fast_graph_not_used_by_backward(self):
        import dfsc_cpu_repair.selective_spectral as module
        original=module.fast.evaluate
        watched=[]
        def watch(a,z,**kw):
            watched.append((torch.is_grad_enabled(),z.detach().clone()))
            return original(a,z,**kw)
        with patch.object(module.fast,'evaluate',side_effect=watch):
            model=SelectiveSpectral(torch.diag(tensor([.2,6.])))
            a=tensor(.75,True)
            model(tensor([[1.,1.]]),tensor([1.]),a,beta=2.).values.sum().backward()
        self.assertEqual(len(watched),2)
        self.assertFalse(watched[0][0])
        self.assertTrue(watched[1][0])
        self.assertEqual(watched[1][1].numel(),1)
        self.assertTrue(bool(torch.isfinite(a.grad)))

    def test_initial_time_identity(self):
        u=tensor([[1.,-.4]],True)
        a,b=tensor(.75,True),tensor(1.4,True)
        result=SelectiveSpectral(torch.diag(tensor([.2,6.])))(u,tensor([0.]),a,beta=b)
        torch.testing.assert_close(result.values[:,0,:],u,rtol=0,atol=1e-14)
        da,db,du=torch.autograd.grad(result.values.sum(),(a,b,u))
        self.assertEqual(float(da),0.)
        self.assertEqual(float(db),0.)
        torch.testing.assert_close(du,torch.ones_like(u))

    def test_resource_refusal_is_atomic(self):
        model=SelectiveSpectral(torch.diag(tensor([.2,15.5])),precision_options={'maximum_digits':40})
        a=torch.nn.Parameter(tensor(.45))
        before=a.detach().clone()
        opt=torch.optim.Adam([a],lr=.1)
        with self.assertRaises(ActionRefusal):
            model(tensor([[1.,1.]]),tensor([0.,1.]),a,beta=2.).values.sum().backward()
            opt.step()
        self.assertIsNone(a.grad)
        self.assertTrue(torch.equal(a.detach(),before))
        self.assertFalse(opt.state)

    def test_guard_only_does_not_silently_repair(self):
        model=SelectiveSpectral(torch.diag(tensor([.2,6.])),mode='guard_only')
        with self.assertRaises(ActionRefusal):
            model(tensor([[1.,1.]]),tensor([1.]),.75,beta=2.)

    def test_rejects_out_of_scope_contracts(self):
        for op in (tensor([[1.,1.],[0.,1.]]),torch.diag(tensor([0.,1.])),
                   torch.diag(tensor([.2,1.])).float(),torch.diag(tensor([.2,1.],True))):
            with self.assertRaises(ValueError):
                SelectiveSpectral(op)
        model=SelectiveSpectral(torch.diag(tensor([.2,6.])))
        for u,t,a,b in ((tensor([[1.,1.]]),tensor([1.],True),.75,2.),
                        (tensor([[1.,1.]]),tensor([1.]),torch.tensor(.75),2.),
                        (tensor([[1.,1.]]),tensor([]),.75,2.),
                        (tensor([[1.,1.]]),tensor([1.]),.3,2.)):
            with self.assertRaises(ValueError):
                model(u,t,a,beta=b)
        with self.assertRaises(ActionRefusal):
            model(tensor([[1.,1.]]),tensor([10.]),.75,beta=2.)


if __name__=='__main__':
    unittest.main()
