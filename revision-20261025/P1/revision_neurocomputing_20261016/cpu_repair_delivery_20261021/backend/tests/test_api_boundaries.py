import json
import unittest
import torch
import dfsc_cpu_repair as repair


def t(x,grad=False):
    return torch.tensor(x,dtype=torch.float64,requires_grad=grad)


class InstalledAPIBoundaries(unittest.TestCase):
    def test_namespace_metadata_and_no_implicit_registration(self):
        info=repair.backend_info()
        json.dumps(info,allow_nan=False)
        self.assertFalse(info['automatic_dfsc_registration'])
        self.assertFalse(info['rigorous_floating_point_certificate'])
        self.assertEqual(repair.__version__,'0.1.0.dev0')
        self.assertEqual(repair.SelectiveSpectral.__module__,'dfsc_cpu_repair.selective_spectral')

    def test_joint_public_action_gradcheck(self):
        model=repair.SelectiveSpectral(torch.diag(t([.2,4.6])))
        args=(t(.75,True),t(1.4,True),t([[1.,.7]],True))
        self.assertTrue(torch.autograd.gradcheck(
            lambda a,b,u:model(u,t([.2,1.]),a,beta=b).values,args,atol=1e-7))

    def test_integer_order_matrix_exponential(self):
        operator=t([[1.,.2],[.2,3.]])
        u=t([[1.,.7],[-.4,1.]])
        times=t([0.,.2,1.])
        result=repair.SelectiveSpectral(operator)(u,times,1.,beta=2.)
        reference=torch.stack([u@torch.matrix_exp(-x*operator) for x in times],dim=1)
        torch.testing.assert_close(result.values,reference,rtol=1e-8,atol=1e-10)

    def test_scalar_second_derivative_not_silently_provided(self):
        a=t(.75,True)
        first=torch.autograd.grad(repair.evaluate_scalar(a,t(-1.)).values,a,create_graph=True)[0]
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(first,a)

    def test_scalar_alpha_one_has_nonzero_alpha_derivative(self):
        a,z=t(1.,True),t(-1.,True)
        r=repair.evaluate_scalar(a,z)
        da,dz=torch.autograd.grad(r.values,(a,z))
        self.assertGreater(abs(float(da)),1e-3)
        torch.testing.assert_close(r.values,z.exp(),rtol=1e-12,atol=1e-13)
        torch.testing.assert_close(dz,z.exp(),rtol=1e-12,atol=1e-13)

    def test_invalid_modes_tolerances_and_scalar_contract(self):
        A=torch.diag(t([.2,1.]))
        for kwargs in ({'mode':'automatic'},{'action_atol':float('nan')},{'action_rtol':-1.}):
            with self.assertRaises(ValueError):
                repair.SelectiveSpectral(A,**kwargs)
        for a,z in ((.3,t(-1.)),(.75,t(-17.)),(.75,torch.tensor(-1.))):
            with self.assertRaises(ValueError):
                repair.evaluate_scalar(a,z)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
