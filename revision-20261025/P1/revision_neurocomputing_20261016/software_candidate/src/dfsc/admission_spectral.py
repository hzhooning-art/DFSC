"""Local opt-in homogeneous SPD spectral action with explicit whole-call refusal.

Matches original MLSL spatial-order convention mu=lambda**(beta/2).
Does not modify the frozen release; no fallback and no claimed basis certificate.
"""
from dataclasses import dataclass
import math
import torch
from .admission_scalar import NumericalRefusal, evaluate


class SpectralRefusal(NumericalRefusal):
    def __init__(self,reason,scalar_refused,total_entries):
        super().__init__(reason)
        self.scalar_refused=int(scalar_refused)
        self.total_entries=int(total_entries)


@dataclass
class SpectralResult:
    values: torch.Tensor
    scalar_diagnostics: object
    propagated_value_proxy: torch.Tensor
    floating_point_certified: bool = False


class GuardedSpectral(torch.nn.Module):
    def __init__(self,operator,*,scalar_rtol=1e-9,scalar_atol=1e-12,
                 gradient_atol=1e-10,action_rtol=1e-8,action_atol=1e-10):
        super().__init__()
        if (operator.ndim!=2 or operator.shape[0]==0 or operator.shape[0]!=operator.shape[1] or
            operator.dtype!=torch.float64 or operator.is_complex() or operator.requires_grad or
            not bool(torch.isfinite(operator).all())):
            raise ValueError('Require a fixed real finite square float64 operator; operator gradients unsupported.')
        if not torch.allclose(operator,operator.T,rtol=0.,atol=1e-13):
            raise ValueError('Require a self-adjoint operator.')
        eigenvalues,basis=torch.linalg.eigh(operator)
        if not bool((eigenvalues>0).all()):
            raise ValueError('Require strictly positive eigenvalues.')
        self.register_buffer('eigenvalues',eigenvalues)
        self.register_buffer('basis',basis)
        self.policy=dict(rtol=scalar_rtol,atol=scalar_atol,gradient_atol=gradient_atol)
        self.action_rtol,self.action_atol=action_rtol,action_atol
        if not all(math.isfinite(x) and x>=0 for x in (action_rtol, action_atol)):
            raise ValueError('Action tolerances must be nonnegative.')

    def forward(self,u0,times,alpha,*,beta,diagnostics=False):
        if (u0.dtype!=torch.float64 or times.dtype!=torch.float64 or u0.is_complex() or times.is_complex()
            or u0.device!=self.basis.device or times.device!=u0.device or u0.ndim!=2
            or u0.shape[-1]!=self.basis.shape[0] or u0.shape[0]==0 or times.ndim!=1 or times.numel()==0 or times.requires_grad
            or not bool(torch.isfinite(u0).all()) or not bool(torch.isfinite(times).all()) or bool((times<0).any())):
            raise ValueError('Require finite batch-by-state u0 and fixed nonnegative vector times, on the operator device in float64.')
        alpha=torch.as_tensor(alpha,dtype=u0.dtype,device=u0.device)
        beta=torch.as_tensor(beta,dtype=u0.dtype,device=u0.device)
        if alpha.ndim!=0 or beta.ndim!=0 or not bool(torch.isfinite(alpha)) or not bool(torch.isfinite(beta)):
            raise ValueError('Require finite scalar alpha and beta.')
        if not 0<float(alpha.detach())<=1 or not 0<float(beta.detach())<=2:
            raise ValueError('This homogeneous diffusion adapter requires 0<alpha<=1, 0<spatial beta<=2.')
        safe_t=torch.where(times>0,times,torch.ones_like(times))
        t_alpha=torch.where(times>0,safe_t.pow(alpha),torch.zeros_like(times))
        z=-t_alpha[:,None]*self.eigenvalues.pow(beta/2)[None,:]
        scalar=evaluate(alpha,z,**self.policy)
        refused=int((~scalar.accepted).sum().detach())
        if refused:
            raise SpectralRefusal('At least one modal query failed admission; entire action refused, no fallback.',refused,z.numel())
        coeff=u0@self.basis
        values=(coeff[:,None,:]*scalar.values[None,:,:])@self.basis.T
        with torch.no_grad():
            proxy=(coeff.detach().abs()[:,None,:]*scalar.estimated_errors[...,0][None,:,:])@self.basis.T.abs()
            target=self.action_atol+self.action_rtol*values.detach().abs()
            if not bool(torch.isfinite(values).all()) or not bool((proxy<=target).all()):
                raise SpectralRefusal('Reconstruction amplification failed local output proxy; whole action refused.',0,z.numel())
        result=SpectralResult(values,scalar,proxy)
        return result if diagnostics else result.values
