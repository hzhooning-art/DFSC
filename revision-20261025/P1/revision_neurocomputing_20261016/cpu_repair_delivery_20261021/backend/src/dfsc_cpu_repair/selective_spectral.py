"""Isolated full-SPD CPU action with explicit selected high-precision repair.

No gradients through routing, times or the fixed eigensystem. No result lookups,
silent device transfers, general solver change, cache or global certificate.
"""
from dataclasses import dataclass
import math

import torch

from . import guarded_series as fast
from . import high_precision_cpu as hp


class ActionRefusal(RuntimeError):
    pass


@dataclass
class ActionResult:
    values: torch.Tensor
    fast_entries: int
    repaired_entries: int
    total_entries: int
    output_risk_proxy: torch.Tensor
    rigorous_floating_point_certificate: bool = False


def parameter(value,name):
    if torch.is_tensor(value) and (value.device.type != 'cpu' or value.dtype != torch.float64):
        raise ValueError(name+' must be CPU float64; no silent conversion')
    answer = torch.as_tensor(value,dtype=torch.float64)
    if answer.ndim != 0 or not bool(torch.isfinite(answer)):
        raise ValueError(name+' must be a finite scalar')
    return answer


class SelectiveSpectral(torch.nn.Module):
    def __init__(self,operator,*,mode='selective_repair',scalar_policy=None,
                 precision_options=None,action_rtol=1e-8,action_atol=1e-10):
        super().__init__()
        if mode not in ('selective_repair','all_high_precision','guard_only'):
            raise ValueError('Unknown bounded action mode')
        if (not torch.is_tensor(operator) or operator.ndim != 2 or operator.shape[0] != operator.shape[1]
            or not operator.numel() or operator.dtype != torch.float64 or operator.device.type != 'cpu'
            or operator.requires_grad or not bool(torch.isfinite(operator).all())):
            raise ValueError('Fixed finite nonempty CPU float64 square operator required')
        if not torch.allclose(operator,operator.T,rtol=0.,atol=1e-13):
            raise ValueError('Symmetric operator required')
        lam,q = torch.linalg.eigh(operator)
        if not bool((lam > 0).all()):
            raise ValueError('Strictly positive spectrum required')
        self.register_buffer('eigenvalues',lam)
        self.register_buffer('basis',q)
        self.mode = mode
        self.scalar_policy = dict(scalar_policy or {'rtol':1e-9,'atol':1e-12,'gradient_atol':1e-10})
        self.precision_options = dict(precision_options or {})
        if any(not math.isfinite(v) or v < 0 for v in (action_rtol,action_atol)):
            raise ValueError('Nonnegative finite action tolerances required')
        self.action_rtol,self.action_atol = action_rtol,action_atol

    def forward(self,u0,times,alpha,*,beta):
        if (not torch.is_tensor(u0) or not torch.is_tensor(times) or u0.ndim != 2 or times.ndim != 1
            or u0.shape[-1] != self.basis.shape[0] or not u0.numel() or not times.numel()
            or u0.dtype != torch.float64 or times.dtype != torch.float64
            or u0.device.type != 'cpu' or times.device.type != 'cpu' or self.basis.device.type != 'cpu'
            or times.requires_grad or not bool(torch.isfinite(u0).all())
            or not bool(torch.isfinite(times).all()) or bool((times < 0).any())):
            raise ValueError('Finite CPU float64 batch-by-state u0 and fixed nonnegative times required')
        a,b = parameter(alpha,'alpha'),parameter(beta,'beta')
        if not .45 <= float(a.detach()) <= 1 or not 0 < float(b.detach()) <= 2:
            raise ValueError('Require .45<=alpha<=1 and 0<spatial beta<=2')
        safe_t = torch.where(times > 0,times,torch.ones_like(times))
        powers = torch.where(times > 0,safe_t.pow(a),torch.zeros_like(times))
        z = (-powers[:,None]*self.eigenvalues.pow(b/2)[None,:]).reshape(-1)
        if not bool(torch.isfinite(z).all()) or bool((z < -16).any()):
            raise ActionRefusal('Modal radius exceeds the frozen CPU repair domain; whole action refused')
        if self.mode == 'all_high_precision':
            accepted = torch.zeros_like(z,dtype=torch.bool)
        else:
            # Discard the graph of refused/cancellation-prone entries entirely.
            # Masking an inaccurate differentiable evaluation is not a safe VJP.
            with torch.no_grad():
                accepted = fast.evaluate(a.detach(),z.detach(),**self.scalar_policy).accepted
        if self.mode == 'guard_only' and not bool(accepted.all()):
            raise ActionRefusal('Fast policy refused modal entries; guard-only does not repair')
        fast_index = torch.where(accepted)[0]
        repair_index = torch.where(~accepted)[0]
        values = torch.zeros_like(z)
        proxy = torch.zeros_like(z)
        if fast_index.numel():
            selected = fast.evaluate(a,z[fast_index],**self.scalar_policy)
            if not bool(selected.accepted.all()):
                raise ActionRefusal('Routing decision was not reproducible')
            values = values.index_copy(0,fast_index,selected.values)
            proxy[fast_index] = selected.estimated_errors[:,0]
        if repair_index.numel():
            try:
                repaired = hp.evaluate(a,z[repair_index],**self.precision_options)
            except hp.RepairRefusal as exc:
                raise ActionRefusal('High-precision budget refused; no old-value fallback: '+str(exc)) from exc
            values = values.index_copy(0,repair_index,repaired.values)
            # Disclosed heuristic: refined tail/roundoff plus float64 conversion.
            estimates = []
            for info in repaired.diagnostics:
                last = info['precision_runs'][-1]
                estimates.append(float(last['tail_proxies'][0])+float(last['roundoff_proxies'][0])
                                 +float(info['precision_disagreements'][0]))
            proxy[repair_index] = torch.tensor(estimates,dtype=torch.float64)+64*torch.finfo(torch.float64).eps*repaired.values.detach().abs()
        kernel = values.reshape(times.numel(),self.eigenvalues.numel())
        coeff = u0@self.basis
        output = (coeff[:,None,:]*kernel[None,:,:])@self.basis.T
        with torch.no_grad():
            absolute_action = (coeff.detach().abs()[:,None,:]*kernel.detach().abs()[None,:,:])@self.basis.T.abs()
            risk = (coeff.detach().abs()[:,None,:]*proxy.reshape_as(kernel)[None,:,:])@self.basis.T.abs()
            risk += 64*self.eigenvalues.numel()*torch.finfo(torch.float64).eps*absolute_action
            target = self.action_atol+self.action_rtol*output.detach().abs()
            if not bool(torch.isfinite(output).all()) or not bool((risk <= target).all()):
                raise ActionRefusal('Output risk proxy exceeds target; whole action refused')
        return ActionResult(output,int(fast_index.numel()),int(repair_index.numel()),z.numel(),risk)
