"""Local revision candidate: reject on joint value/gradient numerical-risk tests.

This is NOT a replacement of the archived dfsc package or a certified evaluator.
Exact-arithmetic tail envelopes are distinguished from a floating-point proxy.
The prototype computes the maximum schedule once; selected_terms is truncation,
not the actual work performed. Only first-order derivatives are tested here.
"""
from dataclasses import dataclass
import math
import torch


class NumericalRefusal(RuntimeError):
    pass


@dataclass
class GuardedResult:
    candidate_values: torch.Tensor
    accepted: torch.Tensor
    selected_terms: torch.Tensor
    estimated_errors: torch.Tensor
    estimated_targets: torch.Tensor
    exact_arithmetic_tail_envelopes: torch.Tensor
    roundoff_proxies: torch.Tensor
    actual_terms_computed: int
    rigorous_floating_point_certificate: bool = False

    @property
    def values(self):
        if not bool(self.accepted.all()):
            raise NumericalRefusal('Joint value/gradient checks refused at least one entry; inspect diagnostics, not candidate values.')
        return self.candidate_values


def evaluate(alpha, z, *, rtol=1e-9, atol=1e-12, gradient_rtol=None,
             gradient_atol=1e-10, schedule=(12,24,48,80,120,180,256), roundoff_safety=32):
    """Inspect all entries; values raises if any fails the numerical-risk policy.

    For first omitted index N, b_N=x**N/Gamma(a*N+1), and r=b_(N+1)/b_N<1,
    gamma log-convexity makes subsequent term ratios no larger than r.
    Bounds use geometric moments S0=1/(1-r), S1=N*S0+r*S0**2,
    S2=N**2*S0+2*N*r*S0**2+r*(1+r)*S0**3.
    Value tail <= b_N (alternating); z tail <= b_N*S1/x;
    a tail <= b_N*(a*S2+S1) if a*N+1>=2, using 0<psi(y)<y.
    Envelopes are evaluated in ordinary floating point, NOT rounded outward.
    """
    if not torch.is_tensor(z) or z.dtype != torch.float64 or torch.is_complex(z):
        raise ValueError('Only real float64 z is supported by this local candidate.')
    alpha = torch.as_tensor(alpha, dtype=z.dtype, device=z.device)
    if alpha.ndim != 0 or not bool(torch.isfinite(alpha)) or not 0<float(alpha.detach())<=2:
        raise ValueError('Require one finite scalar order in (0,2].')
    if not bool(torch.isfinite(z).all()) or bool((z>0).any()):
        raise ValueError('Require finite z<=0.')
    gradient_rtol = rtol if gradient_rtol is None else gradient_rtol
    for v in (rtol, atol, gradient_rtol, gradient_atol, roundoff_safety):
        if not math.isfinite(v) or v<0:
            raise ValueError('Tolerances and safety must be finite and nonnegative.')
    if roundoff_safety<1 or not schedule or any(type(n)!=int or n<2 for n in schedule) or list(schedule)!=sorted(set(schedule)):
        raise ValueError('Require safety>=1 and strictly increasing integer schedule>=2.')
    last = schedule[-1]
    k = torch.arange(last+2, device=z.device, dtype=z.dtype)
    # Avoid log(0) and its undefined derivative even in the unused branch.
    zero = z==0
    safe_z = torch.where(zero, -torch.ones_like(z), z)
    log_x = torch.log(-safe_z)
    log_gamma = torch.lgamma(alpha*k+1)
    log_terms = log_x.unsqueeze(-1)*k-log_gamma
    signs = torch.where(k.to(torch.int64)%2==0, 1., -1.)
    terms = torch.exp(log_terms)*signs
    psi = torch.digamma(alpha*k+1)
    a_terms = -terms*k*psi
    z_terms = terms*k/safe_z.unsqueeze(-1)
    values = None
    selected = torch.full_like(z, last, dtype=torch.int64)
    accepted = torch.zeros_like(z, dtype=torch.bool)
    errors = targets = tails = roundoffs = None
    for n in schedule:
        value = terms[...,:n].sum(-1)
        da = a_terms[...,:n].sum(-1)
        dz = z_terms[...,:n].sum(-1)
        value = torch.where(zero, 1+z*torch.exp(-torch.lgamma(alpha+1)), value)
        da = torch.where(zero, torch.zeros_like(da), da)
        dz = torch.where(zero, torch.exp(-torch.lgamma(alpha+1)), dz)
        with torch.no_grad():
            b = terms[...,n].abs()
            ratio = torch.exp(log_x+log_gamma[n]-log_gamma[n+1])
            valid = (ratio<1) & (alpha*n+1>=2)
            safe_r = ratio.clamp(max=1-torch.finfo(z.dtype).eps)
            s0 = 1/(1-safe_r)
            s1 = n*s0+safe_r*s0*s0
            s2 = n*n*s0+2*n*safe_r*s0*s0+safe_r*(1+safe_r)*s0**3
            tail = torch.stack((b, b*(alpha*s2+s1), b*s1/(-safe_z)), dim=-1)
            tail = torch.where(valid.unsqueeze(-1), tail, torch.full_like(tail, float('inf')))
            tail = torch.where(zero.unsqueeze(-1), torch.zeros_like(tail), tail)
            sums = torch.stack((terms[...,:n].abs().sum(-1), a_terms[...,:n].abs().sum(-1),
                                z_terms[...,:n].abs().sum(-1)),dim=-1)
            sums = torch.where(zero.unsqueeze(-1), torch.stack((torch.ones_like(z),torch.zeros_like(z),dz.abs()),-1),sums)
            # Heuristic budget for accumulation plus coefficient/log evaluations.
            scale = 2*n+10+log_gamma[:n].abs().max()+((log_x.unsqueeze(-1)*k[:n]).abs().max(-1).values)
            proxy = roundoff_safety*torch.finfo(z.dtype).eps*scale.unsqueeze(-1)*sums
            error = tail+proxy
            target = torch.stack((atol+rtol*value.abs(), gradient_atol+gradient_rtol*da.abs(),
                                  gradient_atol+gradient_rtol*dz.abs()),dim=-1)
            passes = torch.isfinite(error).all(-1) & torch.isfinite(target).all(-1) & (error<=target).all(-1)
            use = (~accepted) & (passes | (n==last))
        if values is None:
            values = value
            errors, targets, tails, roundoffs = error, target, tail, proxy
        values = torch.where(use, value, values)
        selected = torch.where(use, n, selected)
        errors = torch.where(use.unsqueeze(-1), error, errors)
        targets = torch.where(use.unsqueeze(-1), target, targets)
        tails = torch.where(use.unsqueeze(-1), tail, tails)
        roundoffs = torch.where(use.unsqueeze(-1), proxy, roundoffs)
        accepted = accepted | passes
    return GuardedResult(values,accepted,selected,errors,targets,tails,roundoffs,last+2)
