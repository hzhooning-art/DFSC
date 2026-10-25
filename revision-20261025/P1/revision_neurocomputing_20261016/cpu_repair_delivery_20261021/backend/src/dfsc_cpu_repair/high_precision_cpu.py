"""Separate CPU feasibility prototype: actual high-precision values and first VJP.

No dfsc import, old-result lookup or external-reference call. This is deliberately
slow, not a public API, GPU kernel, double-backward implementation or certificate.
"""
from dataclasses import dataclass
import math

import mpmath
import torch
from torch.autograd.function import once_differentiable


class RepairRefusal(RuntimeError):
    pass


def _summation(alpha, radius, digits, *, max_terms, internal_atol, internal_rtol, roundoff_safety):
    ctx = mpmath.mp.clone()
    ctx.dps = digits
    # Round-trip decimal conversion preserves the input float, rather than a nominal decimal.
    a, x = ctx.mpf(alpha), ctx.mpf(radius)
    atol, rtol = ctx.mpf(internal_atol), ctx.mpf(internal_rtol)
    if x == 0:
        triple = [ctx.mpf(1), ctx.mpf(0), ctx.rgamma(a+1)]
        return triple, {'digits': digits, 'terms': 1, 'tail_proxies': ['0']*3,
                        'roundoff_proxies': [ctx.nstr(ctx.eps*abs(v), 10) for v in triple]}
    logx = ctx.log(x)
    values = [ctx.mpf(1), ctx.mpf(0), ctx.mpf(0)]
    absolute_sums = [ctx.mpf(1), ctx.mpf(0), ctx.mpf(0)]
    for k in range(1, max_terms):
        y = a*k+1
        term = ctx.exp(k*logx-ctx.loggamma(y))
        if k % 2:
            term = -term
        contributions = [term, -k*ctx.digamma(y)*term, -k*term/x]
        for j in range(3):
            values[j] += contributions[j]
            absolute_sums[j] += abs(contributions[j])
        n = k+1
        if a*n+1 < 2:
            continue
        b = ctx.exp(n*logx-ctx.loggamma(a*n+1))
        ratio = ctx.exp(logx+ctx.loggamma(a*n+1)-ctx.loggamma(a*(n+1)+1))
        if ratio >= 1:
            continue
        s0 = 1/(1-ratio)
        s1 = n*s0+ratio*s0**2
        s2 = n*n*s0+2*n*ratio*s0**2+ratio*(1+ratio)*s0**3
        # Absolute geometric envelopes; psi(y) < y for y>=2 bounds the alpha tail.
        tails = [b*s0, b*(a*s2+s1), b*s1/x]
        # MP arithmetic/coefficient roundoff proxy, not an outward-rounded bound.
        risk = [roundoff_safety*ctx.eps*(n+20)**2*v for v in absolute_sums]
        targets = [atol+rtol*abs(v) for v in values]
        if all(ctx.isfinite(v) for v in values) and all(t+r <= target for t,r,target in zip(tails,risk,targets)):
            return values, {'digits': digits, 'terms': n, 'tail_proxies': [ctx.nstr(v, 16) for v in tails],
                            'roundoff_proxies': [ctx.nstr(v, 16) for v in risk]}
    raise RepairRefusal(f'Precision {digits} exhausted max_terms={max_terms}')


def scalar(alpha, radius, *, minimum_digits=40, cancellation_safety_digits=32,
           maximum_digits=300, verification_extra_digits=32, max_terms=6000,
           internal_atol='1e-22', internal_rtol='1e-20', roundoff_safety=64):
    if not math.isfinite(alpha) or not .45 <= alpha <= 1.95 or not math.isfinite(radius) or not 0 <= radius <= 16:
        raise ValueError('Feasibility scope: .45<=alpha<=1.95, -16<=z<=0, ML beta=1.')
    if minimum_digits < 20 or verification_extra_digits < 10 or maximum_digits < minimum_digits or max_terms < 2:
        raise ValueError('Invalid precision/work budget')
    # Conservative initial cancellation scale, not a proof; both MP runs still check their proxies.
    predicted_digits = 0 if radius == 0 else radius**(1/alpha)/math.log(10)
    digits = max(minimum_digits, math.ceil(predicted_digits)+cancellation_safety_digits)
    if digits+verification_extra_digits > maximum_digits:
        raise RepairRefusal('Declared precision budget cannot fund independent precision refinement')
    low, first = _summation(alpha, radius, digits, max_terms=max_terms,
        internal_atol=internal_atol, internal_rtol=internal_rtol, roundoff_safety=roundoff_safety)
    high, second = _summation(alpha, radius, digits+verification_extra_digits, max_terms=max_terms,
        internal_atol=internal_atol, internal_rtol=internal_rtol, roundoff_safety=roundoff_safety)
    ctx = mpmath.mp.clone()
    ctx.dps = digits+verification_extra_digits+10
    differences = [abs(ctx.mpf(x)-ctx.mpf(y)) for x,y in zip(low, high)]
    if not all(d <= ctx.mpf(internal_atol)+ctx.mpf(internal_rtol)*abs(ctx.mpf(y)) for d,y in zip(differences,high)):
        raise RepairRefusal('Precision-pair disagreement exceeds declared internal targets')
    converted = [float(v) for v in high]
    if not all(math.isfinite(v) for v in converted):
        raise RepairRefusal('Float64 conversion is nonfinite')
    return converted, {'precision_runs': [first, second],
                       'precision_disagreements': [ctx.nstr(v, 16) for v in differences],
                       'float64_values_value_alpha_z': converted,
                       'rigorous_floating_point_certificate': False}


class _FirstOrderVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, z, value, da, dz):
        ctx.alpha_shape = alpha.shape
        ctx.save_for_backward(da, dz)
        return value.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        da, dz = ctx.saved_tensors
        return (grad_output*da).sum().reshape(ctx.alpha_shape), grad_output*dz, None, None, None


@dataclass
class RepairResult:
    values: torch.Tensor
    diagnostics: list
    rigorous_floating_point_certificate: bool = False
    backend: str = 'separate-cpu-arbitrary-precision-series'


def evaluate(alpha, z, **precision_options):
    if not torch.is_tensor(z) or z.dtype != torch.float64 or z.device.type != 'cpu' or z.numel() == 0:
        raise ValueError('Nonempty CPU float64 tensor required; no silent GPU transfer.')
    a = torch.as_tensor(alpha, dtype=torch.float64, device=z.device)
    if a.numel() != 1 or (torch.is_tensor(alpha) and (alpha.dtype != torch.float64 or alpha.device.type != 'cpu')):
        raise ValueError('A scalar CPU float64 alpha is required.')
    triples, diagnostics = [], []
    for item in z.detach().reshape(-1):
        values, info = scalar(float(a.detach()), -float(item), **precision_options)
        triples.append(values)
        diagnostics.append(info)
    # Nothing is exposed until the complete batch has passed, including later entries.
    arrays = torch.tensor(triples, dtype=torch.float64)
    value, da, dz = [arrays[:,j].reshape(z.shape) for j in range(3)]
    return RepairResult(_FirstOrderVJP.apply(a, z, value, da, dz), diagnostics)
