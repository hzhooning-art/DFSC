"""Extensions for oscillatory modes, partial sharing, and group calibration."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import numpy as np
from scipy.optimize import least_squares
from .core import CurveRecord, _stack

@dataclass(frozen=True)
class OscillatoryBounds:
    decay: tuple[float,float]=(1/300,2.0)
    damping: tuple[float,float]=(1/300,2.0)
    frequency: tuple[float,float]=(0.02,20.0)

def generalized_design(time, *, decay_rates=(), oscillatory_pairs=()):
    """Real basis for stable real poles and conjugate-complex pole pairs."""
    t=np.asarray(time,float); rates=np.asarray(decay_rates,float); pairs=np.asarray(oscillatory_pairs,float)
    if t.ndim!=1 or not np.isfinite(t).all(): raise ValueError('time must be finite and one-dimensional')
    if rates.size and (rates.ndim!=1 or np.any(rates<=0) or not np.isfinite(rates).all()): raise ValueError('decay rates must be finite and positive')
    if pairs.size:
        pairs=np.atleast_2d(pairs)
        if pairs.shape[1]!=2 or np.any(pairs<=0) or not np.isfinite(pairs).all(): raise ValueError('pairs require positive damping and frequency')
    cols=[np.ones_like(t)]+[np.exp(-r*t) for r in rates]
    for d,w in pairs:
        e=np.exp(-d*t); cols.extend((e*np.cos(w*t),e*np.sin(w*t)))
    return np.column_stack(cols)

def _predict(time,values,rates,pairs):
    x=generalized_design(time,decay_rates=rates,oscillatory_pairs=pairs)
    coef=np.linalg.lstsq(x,values.T,rcond=None)[0]
    return coef,(x@coef).T

def fit_oscillatory_shared(curves: Iterable[CurveRecord], *, real_rank=0, oscillatory_rank=1, starts=8, bounds=OscillatoryBounds()):
    """Fit shared stable real modes and shared conjugate-pole pairs."""
    if real_rank<0 or oscillatory_rank<0 or real_rank+oscillatory_rank==0: raise ValueError('at least one mode is required')
    rows=list(curves); time,values=_stack(rows)
    lo=np.r_[np.full(real_rank,np.log(bounds.decay[0])),np.full(oscillatory_rank,np.log(bounds.damping[0])),np.full(oscillatory_rank,np.log(bounds.frequency[0]))]
    hi=np.r_[np.full(real_rank,np.log(bounds.decay[1])),np.full(oscillatory_rank,np.log(bounds.damping[1])),np.full(oscillatory_rank,np.log(bounds.frequency[1]))]
    def unpack(p):
        rates=np.sort(np.exp(p[:real_rank])); c=real_rank
        d=np.exp(p[c:c+oscillatory_rank]); c+=oscillatory_rank; w=np.exp(p[c:c+oscillatory_rank])
        pairs=np.column_stack((d,w)) if oscillatory_rank else np.empty((0,2))
        return rates,pairs[np.argsort(pairs[:,1])] if len(pairs) else pairs
    def residual(p):
        rates,pairs=unpack(p); return (_predict(time,values,rates,pairs)[1]-values).ravel()
    best=None
    for s in range(starts):
        out=least_squares(residual,lo+(s+1)/(starts+1)*(hi-lo),bounds=(lo,hi),max_nfev=600,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        rates,pairs=unpack(out.x); e=residual(out.x)
        cand={'decay_rates':rates.tolist(),'oscillatory_pairs':pairs.tolist(),'sse':float(e@e),'success':bool(out.success)}
        if best is None or cand['sse']<best['sse']: best=cand
    n=values.size; k=real_rank+2*oscillatory_rank+len(rows)*(1+real_rank+2*oscillatory_rank)
    best.update({'bic':float(n*np.log(max(best['sse']/n,1e-300))+k*np.log(n)),'n_curves':len(rows),'n_observations':int(n),'real_rank':real_rank,'oscillatory_rank':oscillatory_rank,'stability_contract':'positive damping; real conjugate-pair representation'})
    return best

def fit_partially_shared(curves: Iterable[CurveRecord], rank, *, shrinkage=1.0, starts=4, rate_bounds=(1/300,2.0)):
    """Fit group rates shrunk toward a shared geometric centre."""
    if rank<1 or shrinkage<0: raise ValueError('rank must be positive and shrinkage nonnegative')
    rows=list(curves); time,_=_stack(rows); groups=sorted({r.group for r in rows})
    if len(groups)<2: raise ValueError('partial sharing requires at least two groups')
    ys={g:np.stack([np.asarray(r.value,float) for r in rows if r.group==g]) for g in groups}
    low,high=np.log(rate_bounds); free=(len(groups)-1)*rank
    def unpack(p):
        centre=np.sort(p[:rank]); q=p[rank:].reshape(len(groups)-1,rank); dev=np.vstack((q,-q.sum(0,keepdims=True)))
        return np.exp(centre),{g:np.exp(centre+dev[i]) for i,g in enumerate(groups)},dev
    def blocks(p,penalty=True):
        _,rates,dev=unpack(p); out=[]
        for g in groups:
            x=generalized_design(time,decay_rates=rates[g]); coef=np.linalg.lstsq(x,ys[g].T,rcond=None)[0]; out.append(((x@coef).T-ys[g]).ravel())
        if penalty and shrinkage: out.append(np.sqrt(shrinkage)*dev.ravel())
        return out
    def residual(p): return np.concatenate(blocks(p,True))
    lo=np.r_[np.full(rank,low),np.full(free,-3.0)]; hi=np.r_[np.full(rank,high),np.full(free,3.0)]; base=np.linspace(low,high,rank+2)[1:-1]; best=None
    for s in range(starts):
        out=least_squares(residual,np.r_[base+(s-(starts-1)/2)*.03,np.zeros(free)],bounds=(lo,hi),max_nfev=700,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        centre,rates,dev=unpack(out.x); e=np.concatenate(blocks(out.x,False)); pen=float(shrinkage*np.sum(dev**2))
        cand={'shared_geometric_centre':centre.tolist(),'group_rates':{k:v.tolist() for k,v in rates.items()},'data_sse':float(e@e),'penalty_sse':pen,'penalized_objective':float(e@e)+pen,'success':bool(out.success)}
        if best is None or cand['penalized_objective']<best['penalized_objective']: best=cand
    centre=np.asarray(best['shared_geometric_centre']); best.update({'rank':rank,'groups':groups,'shrinkage':float(shrinkage),'maximum_absolute_log_deviation':float(max(np.max(np.abs(np.log(v)-np.log(centre))) for v in best['group_rates'].values())),'interpretation':'Select shrinkage without the final held-out evidence group.'})
    return best

def continuous_spectrum_curves(time, *, groups=6, curves_per_group=2, grid_size=256, centre_rate=.25, log_width=1.0, noise_std=0.0, seed=0):
    """Generate a dense log-normal relaxation-spectrum stress test."""
    if groups<2 or curves_per_group<1 or grid_size<16 or log_width<=0: raise ValueError('invalid configuration')
    rng=np.random.default_rng(seed); t=np.asarray(time,float); lr=np.linspace(np.log(centre_rate)-4*log_width,np.log(centre_rate)+4*log_width,grid_size); rates=np.exp(lr)
    base=np.exp(-.5*((lr-np.log(centre_rate))/log_width)**2); base/=np.trapezoid(base,lr); kernel=np.exp(-np.outer(t,rates)); rows=[]
    for gi in range(groups):
        tilt=(gi-(groups-1)/2)/(groups-1); w=base*np.exp(.15*tilt*(lr-np.log(centre_rate))); w/=np.trapezoid(w,lr); relax=np.trapezoid(kernel*w[None,:],lr,axis=1)
        for ci in range(curves_per_group):
            value=.02*rng.normal()+(.8+.4*rng.random())*relax+noise_std*rng.normal(size=len(t)); rows.append(CurveRecord(f'g{gi}_c{ci}',f'g{gi}',f'c{ci}',t.copy(),value))
    return rows

def conformal_upper_quantile(scores: Sequence[float], alpha=.10):
    values=np.asarray(scores,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all(): raise ValueError('scores must be finite')
    if not 0<alpha<1: raise ValueError('alpha must lie in (0,1)')
    rank=min(len(values),math.ceil((len(values)+1)*(1-alpha))); return float(np.partition(values,rank-1)[rank-1])

def conformal_upper_pvalue(calibration_scores: Sequence[float], test_score):
    values=np.asarray(calibration_scores,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or not np.isfinite(test_score): raise ValueError('scores must be finite')
    return float((1+np.sum(values>=test_score))/(len(values)+1))

def grouped_conformal_audit(calibration: Mapping[str,float], tests: Mapping[str,float], *, alpha=.10):
    threshold=conformal_upper_quantile(list(calibration.values()),alpha); values=list(calibration.values())
    records={g:{'score':float(s),'p_value':conformal_upper_pvalue(values,float(s)),'exceeds_threshold':bool(s>threshold)} for g,s in tests.items()}
    return {'alpha':float(alpha),'threshold':threshold,'calibration_groups':sorted(calibration),'test_records':records,'guarantee_scope':'Finite-sample marginal coverage requires exchangeable independent groups; no distribution-free guarantee under shift.'}
