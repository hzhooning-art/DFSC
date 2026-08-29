from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import nnls
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from p5_memory_protocol import (CurveRecord, conformal_upper_pvalue, continuous_spectrum_curves, fit, fit_oscillatory_shared, fit_partially_shared, generalized_design)

SEEDS=tuple(range(12))

def nrmse(y,p,cal): return float(np.sqrt(np.mean((y-p)**2))/max(np.ptp(cal),.1))
def predict_rates(rows,rates,split):
    out=[]
    for row in rows:
        x=generalized_design(row.time[:split],decay_rates=rates); c=np.linalg.lstsq(x,row.value[:split],rcond=None)[0]
        p=generalized_design(row.time[split:],decay_rates=rates)@c; out.append(nrmse(row.value[split:],p,row.value[:split]))
    return float(np.median(out))
def predict_pairs(rows,rates,pairs,split):
    out=[]
    for row in rows:
        x=generalized_design(row.time[:split],decay_rates=rates,oscillatory_pairs=pairs); c=np.linalg.lstsq(x,row.value[:split],rcond=None)[0]
        p=generalized_design(row.time[split:],decay_rates=rates,oscillatory_pairs=pairs)@c; out.append(nrmse(row.value[split:],p,row.value[:split]))
    return float(np.median(out))
def short(rows,split): return [CurveRecord(r.unit,r.group,r.channel,r.time[:split],r.value[:split]) for r in rows]

def oscillatory(seed):
    rng=np.random.default_rng(seed); t=np.linspace(0,10,120); split=72; rows=[]
    for g in range(6):
        for c in range(2):
            e=np.exp(-.16*t); y=.02*rng.normal()+(1+.15*rng.random())*e*np.cos(2.2*t)+(.15+.1*rng.random())*e*np.sin(2.2*t)+.004*rng.normal(size=len(t))
            rows.append(CurveRecord(f'g{g}c{c}',f'g{g}',f'c{c}',t,y))
    train=short(rows,split); real=fit(train,3,starts=4,rate_bounds=(1/300,5)); osc=fit_oscillatory_shared(train,starts=5)
    return {'seed':seed,'real_nrmse':predict_rates(rows,real['rates'],split),'oscillatory_nrmse':predict_pairs(rows,osc['decay_rates'],osc['oscillatory_pairs'],split),'estimated_damping':osc['oscillatory_pairs'][0][0],'estimated_frequency':osc['oscillatory_pairs'][0][1],'delta_bic_real_minus_osc':real['bic']-osc['bic']}

def partial(seed):
    rng=np.random.default_rng(100+seed); t=np.linspace(0,14,100); split=62; rows=[]; truth={}
    for g,dev in enumerate(np.linspace(-.35,.35,6)):
        rate=.28*np.exp(dev); truth[f'g{g}']=rate
        for c in range(2):
            y=.02*rng.normal()+(1+.2*rng.random())*np.exp(-rate*t)+.004*rng.normal(size=len(t)); rows.append(CurveRecord(f'g{g}c{c}',f'g{g}',f'c{c}',t,y))
    train=short(rows,split); pooled=fit(train,1,starts=4); part=fit_partially_shared(train,1,shrinkage=.01,starts=3)
    pooled_err=predict_rates(rows,pooled['rates'],split); errors=[]; rate_errors=[]
    for row in rows:
        r=part['group_rates'][row.group]; x=generalized_design(t[:split],decay_rates=r); c=np.linalg.lstsq(x,row.value[:split],rcond=None)[0]; p=generalized_design(t[split:],decay_rates=r)@c; errors.append(nrmse(row.value[split:],p,row.value[:split])); rate_errors.append(abs(np.log(r[0]/truth[row.group])))
    return {'seed':seed,'pooled_nrmse':pooled_err,'partial_nrmse':float(np.median(errors)),'median_log_rate_error':float(np.median(rate_errors)),'maximum_log_deviation':part['maximum_absolute_log_deviation']}

def continuous(seed):
    t=np.linspace(0,30,100); split=60; rows=continuous_spectrum_curves(t,groups=6,curves_per_group=2,log_width=1.15,noise_std=.001,seed=200+seed); train=short(rows,split); finite=fit(train,3,starts=4)
    finite_err=predict_rates(rows,finite['rates'],split); dense=[]; grid=np.geomspace(1/300,2,32)
    for row in rows:
        x=generalized_design(t[:split],decay_rates=grid); coef=nnls(x,row.value[:split])[0]; p=generalized_design(t[split:],decay_rates=grid)@coef; dense.append(nrmse(row.value[split:],p,row.value[:split]))
    dense_err=float(np.median(dense)); gain=(finite_err-dense_err)/max(finite_err,1e-15)
    return {'seed':seed,'finite_rank3_nrmse':finite_err,'dense_grid_nrmse':dense_err,'dense_relative_gain':gain,'refuse_finite_mechanism':bool(gain>=.20),'finite_rates':finite['rates']}

def group_score(rng,shift=False):
    t=np.linspace(0,12,80); split=48
    if shift:
        y=.01*rng.normal()+np.exp(-.18*t)*(.65*np.cos(1.1*t)+.35)+.004*rng.normal(size=len(t))
    else:
        y=.01*rng.normal()+(1+.1*rng.normal())*np.exp(-.30*t)+.004*rng.normal(size=len(t))
    x=generalized_design(t[:split],decay_rates=[.30]); c=np.linalg.lstsq(x,y[:split],rcond=None)[0]; p=generalized_design(t[split:],decay_rates=[.30])@c
    return nrmse(y[split:],p,y[:split])
def conformal(seed):
    rng=np.random.default_rng(300+seed); calibration=[group_score(rng) for _ in range(39)]; null=[group_score(rng) for _ in range(20)]; shifted=[group_score(rng,True) for _ in range(20)]
    p0=[conformal_upper_pvalue(calibration,s) for s in null]; p1=[conformal_upper_pvalue(calibration,s) for s in shifted]
    return {'seed':seed,'null_false_alarm_rate':float(np.mean(np.asarray(p0)<=.05)),'shift_detection_rate':float(np.mean(np.asarray(p1)<=.05)),'median_null_p':float(np.median(p0)),'median_shift_p':float(np.median(p1))}

def paired_summary(records,a,b):
    x=np.asarray([r[a] for r in records]); y=np.asarray([r[b] for r in records]); stat=wilcoxon(x,y,alternative='greater')
    return {'median_'+a:float(np.median(x)),'median_'+b:float(np.median(y)),'median_relative_gain':float(np.median((x-y)/np.maximum(x,1e-15))),'wilcoxon_one_sided_p':float(stat.pvalue)}
def main():
    osc=[oscillatory(s) for s in SEEDS]; part=[partial(s) for s in SEEDS]; cont=[continuous(s) for s in SEEDS]; conf=[conformal(s) for s in SEEDS]
    payload={'stage':'66-recommended-extension-combination','seeds':list(SEEDS),'oscillatory':{'records':osc,'summary':paired_summary(osc,'real_nrmse','oscillatory_nrmse')},'partial_sharing':{'records':part,'summary':paired_summary(part,'pooled_nrmse','partial_nrmse')},'continuous_spectrum':{'records':cont,'summary':{'refusal_rate':float(np.mean([r['refuse_finite_mechanism'] for r in cont])),'median_dense_relative_gain':float(np.median([r['dense_relative_gain'] for r in cont]))}},'group_conformal':{'records':conf,'summary':{'mean_null_false_alarm_rate':float(np.mean([r['null_false_alarm_rate'] for r in conf])),'mean_shift_detection_rate':float(np.mean([r['shift_detection_rate'] for r in conf])),'scope':'Exchangeable group guarantee only; shifted detection is empirical.'}}}
    checks={'oscillatory_gain':payload['oscillatory']['summary']['median_relative_gain']>.30 and payload['oscillatory']['summary']['wilcoxon_one_sided_p']<.01,'partial_gain':payload['partial_sharing']['summary']['median_relative_gain']>.15 and payload['partial_sharing']['summary']['wilcoxon_one_sided_p']<.01,'continuous_refusal':payload['continuous_spectrum']['summary']['refusal_rate']>=.8,'conformal_control':payload['group_conformal']['summary']['mean_null_false_alarm_rate']<=.10 and payload['group_conformal']['summary']['mean_shift_detection_rate']>=.8}; payload['checks']=checks; payload['route_pass']=all(checks.values())
    out=ROOT/'results'/'stage66_extension_combination.json'; out.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps({'route_pass':payload['route_pass'],'checks':checks,'summaries':{k:payload[k]['summary'] for k in ('oscillatory','partial_sharing','continuous_spectrum','group_conformal')}},indent=2))
if __name__=='__main__': main()
