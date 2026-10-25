"""Bounded empirical material pilot under EXPLICIT assumed noise and state class."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.optimize import nnls
from prediction_time_search import Information,strict_witness,decision
from prediction_time_dual import enclose_dual
ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'experiments/specs/orbital_hold_pilot_20261021.json'
OUT=ROOT/'results/orbital_hold_pilot_20261021'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n',encoding='utf-8')
def main():
    if (OUT/'protocol_receipt.json').exists():raise FileExistsError('Do not overwrite frozen pilot')
    spec=json.loads(SPEC.read_text());data=json.loads((OUT/'extracted.json').read_text())
    bindings=[SPEC,OUT/'extracted.json',Path(__file__),ROOT/'tools/extract_orbital_hold_20261021.py']+[ROOT/'experiments'/f for f in ('prediction_time_dual.py','prediction_time_search.py','observation_weighted_prediction.py')]
    put(OUT/'protocol_receipt.json',{'frozen_before_prediction':True,'bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in bindings},
        'noise_calibrated':False,'positive_hold_state_qualified':False,'full_loading_history_qualified':False,'scope':spec['scope']})
    for p in bindings:
        target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    archive,logs,results=[],[],[]
    saved={};start=time.perf_counter()
    for trace in data['traces']:
        raw=trace['raw_records'];ordered=sorted(raw,key=lambda r:r['time_seconds']);uniq={}
        for record in ordered:uniq.setdefault(record['time_seconds'],record)
        ordered=list(uniq.values());t=np.array([r['time_seconds'] for r in ordered]);stress=np.array([r['stress_Pa'] for r in ordered])
        choose=lambda target:int(np.argmin(abs(t-target)))
        anchor=choose(spec['anchor_seconds']);divisor=stress[anchor]
        if divisor<=0:raise ValueError('Nonpositive anchor: requires explicit pilot amendment')
        obs=[choose(x) for x in spec['observation_seconds']];targets=[choose(x) for x in spec['target_seconds']]
        assert min(targets)>max(obs) and len(set(obs))==len(obs)
        centers=stress[obs]/divisor;delays=(t[obs]-t[anchor])/spec['time_scale_seconds']
        assert delays[0]==0
        rates=np.r_[0.,np.geomspace(1e-6,128.,128)]
        E=np.exp(-delays[:,None]*rates)
        fit,fitnorm=nnls(E,centers,maxiter=10000)
        for band in spec['normalized_stress_bands']:
            info=Information(1.,0.,tuple(delays),tuple(centers),(band,)*len(obs));feasible=strict_witness(info)
            for targetindex in targets:
                wait=float((t[targetindex]-t[anchor])/spec['time_scale_seconds']);actual=float(stress[targetindex]/divisor)
                stem=f'{trace["sheet"].strip().replace(" ","_")}_b{band}_t{t[targetindex]}'
                bracket=None;answer={'state':'unresolved'};failures=[];levels=[]
                if feasible is not None:
                    for level in spec['mesh_levels']:
                        levels.append(level)
                        try:bracket=enclose_dual(info,1.,wait,level,stem+f'_c{level}',saved,logs,feasible=feasible)
                        except RuntimeError as error:
                            failures.append(str(error));break
                        answer=decision(bracket,spec['absolute_tolerance'],spec['relative_tolerance'],0.)
                        if answer['state']!='unresolved':break
                fitvalue=float(np.exp(-wait*rates)@fit)
                z=answer.get('joint_prediction')
                results.append({'sheet':trace['sheet'],'band':band,'nominal_target_seconds':spec['target_seconds'][targets.index(targetindex)],
                    'actual_target_seconds':float(t[targetindex]),'anchor_seconds':float(t[anchor]),'anchor_stress_Pa':float(divisor),
                    'observation_seconds':t[obs].tolist(),'centers':centers.tolist(),'delays':delays.tolist(),
                    'wait':wait,'actual_normalized_stress':actual,'strict_witness':feasible,'levels':levels,
                    'decision':answer,'bracket':bracket,'failures':failures,
                    'measured_target_in_outer':bracket['outer'][0]<=actual<=bracket['outer'][1] if bracket else None,
                    'joint_estimate_measured_abs_error':abs(z-actual) if z is not None else None,
                    'joint_estimate_measured_relative_error':abs(z-actual)/abs(actual) if z is not None and actual!=0 else None,
                    'fit':{'prediction':fitvalue,'training_residual_norm':float(fitnorm),'absolute_error':abs(fitvalue-actual),
                           'relative_error':abs(fitvalue-actual)/abs(actual) if actual!=0 else None}})
        print(json.dumps({'sheet':trace['sheet'],'strict_feasible_bands':[b for b in spec['normalized_stress_bands'] if any(r['sheet']==trace['sheet'] and r['band']==b and r['strict_witness'] is not None for r in results)]}),flush=True)
    np.savez_compressed(OUT/'lp_witnesses.npz',**saved)
    summary={}
    for band in spec['normalized_stress_bands']:
        rows=[r for r in results if r['band']==band];accepted=[r for r in rows if r['decision']['state']=='sufficient']
        summary[str(band)]={'traces':len(data['traces']),'requests':len(rows),
            'strict_feasible_traces':len({r['sheet'] for r in rows if r['strict_witness'] is not None}),
            'states':{s:sum(r['decision']['state']==s for r in rows) for s in ('sufficient','insufficient','unresolved')},
            'measured_targets_in_outer':sum(r['measured_target_in_outer'] is True for r in rows),
            'computed_outer_requests':sum(r['bracket'] is not None for r in rows),
            'accepted_joint_error_exceedances':sum(r['joint_estimate_measured_abs_error']>spec['absolute_tolerance'] or r['joint_estimate_measured_relative_error']>spec['relative_tolerance'] for r in accepted),
            'fit_joint_error_exceedances':sum(r['fit']['absolute_error']>spec['absolute_tolerance'] or r['fit']['relative_error']>spec['relative_tolerance'] for r in rows)}
    put(OUT/'result.json',{'spec_sha256':sha(SPEC),'protocol_sha256':sha(OUT/'protocol_receipt.json'),
        'results':results,'summary':summary,'lp_logs':logs,'lp_witnesses_sha256':sha(OUT/'lp_witnesses.npz'),
        'elapsed_seconds':time.perf_counter()-start,'guarantee_validated':False,'qualification_gaps':spec['qualification_gaps']})
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
