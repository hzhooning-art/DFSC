"""One frozen sensitivity grid; successful cells are not calibrated validation."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.optimize import nnls,lsq_linear
from prediction_time_search import Information,strict_witness,decision
from prediction_time_dual import enclose_dual
from run_polyborosiloxane_hold_20261021 import prony_fit
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/polymer_glass_hold_20261021'
SPEC=ROOT/'experiments/specs/polymer_glass_hold_20261021.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n',encoding='utf-8')
def main():
    if (OUT/'protocol_receipt.json').exists():raise FileExistsError('Do not overwrite frozen run')
    spec=json.loads(SPEC.read_text());ext=json.loads((OUT/'extraction_protocol_receipt.json').read_text())
    assert ext['spec_sha256']==sha(SPEC)
    bindings=[SPEC,Path(__file__),OUT/'extracted.json',OUT/'extraction_protocol_receipt.json',
        ROOT/'tools/extract_polymer_glass_hold_20261021.py']+[
        ROOT/'experiments'/f for f in ('run_polyborosiloxane_hold_20261021.py','prediction_time_search.py','prediction_time_dual.py','observation_weighted_prediction.py')]
    put(OUT/'protocol_receipt.json',{'frozen_before_predictions':True,'bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in bindings},
        'assumed_not_calibrated_bands':True,'claim_ceiling':spec['claim_ceiling']})
    for p in bindings:
        dest=OUT/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    data=json.loads((OUT/'extracted.json').read_text());rows=[];baselines=[];logs=[];arrays={};start=time.perf_counter()
    for trace in data['traces']:
        t=np.array([r['time_seconds'] for r in trace['records']]);y=np.array([r['modulus_MPa'] for r in trace['records']])
        divisor=float(y[0]);assert divisor>0 and np.isfinite(y).all();span=float(t[-1]-t[0]);pick=lambda q:int(np.argmin(abs(t-q)))
        targets=[pick(t[0]+f*span) for f in spec['target_span_fractions']];waits=(t[targets]-t[0])/span
        for window in spec['observation_window_fractions']:
            obs=[pick(t[0]+window*f*span) for f in spec['observation_within_window_fractions']]
            if len(set(obs))!=5 or len(set(targets))!=3 or min(targets)<=max(obs):raise ValueError('Time split invalid; stop, do not shift')
            delays=(t[obs]-t[0])/span;centers=y[obs]/divisor
            rates=np.r_[0.,np.geomspace(1e-6,128.,128)];E=np.exp(-delays[:,None]*rates);Q=np.exp(-waits[:,None]*rates)
            fits={};fitmeta={}
            try:
                x,res=nnls(E,centers,maxiter=10000);fits['nnls']=Q@x;fitmeta['nnls']={'success':True,'training_residual':float(res)}
            except RuntimeError as e:fitmeta['nnls']={'success':False,'error':str(e)}
            ridge=lsq_linear(np.vstack((E,np.sqrt(.001)*np.eye(len(rates)))),np.r_[centers,np.zeros(len(rates))],
                bounds=(0,np.inf),tol=1e-12,max_iter=2000,lsq_solver='exact')
            fitmeta['ridge']={'success':bool(ridge.success),'message':ridge.message}
            if ridge.success:fits['ridge']=Q@ridge.x
            candidates=[prony_fit(delays,centers,k,waits) for k in (1,2)];valid=[p for p in candidates if p['success']]
            best=min(valid,key=lambda p:p['aic']) if valid else None
            fitmeta['prony']={'success':best is not None,'candidates':candidates}
            if best:fits['prony']=best['predictions']
            baselines.append({'temperature_K':trace['temperature_K'],'window_fraction':window,'fits':fitmeta})
            for band in spec['assumed_relative_to_first_measured_modulus_bands']:
                info=Information(1.,0.,tuple(delays),tuple(centers),(band,)*5);feasible=strict_witness(info)
                for j,idx in enumerate(targets):
                    bracket=None;answer={'state':'unresolved'};levels=[];failures=[]
                    if feasible is not None:
                        for mesh in spec['mesh_levels']:
                            levels.append(mesh)
                            try:bracket=enclose_dual(info,1.,float(waits[j]),mesh,
                                f'glass_T{trace["temperature_K"]}_w{window}_b{band}_t{j}_m{mesh}',arrays,logs,feasible=feasible)
                            except RuntimeError as e:failures.append(str(e));break
                            answer=decision(bracket,.05,.10,0.)
                            if answer['state']!='unresolved':break
                    actual=float(y[idx]/divisor);z=answer.get('joint_prediction')
                    def metrics(q):
                        ae=abs(q-actual);re=ae/abs(actual) if actual!=0 else None
                        return {'prediction_scaled':float(q),'absolute_error_scaled':float(ae),'absolute_error_MPa':float(ae*divisor),
                            'relative_error':float(re) if re is not None else None,
                            'joint_tolerance_met':bool(ae<=.05 and re is not None and re<=.10)}
                    rows.append({'temperature_K':trace['temperature_K'],'window_fraction':window,'band_fraction':band,
                        'numerical_divisor_MPa':divisor,'absolute_band_MPa':band*divisor,'span_seconds':span,
                        'actual_observation_seconds':t[obs].tolist(),'observation_source_rows':[trace['records'][i]['source_row'] for i in obs],
                        'centers':centers.tolist(),'delays':delays.tolist(),'band':band,
                        'target_source_row':trace['records'][idx]['source_row'],'target_index':j,'actual_seconds':float(t[idx]),
                        'wait':float(waits[j]),'measured_response':actual,'measured_modulus_MPa':float(y[idx]),
                        'strict_witness':feasible,'levels':levels,'failures':failures,'decision':answer,'bracket':bracket,
                        'measured_target_in_outer':bool(bracket['outer'][0]<=actual<=bracket['outer'][1]) if bracket else None,
                        'joint_estimate_metrics':metrics(float(z)) if z is not None else None,
                        'point_baselines':{name:metrics(float(v[j])) for name,v in fits.items()}})
        print(json.dumps({'temperature_K':trace['temperature_K'],'requests_done':len(rows)}),flush=True)
    np.savez_compressed(OUT/'lp_witnesses.npz',**arrays);put(OUT/'early_baselines.json',baselines)
    def summarize(subset):
        accepted=[r for r in subset if r['decision']['state']=='sufficient']
        return {'requests':len(subset),'states':{s:sum(r['decision']['state']==s for r in subset) for s in ('sufficient','insufficient','unresolved')},
            'computed_outers':sum(r['bracket'] is not None for r in subset),'measured_outer_coverage':sum(r['measured_target_in_outer'] is True for r in subset),
            'accepted_measured_joint_exceedances':sum(not r['joint_estimate_metrics']['joint_tolerance_met'] for r in accepted),
            'baseline_joint_passes':{name:sum(r['point_baselines'].get(name,{}).get('joint_tolerance_met',False) for r in subset) for name in ('nnls','ridge','prony')},
            'baseline_available':{name:sum(name in r['point_baselines'] for r in subset) for name in ('nnls','ridge','prony')}}
    put(OUT/'result.json',{'spec_sha256':sha(SPEC),'protocol_sha256':sha(OUT/'protocol_receipt.json'),
        'results':rows,'summary':summarize(rows),'by_window_band':[
            {'window_fraction':w,'band_fraction':b,**summarize([r for r in rows if r['window_fraction']==w and r['band_fraction']==b])}
            for w in spec['observation_window_fractions'] for b in spec['assumed_relative_to_first_measured_modulus_bands']],
        'lp_logs':logs,'lp_witnesses_sha256':sha(OUT/'lp_witnesses.npz'),'elapsed_seconds':time.perf_counter()-start,
        'calibrated_real_material_validation':False,'cross_loading_validated':False,'claim_ceiling':spec['claim_ceiling'],
        'qualification_gaps':spec['qualification_gaps']})
    print(json.dumps(json.loads((OUT/'result.json').read_text())['by_window_band'],indent=2))
if __name__=='__main__':main()
