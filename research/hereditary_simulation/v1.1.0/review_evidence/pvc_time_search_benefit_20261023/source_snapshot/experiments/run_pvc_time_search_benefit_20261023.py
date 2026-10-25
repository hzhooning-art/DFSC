"""One bounded matched-oracle scheduling study; old material protocol unchanged."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from prediction_time_search import Information
from run_matched_method_benefit_20261021 import workflow
ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'experiments/specs/pvc_time_search_benefit_20261023.json'
OUT=ROOT/'results/pvc_time_search_benefit_20261023'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n',encoding='utf-8')
def main():
    if OUT.exists():raise FileExistsError('Preserve frozen study')
    OUT.mkdir(parents=True);spec=json.loads(SPEC.read_text());parent=json.loads((ROOT/spec['parent_spec']).read_text())
    data=json.loads((ROOT/spec['parent_data']).read_text())
    bindings=[SPEC,Path(__file__),ROOT/spec['parent_spec'],ROOT/spec['parent_data']]+[
        ROOT/'experiments'/f for f in ('run_matched_method_benefit_20261021.py','prediction_time_search.py',
        'prediction_time_dual.py','prediction_time_transport.py','observation_weighted_prediction.py')]
    put(OUT/'protocol_receipt.json',{'bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in bindings},
        'frozen_before_this_study_execution':True,'previous_material_results_known':True,'claim_ceiling':spec['claim_ceiling']})
    for p in bindings:
        dst=OUT/'source_snapshot'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    cases=[]
    for tr in data['traces']:
        t=np.array([r['time_seconds'] for r in tr['records']]);y=np.array([r['modulus_MPa'] for r in tr['records']])
        span=t[-1]-t[0];divisor=y[0]
        for window in parent['observation_window_fractions']:
            obs=[int(np.argmin(abs(t-(t[0]+window*f*span)))) for f in parent['observation_within_window_fractions']]
            indices=np.flatnonzero(t>max(t[obs]));waits=((t[indices]-t[0])/span).tolist()
            for band in parent['assumed_relative_to_first_measured_modulus_bands']:
                cases.append({'id':f'T{tr["temperature_K"]}_w{window}_b{band}','temperature_K':tr['temperature_K'],
                    'window_fraction':window,'band_fraction':band,'delays':((t[obs]-t[0])/span).tolist(),
                    'centers':(y[obs]/divisor).tolist(),'waits':waits,'actual_seconds':t[indices].tolist(),
                    'measured_targets_scaled':(y[indices]/divisor).tolist()})
    put(OUT/'cases.json',cases)
    # Exclude a shared warmup identically; it does not populate any algorithm cache.
    first=cases[0];info=Information(1.,0.,tuple(first['delays']),tuple(first['centers']),(first['band_fraction'],)*5)
    workflow(info,[first['waits'][0]],0.,spec,False,'warmup',False)
    timings=[];initial={}
    for rep in range(spec['repetitions']):
        order=('pointwise_classical','loading_guided_reuse') if rep%2==0 else ('loading_guided_reuse','pointwise_classical')
        for case in cases:
            info=Information(1.,0.,tuple(case['delays']),tuple(case['centers']),(case['band_fraction'],)*5)
            for name in order:
                prefix=case['id']+'_'+name
                result,arrays=workflow(info,case['waits'],0.,spec,name=='loading_guided_reuse',prefix,rep==0)
                record={'case':case['id'],'algorithm':name,'repetition':rep,
                    **{k:result[k] for k in ('endpoint_programs','solver_calls','total_seconds','feasibility_seconds','states','first_sufficient','trace','failures')}}
                timings.append(record)
                if rep==0:
                    put(OUT/(prefix+'.json'),result);np.savez_compressed(OUT/(prefix+'.npz'),**arrays)
                    initial[(case['id'],name)]=result
        print(json.dumps({'repetition':rep,'workflows':len(cases),'completed':True}),flush=True)
    conflicts=[];unequal=[];accepted=[];bycase=[]
    for case in cases:
        a=initial[(case['id'],'pointwise_classical')];b=initial[(case['id'],'loading_guided_reuse')]
        for i,(x,y) in enumerate(zip(a['rows'],b['rows'])):
            if x['state']!=y['state']:
                unequal.append({'case':case['id'],'wait':x['wait'],'states':[x['state'],y['state']]})
                if 'unresolved' not in (x['state'],y['state']):conflicts.append(unequal[-1])
            for name,row in (('pointwise_classical',x),('loading_guided_reuse',y)):
                if row['state']=='sufficient':
                    measured=case['measured_targets_scaled'][i];q=row['joint_prediction'];ae=abs(q-measured);re=ae/abs(measured)
                    accepted.append({'case':case['id'],'algorithm':name,'seconds':case['actual_seconds'][i],
                        'relative_error':re,'absolute_error_scaled':ae,'joint_tolerance_met':bool(ae<=.05 and re<=.10)})
        bycase.append({'case':case['id'],'requests':len(case['waits']),
            'pointwise_endpoint_programs':a['endpoint_programs'],'reuse_endpoint_programs':b['endpoint_programs'],
            'first_sufficient_equal':a['first_sufficient']==b['first_sufficient'],
            'pointwise_states':a['states'],'reuse_states':b['states']})
    aggregate=[]
    for rep in range(spec['repetitions']):
        item={'repetition':rep}
        for name in ('pointwise_classical','loading_guided_reuse'):
            subset=[r for r in timings if r['algorithm']==name and r['repetition']==rep]
            item[name]={'total_seconds':sum(r['total_seconds'] for r in subset),
                'endpoint_programs':sum(r['endpoint_programs'] for r in subset)}
        item['speed_ratio']=item['pointwise_classical']['total_seconds']/item['loading_guided_reuse']['total_seconds'];aggregate.append(item)
    budgets=[]
    for budget in spec['endpoint_program_budgets']:
        item={'endpoint_program_budget_per_case':budget}
        for name in ('pointwise_classical','loading_guided_reuse'):
            snapshots=[next(s for s in reversed(initial[(c['id'],name)]['trace']) if s['endpoint_programs']<=budget) for c in cases]
            item[name]={s:sum(x['states'][s] for x in snapshots) for s in ('sufficient','insufficient','unresolved')}
        budgets.append(item)
    put(OUT/'result.json',{'cases':len(cases),'correlated_requests':sum(len(c['waits']) for c in cases),
        'spec_sha256':sha(SPEC),'protocol_sha256':sha(OUT/'protocol_receipt.json'),'timing_rows':timings,
        'aggregate':aggregate,'median_speed_ratio':float(np.median([a['speed_ratio'] for a in aggregate])),
        'by_case':bycase,'budget_results':budgets,'opposite_decisive_labels':conflicts,'all_label_differences':unequal,
        'accepted_against_raw_targets':accepted,'claimed_physical_guarantee':False,'claim_ceiling':spec['claim_ceiling']})
    print(json.dumps({'aggregate':aggregate,'opposite_decisive_labels':conflicts,'label_differences':len(unequal)},indent=2))
if __name__=='__main__':main()
