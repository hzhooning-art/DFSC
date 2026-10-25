"""Parallel scheduling only; reuses the pilot's frozen fitting implementation."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import time
from pathlib import Path
import numpy as np
import scipy
from direct_fitting_baselines import ROOT,OUT,SPEC,sha,fit_models,evaluate,summarize
from ultraslow_calibrated_validation import inputs


def worker(job):
    p,pieces,y,heldout,bhat,truth,intervals,cid,family,seed,window,valid,symmetric=job
    started=time.perf_counter()
    models=fit_models(pieces,y,p)
    evaluations={name:evaluate(m,pieces,y,heldout,bhat,truth,intervals,p['absolute_point_tolerances']) for name,m in models.items()}
    for name,source in [('mid_symmetric',symmetric),('mid_classical',intervals)]:
        point={t:float(np.mean(source[t])) for t in p['targets']}
        errors={t:abs(point[t]-truth[t]) for t in p['targets']}
        evaluations[name]={'success':True,'predictions':point,'absolute_errors':errors,
            'exceeds_tolerance':{t:e>p['absolute_point_tolerances'][t]+2e-7 for t,e in errors.items()}}
    seconds={name:m.get('seconds',0)+sum(c['seconds'] for c in m.get('selection',[])) for name,m in models.items()}
    return {'case':cid,'family':family,'seed':seed,'window':window,'calibration_valid':valid,
            'calibrated_bound':bhat,'models':models,'evaluation':evaluations,'fit_seconds':seconds,
            'seconds':time.perf_counter()-started}


def main():
    p=json.loads(SPEC.read_text());parent_path=ROOT/'results'/p['parent_result']
    parent=json.loads(parent_path.read_text());classical_path=ROOT/'results'/p['classical_result']
    classical=json.loads(classical_path.read_text())
    _,_,heldout,pilot,schedules=inputs()
    bindings={str(path.relative_to(ROOT)):sha(path) for path in [SPEC,ROOT/'experiments/direct_fitting_baselines.py',
        parent_path,parent_path.with_suffix('.npz'),classical_path]}
    pil=json.loads((OUT/'pilot.json').read_text());qa=json.loads((OUT/'pilot_audit.json').read_text())
    assert pil['bindings']==bindings and qa['all_checks_pass'] and qa['result_sha256']==sha(OUT/'pilot.json')
    assert sha(parent_path.with_suffix('.npz'))==parent['witness_archive_sha256']
    assert classical['parent_result_sha256']==sha(parent_path)
    bindings[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    for rel in ['experiments/audit_direct_fitting_baselines.py','experiments/constitutive_response_gate.py',
                'experiments/ultraslow_calibrated_validation.py','experiments/target_support_sensitivity.py']:
        bindings[rel]=sha(ROOT/rel)
    if (OUT/'full.json').exists() or (OUT/'full_protocol_receipt.json').exists():raise FileExistsError('Full outputs exist')
    workers=4
    (OUT/'full_protocol_receipt.json').write_text(json.dumps({'bindings':bindings,'phase':'full','before_first_fit':True,
        'workers':workers,'numpy':np.__version__,'scipy':scipy.__version__,
        'blas_thread_limit':os.environ.get('OPENBLAS_NUM_THREADS'),
        'scope':p['scope'],'pilot_fit_implementation_unchanged':True},indent=2)+'\n')
    old={(r['case'],r['horizon']):r for r in parent['rows']}
    cr={(r['case'],r['horizon']):r for r in classical['rows']}
    jobs=[]
    with np.load(parent_path.with_suffix('.npz'),allow_pickle=False) as z:
        for case in parent['cases']:
            cid=case['id']
            for hi,window in enumerate(p['windows']):
                row=old[cid,window];intervals=cr[cid,window]['methods']['classical_sign_convex']
                jobs.append((p,pilot+schedules[str(window)],
                    np.r_[z[f'case{cid}_pilot'],z[f'case{cid}_future_{hi}']],heldout,
                    case['calibrated_bound'],parent['truth_targets'][case['truth']],
                    {t:intervals[t]['interval'] for t in p['targets']},cid,case['truth'],case['seed'],window,
                    case['calibration_valid'],{t:row['posterior'][t]['interval'] for t in p['targets']}))
    records=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks=[pool.submit(worker,j) for j in jobs]
        for task in as_completed(tasks):
            records.append(task.result())
            if len(records)%16==0:
                print(json.dumps({'completed_window_records':len(records),'total':512,
                    'seconds':round(time.perf_counter()-started,2)}),flush=True)
                (OUT/'full_checkpoint.json').write_text(json.dumps({'complete':False,'records':records}))
    records.sort(key=lambda r:(r['case'],r['window']))
    result={'schema':'P4-direct-fitting-results-v1','phase':'full','complete':True,'case_count':256,
        'window_records':512,'new_random_trials':False,'new_fitting_executions':True,
        'bindings':bindings,'duration_seconds':time.perf_counter()-started,'workers':workers,
        'summaries':summarize(records,p),'records':records,'limits':p['limits']}
    (OUT/'full.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':True,'seconds':result['duration_seconds']}),flush=True)


if __name__=='__main__':main()
