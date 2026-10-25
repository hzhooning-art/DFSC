"""Replay predictions through a separately written exponential primitive."""
import argparse
import json
from pathlib import Path
import numpy as np
from direct_fitting_baselines import ROOT, OUT, SPEC, sha, summarize
from ultraslow_calibrated_validation import inputs


def independent_response(times, segments, weights, rates):
    answer=np.zeros(len(times))
    for start,end,velocity in segments:
        age1=np.maximum(times-start,0.);age2=np.maximum(times-end,0.)
        answer+=velocity*weights[0]*(age1-age2)
        for weight,rate in zip(weights[1:],rates):
            if rate==0: part=age1-age2
            else: part=(-np.expm1(-rate*age1)+np.expm1(-rate*age2))/rate
            answer+=velocity*weight*part
    return answer


def audit(phase):
    p=json.loads(SPEC.read_text());path=OUT/f'{phase}.json';result=json.loads(path.read_text())
    freeze=json.loads((OUT/f'{phase}_protocol_receipt.json').read_text())
    assert result['complete'] and result['bindings']==freeze['bindings']
    for rel,value in result['bindings'].items():assert sha(ROOT/rel)==value,rel
    assert result['case_count']==(256 if phase=='full' else 16)
    assert result['window_records']==2*result['case_count']
    if phase=='full':
        pilot=json.loads((OUT/'pilot.json').read_text())
        assert all(result['bindings'].get(k)==v for k,v in pilot['bindings'].items()),'Protocol/source changed after pilot'
    parent_path=ROOT/'results'/p['parent_result'];parent=json.loads(parent_path.read_text())
    _,_,heldout,pilot,schedules=inputs()
    max_prediction_gap=0.;checked=0;candidate_failures={};refit_failures={};start_failures=0
    with np.load(parent_path.with_suffix('.npz'),allow_pickle=False) as data:
        for row in result['records']:
            cid=row['case'];hi=p['windows'].index(row['window'])
            y=np.r_[data[f'case{cid}_pilot'],data[f'case{cid}_future_{hi}']]
            pieces=pilot+schedules[str(row['window'])]
            assert parent['cases'][cid]['calibration_valid']==row['calibration_valid']
            for arm,model in row['models'].items():
                candidate_failures[arm]=candidate_failures.get(arm,0)+sum(not s['success'] for s in model.get('selection',[]))
                refit_failures[arm]=refit_failures.get(arm,0)+int(not model['success'])
                for fit in [model]+model.get('selection',[]):
                    start_failures+=sum(not a['success'] for a in fit.get('attempts',[]))
                good=[s for s in model.get('selection',[]) if s['success']]
                if good:
                    selected=min(good,key=lambda s:s['validation_mse'])
                    if arm=='grid_ridge':assert model['penalty']==selected['penalty']
                    if arm=='prony_cv':assert model['order']==selected['order']
                if not model['success']:
                    assert not row['evaluation'][arm]['success'];continue
                w=np.array(model['weights']);rates=np.array(model['rates'])
                assert w.min()>=-1e-8 and abs(w.sum()-1)<=1e-8
                assert rates.min()>=0 and rates.max()<=4.+1e-12
                predictions=np.concatenate([independent_response(t,s,w,rates) for t,s in pieces])
                ev=row['evaluation'][arm]
                assert abs(np.sqrt(np.mean((predictions-y)**2))-ev['response_rmse'])<2e-8
                assert abs(np.max(np.abs(predictions-y))-ev['response_max_residual'])<2e-8
                g=float(w[0]+np.dot(w[1:],np.exp(-20*rates)))
                sigma=float(independent_response(np.array([20.]),heldout,w,rates)[0])
                gap=max(abs(g-ev['predictions']['G20']),abs(sigma-ev['predictions']['heldout_sigma20']))
                max_prediction_gap=max(max_prediction_gap,gap);assert gap<2e-8
                for target in p['targets']:
                    error=abs(ev['predictions'][target]-parent['truth_targets'][row['family']][target])
                    assert abs(error-ev['absolute_errors'][target])<1e-12
                    assert ev['exceeds_tolerance'][target]==(error>p['absolute_point_tolerances'][target]+2e-7)
                checked+=1
    assert summarize(result['records'],p)==result['summaries']
    report={'schema':'P4-direct-fitting-audit-v1','phase':phase,'all_checks_pass':True,
        'result_sha256':sha(path),'protocol_sha256':sha(SPEC),'checked_successful_fits':checked,
        'maximum_independent_prediction_gap':max_prediction_gap,'candidate_order_or_penalty_failures':candidate_failures,
        'final_refit_failures':refit_failures,'failed_nonlinear_starts':start_failures,
        'new_random_trials':False,'independent_external_validation':False,
        'limits':'Separate primitive replay, metadata and aggregation checks; not a global-optimum certificate or an independent maintainer review.'}
    (OUT/f'{phase}_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','full'],required=True)
    audit(parser.parse_args().phase)
