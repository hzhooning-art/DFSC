"""Frozen one-condition empirical hold-state pilot; no sensor guarantee asserted."""
from pathlib import Path
import csv, hashlib, io, json, time, zipfile
import numpy as np
from scipy.optimize import nnls, lsq_linear, least_squares
from prediction_time_search import Information, strict_witness, decision
from prediction_time_dual import enclose_dual

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT/'experiments/specs/polyborosiloxane_hold_20261021.json'
DATA = ROOT/'data/external/polyborosiloxane_20261021'
OUT = ROOT/'results/polyborosiloxane_hold_20261021'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())+'\n', encoding='utf-8')

def read_panel(z, spec, panel):
    prefix=spec['figure']+panel
    member=f"SupplementaryInformation/{spec['figure']}/{prefix}.csv"
    rows=list(csv.DictReader(io.StringIO(z.read(member).decode('utf-8-sig'))))
    xkey=f"{prefix}_{spec['temperature_C']}C_x"; ykey=xkey[:-1]+'y'
    records=[]
    for i, row in enumerate(rows, 2):
        if not row[xkey].strip() and not row[ykey].strip(): continue
        if not row[xkey].strip() or not row[ykey].strip(): raise ValueError('Unpaired CSV coordinates')
        t, y=float(row[xkey]), float(row[ykey])
        if not np.isfinite([t,y]).all(): raise ValueError('Nonfinite source')
        records.append({'csv_row':i, 'time':t, 'response':y})
    t=np.array([r['time'] for r in records]); y=np.array([r['response'] for r in records])
    if len(t)<10 or np.any(np.diff(t)<=0): raise ValueError('Source time must strictly increase')
    return {'panel':panel, 'member':member, 'x_column':xkey, 'y_column':ykey, 'records':records},t,y

def pick(t, requested):
    if requested < t[0]-1e-10 or requested > t[-1]+1e-10:
        raise ValueError(f'Requested time {requested} outside source range {t[0],t[-1]}')
    return int(np.argmin(abs(t-requested)))

def crossing(t, y, anchor):
    v=y/y[anchor]; threshold=np.exp(-1)
    indices=np.flatnonzero((np.arange(len(t))>anchor)&(v<=threshold))
    if not len(indices): raise ValueError('Calibration has no 1/e crossing')
    j=int(indices[0]); i=j-1
    if not v[i]>threshold: raise ValueError('Invalid first crossing')
    return float(t[i]+(t[j]-t[i])*(v[i]-threshold)/(v[i]-v[j])-t[anchor])

def prony_fit(delays, centers, components, targets):
    attempts=[]
    for factor in (0.1, 1., 10.):
        initial=np.r_[np.full(components,max(centers[0],1e-8)/components),
                      factor*np.geomspace(.25,4.,components)]
        try:
            fit=least_squares(lambda q: np.exp(-delays[:,None]*q[components:])@q[:components]-centers,
                initial, bounds=(0,np.inf),max_nfev=10000,xtol=1e-12,ftol=1e-12,gtol=1e-12)
            rss=float(np.sum(fit.fun**2))
            attempts.append({'success':bool(fit.success),'rss':rss,'parameters':fit.x.tolist(),
                             'nfev':fit.nfev,'message':fit.message})
        except (ValueError,RuntimeError) as e: attempts.append({'success':False,'error':str(e)})
    valid=[a for a in attempts if a['success']]
    if not valid:return {'components':components,'success':False,'attempts':attempts}
    best=min(valid,key=lambda a:a['rss']);q=np.array(best['parameters'])
    return {'components':components,'success':True,'attempts':attempts,'parameters':q.tolist(),
            'rss':best['rss'],'aic':len(centers)*np.log(max(best['rss']/len(centers),1e-300))+4*components,
            'predictions':(np.exp(-targets[:,None]*q[components:])@q[:components]).tolist()}

def main():
    if (OUT/'protocol_receipt.json').exists(): raise FileExistsError('Frozen pilot may not be overwritten')
    OUT.mkdir(parents=True,exist_ok=True);spec=json.loads(SPEC.read_text())
    archive=DATA/spec['archive'];assert sha(archive)==spec['archive_sha256']
    bindings=[SPEC,Path(__file__),archive,DATA/'header_inventory.json']+[
        ROOT/'experiments'/f for f in ('prediction_time_search.py','prediction_time_dual.py','observation_weighted_prediction.py')]
    put(OUT/'protocol_receipt.json',{'frozen_before_response_values_opened_by_this_study':True,
        'frozen_before_prediction':True,'bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in bindings},
        'not_independent_external_blinding':True,'instrument_error_calibrated':False,
        'cross_loading_qualified':False,'scope':'Empirical normalized hold-state pilot'})
    for p in bindings:
        if p.suffix=='.zip':continue
        target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    try:
        start=time.perf_counter();records=[];cal=[]
        with zipfile.ZipFile(archive) as z:
            for panel in spec['calibration_panels']:
                record,t,y=read_panel(z,spec,panel);a=pick(t,spec['anchor_seconds'])
                if y[a]<=0:raise ValueError('Nonpositive calibration anchor')
                tau=crossing(t,y,a);cal.append((t,y/y[a],a,tau));records.append(record)
            scale=float(np.mean([c[3] for c in cal]));assert scale>0
            nominal_obs=spec['anchor_seconds']+scale*np.array(spec['observation_scale_multipliers'])
            nominal_targets=spec['anchor_seconds']+scale*np.array(spec['target_scale_multipliers'])
            cal_values=np.array([[v[pick(t,u)] for u in nominal_obs] for t,v,a,tau in cal])
            bands=np.maximum(.01,abs(cal_values[0]-cal_values[1]))
            calibration={'scale_seconds':scale,'individual_crossings_seconds':[c[3] for c in cal],
                'nominal_observation_seconds':nominal_obs.tolist(),'nominal_target_seconds':nominal_targets.tolist(),
                'calibration_normalized_observations':cal_values.tolist(),'frozen_bands':bands.tolist(),
                'uses_held_out_panel':False,'error_bound_is_empirical':True}
            put(OUT/'calibration.json',calibration)
            record,t,y=read_panel(z,spec,spec['held_out_panel']);records.append(record)
        put(OUT/'extracted.json',{'traces':records,'scope':'Unmodified selected source columns; no smoothing or tail exclusions'})
        anchor=pick(t,spec['anchor_seconds']);divisor=float(y[anchor])
        if divisor<=0:raise ValueError('Nonpositive held-out anchor')
        obs=[pick(t,u) for u in nominal_obs];targets=[pick(t,u) for u in nominal_targets]
        if len(set(obs))!=len(obs) or len(set(targets))!=len(targets) or min(targets)<=max(obs):
            raise ValueError('Time split has duplicate observations or overlapping held-out targets')
        delays=(t[obs]-t[anchor])/scale;waits=(t[targets]-t[anchor])/scale
        centers=y[obs]/divisor;actual=y[targets]/divisor
        info=Information(1.,0.,tuple(delays),tuple(centers),tuple(bands));feasible=strict_witness(info)
        dictionary=spec['nnls_dictionary'];rates=np.r_[0.,np.geomspace(dictionary['min_rate'],dictionary['max_rate'],dictionary['positive_nodes'])]
        E=np.exp(-delays[:,None]*rates);Q=np.exp(-waits[:,None]*rates)
        coeff,resnorm=nnls(E,centers,maxiter=10000)
        lam=spec['nonnegative_ridge_lambda'];ridge=lsq_linear(np.vstack((E,np.sqrt(lam)*np.eye(len(rates)))),
            np.r_[centers,np.zeros(len(rates))],bounds=(0,np.inf),tol=1e-12,max_iter=2000,lsq_solver='exact')
        prony=[prony_fit(delays,centers,k,waits) for k in spec['prony_components']]
        best=min([p for p in prony if p['success']],key=lambda p:p['aic'],default=None)
        baseline={'nnls':{'success':True,'training_residual_norm':float(resnorm),'predictions':(Q@coeff).tolist()},
                  'nonnegative_ridge':{'success':bool(ridge.success),'message':ridge.message,
                      'training_residual_norm':float(np.linalg.norm(E@ridge.x-centers)),
                      'predictions':(Q@ridge.x).tolist()},'prony_candidates':prony,
                  'prony_selected':best,'dictionary_is_not_a_physical_rate_cap':True}
        put(OUT/'early_fit_baselines.json',baseline)
        arrays={};logs=[];rows=[]
        for j,targetindex in enumerate(targets):
            bracket=None;answer={'state':'unresolved'};levels=[];failures=[]
            if feasible is not None:
                for cells in spec['mesh_levels']:
                    levels.append(cells)
                    try:bracket=enclose_dual(info,1.,float(waits[j]),cells,f'pbs_t{j}_c{cells}',arrays,logs,feasible=feasible)
                    except RuntimeError as e:failures.append(str(e));break
                    answer=decision(bracket,spec['absolute_tolerance'],spec['relative_tolerance'],0.)
                    if answer['state']!='unresolved':break
            z=answer.get('joint_prediction')
            fits={name:float(baseline[name]['predictions'][j]) for name in ('nnls','nonnegative_ridge')}
            if best is not None:fits['prony']=float(best['predictions'][j])
            def metrics(value):
                ae=abs(value-float(actual[j]));re=ae/abs(float(actual[j])) if actual[j]!=0 else None
                return {'prediction':value,'absolute_error':ae,'relative_error':re,
                    'meets_joint_tolerance':bool(ae<=spec['absolute_tolerance'] and re is not None and re<=spec['relative_tolerance'])}
            rows.append({'target_index':j,'actual_seconds':float(t[targetindex]),'nominal_seconds':float(nominal_targets[j]),
                'wait':float(waits[j]),'csv_row':record['records'][targetindex]['csv_row'],
                'measured_response':float(actual[j]),'bracket':bracket,'decision':answer,'levels':levels,'failures':failures,
                'measured_target_in_outer':bool(bracket['outer'][0]<=actual[j]<=bracket['outer'][1]) if bracket else None,
                'joint_estimate_metrics':metrics(float(z)) if z is not None else None,
                'point_baselines':{name:metrics(value) for name,value in fits.items()}})
        np.savez_compressed(OUT/'lp_witnesses.npz',**arrays)
        accepted=[r for r in rows if r['decision']['state']=='sufficient']
        result={'spec_sha256':sha(SPEC),'protocol_sha256':sha(OUT/'protocol_receipt.json'),
            'calibration_sha256':sha(OUT/'calibration.json'),'raw_archive_sha256':sha(archive),
            'observations':{'actual_seconds':t[obs].tolist(),'delays':delays.tolist(),'centers':centers.tolist(),
                'bands':bands.tolist(),'anchor_seconds':float(t[anchor]),'anchor_divisor':divisor,
                'csv_rows':[record['records'][i]['csv_row'] for i in obs]},
            'strict_witness':feasible,'results':rows,'lp_logs':logs,'lp_witnesses_sha256':sha(OUT/'lp_witnesses.npz'),
            'summary':{'held_out_specimens':1,'correlated_targets':len(rows),
                'states':{s:sum(r['decision']['state']==s for r in rows) for s in ('sufficient','insufficient','unresolved')},
                'computed_outer_targets':sum(r['bracket'] is not None for r in rows),
                'outer_coverage_count':sum(r['measured_target_in_outer'] is True for r in rows),
                'accepted_joint_error_exceedances':sum(not r['joint_estimate_metrics']['meets_joint_tolerance'] for r in accepted),
                'point_baseline_joint_tolerance_counts':{name:sum(r['point_baselines'][name]['meets_joint_tolerance'] for r in rows) for name in fits}},
            'elapsed_seconds':time.perf_counter()-start,'deterministic_material_guarantee_validated':False,
            'cross_loading_validated':False,'qualification_gaps':spec['qualification_gaps']}
        put(OUT/'result.json',result);print(json.dumps(result['summary'],indent=2))
    except Exception as e:
        put(OUT/'execution_failure.json',{'error_type':type(e).__name__,'error':str(e),'protocol_sha256':sha(OUT/'protocol_receipt.json'),
             'action':'Stop without window/band/sample retuning. No successful material validation asserted.'})
        raise

if __name__=='__main__':main()
