"""Matched point-prediction baselines; target truth enters evaluation only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares, minimize

from ultraslow_calibrated_validation import inputs
from constitutive_response_gate import strain

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'experiments/specs/direct_fitting_baselines_20261014.json'
OUT = ROOT / 'results/direct_fitting_20261014'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response_basis(pieces, rates):
    """Stable convolution with unit exponentials, including exact zero rate."""
    rates = np.asarray(rates)
    def primitive(age):
        age = np.asarray(age)[:, None]
        z = age * rates[None, :]
        small = np.abs(z) < 1e-4
        safe = np.where(rates == 0, 1., rates)[None, :]
        value = -np.expm1(-z) / safe
        derivative = ((z + 1) * np.exp(-z) - 1) / safe**2
        value = np.where(small, age * (1-z/2+z*z/6-z**3/24+z**4/120), value)
        derivative = np.where(small, age**2 * (-.5+z/3-z*z/8+z**3/30), derivative)
        return value, derivative
    matrices, derivatives = [], []
    for times, segments in pieces:
        a = np.column_stack((strain(times, segments), np.zeros((len(times), len(rates)))))
        d = np.zeros((len(times), len(rates)))
        for start, end, velocity in segments:
            v1, d1 = primitive(np.maximum(times-start, 0.))
            v2, d2 = primitive(np.maximum(times-end, 0.))
            a[:, 1:] += velocity*(v1-v2)
            d += velocity*(d1-d2)
        matrices.append(a); derivatives.append(d)
    return np.vstack(matrices), np.vstack(derivatives)


def simplex(sticks):
    k = len(sticks)
    weights = np.empty(k+1); jac = np.zeros((k+1, k))
    for j in range(k+1):
        preceding = np.prod(1-sticks[:j])
        weights[j] = preceding * (sticks[j] if j < k else 1.)
        if j < k: jac[j, j] = preceding
        for i in range(min(j, k)):
            jac[j, i] = -(sticks[j] if j < k else 1.) * np.prod(
                1-np.delete(sticks[:j], i))
    return weights, jac


def linear_fit(a, y, penalty, protocol):
    started = time.perf_counter()
    n, m = a.shape
    def objective(w):
        residual = a @ w-y
        return .5*np.dot(residual, residual)/n+.5*penalty*np.dot(w[1:], w[1:])
    def gradient(w):
        g = a.T @ (a @ w-y)/n
        g[1:] += penalty*w[1:]
        return g
    result = minimize(objective, np.full(m, 1/m), jac=gradient, method='SLSQP',
                      bounds=[(0., 1.)]*m,
                      constraints={'type':'eq', 'fun':lambda w: w.sum()-1,
                                   'jac':lambda w: np.ones(m)},
                      options={k: protocol['linear_solver'][k] for k in ('ftol', 'maxiter')})
    success = bool(result.success and np.isfinite(result.x).all()
                   and result.x.min() >= -1e-8 and abs(result.x.sum()-1) <= 1e-8)
    return {'success':success, 'weights':result.x.tolist(), 'status':int(result.status),
            'message':str(result.message), 'iterations':int(result.nit),
            'penalty':penalty, 'seconds':time.perf_counter()-started}


def prony_fit(pieces, y, order, protocol):
    started = time.perf_counter(); attempts = []
    for floor in protocol['prony_start_minimum_rates']:
        rates = np.geomspace(floor, protocol['prony_rate_bounds'][1], order)
        x0 = np.r_[1/np.arange(order+1, 1, -1), rates]
        def residual(x):
            weights, _ = simplex(x[:order])
            return response_basis(pieces, x[order:])[0] @ weights-y
        def jacobian(x):
            weights, dw = simplex(x[:order])
            a, da = response_basis(pieces, x[order:])
            return np.column_stack((a @ dw, da * weights[None, 1:]))
        r = least_squares(residual, x0, jac=jacobian,
                          bounds=(np.zeros(2*order), np.r_[np.ones(order),
                                  np.full(order, protocol['prony_rate_bounds'][1])]),
                          **protocol['prony_solver'])
        w, _ = simplex(r.x[:order])
        attempts.append({'success':bool(r.success and np.isfinite(r.x).all()),
                         'status':int(r.status), 'nfev':int(r.nfev),
                         'cost':float(r.cost), 'optimality':float(r.optimality),
                         'weights':w.tolist(), 'rates':r.x[order:].tolist(),
                         'initial_minimum_rate':floor})
    good = [a for a in attempts if a['success']]
    best = min(good, key=lambda a:a['cost']) if good else None
    return {'success':bool(good), 'order':order, 'attempts':attempts,
            'best':best, 'seconds':time.perf_counter()-started}


def split_pieces(pieces):
    train, valid, train_idx, valid_idx, offset = [], [], [], [], 0
    for t, s in pieces:
        cut = 3*len(t)//4
        train.append((t[:cut], s)); valid.append((t[cut:], s))
        train_idx.extend(range(offset, offset+cut)); valid_idx.extend(range(offset+cut, offset+len(t)))
        offset += len(t)
    return train, valid, np.array(train_idx), np.array(valid_idx)


def fit_models(pieces, y, protocol):
    """No truth, case identity, old intervals or calibration-valid flag accepted."""
    rates = np.geomspace(protocol['grid']['minimum_positive_rate'],
                         protocol['grid']['maximum_rate'], protocol['grid']['positive_nodes'])
    a, _ = response_basis(pieces, rates)
    train, valid, train_idx, valid_idx = split_pieces(pieces)
    models = {'grid_ls':linear_fit(a, y, 0., protocol)}
    models['grid_ls']['rates'] = rates.tolist()
    candidates = []
    for penalty in protocol['ridge_candidates']:
        fit = linear_fit(a[train_idx], y[train_idx], penalty, protocol)
        fit['validation_mse'] = float(np.mean((a[valid_idx] @ fit['weights']-y[valid_idx])**2)) if fit['success'] else None
        candidates.append(fit)
    good = [c for c in candidates if c['success']]
    if good:
        selected = min(good, key=lambda c:c['validation_mse'])
        models['grid_ridge'] = linear_fit(a, y, selected['penalty'], protocol)
        models['grid_ridge'].update(rates=rates.tolist(), selection=candidates)
    else:
        models['grid_ridge'] = {'success':False, 'selection':candidates, 'seconds':0.}
    candidates = []
    for order in protocol['prony_orders']:
        fit = prony_fit(train, y[train_idx], order, protocol)
        fit['validation_mse'] = float(np.mean((response_basis(valid, fit['best']['rates'])[0]
                @ fit['best']['weights']-y[valid_idx])**2)) if fit['success'] else None
        candidates.append(fit)
    good = [c for c in candidates if c['success']]
    if good:
        selected = min(good, key=lambda c:c['validation_mse'])
        fit = prony_fit(pieces, y, selected['order'], protocol)
        models['prony_cv'] = fit
        if fit['success']: fit.update(weights=fit['best']['weights'], rates=fit['best']['rates'])
        fit['selection'] = candidates
    else:
        models['prony_cv'] = {'success':False, 'selection':candidates, 'seconds':0.}
    return models


def evaluate(model, pieces, y, heldout, bhat, truth, intervals, tolerances):
    if not model['success']: return {'success':False}
    w = np.asarray(model['weights']); rates = np.asarray(model['rates'])
    prediction = response_basis(pieces, rates)[0] @ w
    targets = {'G20':float(np.r_[1., np.exp(-20*rates)] @ w),
               'heldout_sigma20':float(response_basis([(np.array([20.]), heldout)], rates)[0][0] @ w)}
    errors = {t:abs(value-truth[t]) for t, value in targets.items()}
    return {'success':True, 'response_rmse':float(np.sqrt(np.mean((prediction-y)**2))),
            'response_max_residual':float(np.max(np.abs(prediction-y))),
            'fit_band_compatible':bool(np.max(np.abs(prediction-y)) <= bhat+2e-7),
            'predictions':targets, 'absolute_errors':errors,
            'exceeds_tolerance':{t:e>tolerances[t]+2e-7 for t,e in errors.items()},
            'inside_classical_interval':{t:intervals[t][0]-2e-7<=value<=intervals[t][1]+2e-7 for t,value in targets.items()}}


def summarize(records, protocol):
    output = {}
    for family in ['all']+sorted({r['family'] for r in records}):
        for window in protocol['windows']:
            for scope in ('all', 'valid', 'invalid'):
                group = [r for r in records if (family=='all' or r['family']==family)
                         and r['window']==window and (scope=='all' or r['calibration_valid']==(scope=='valid'))]
                if not group: continue
                arms = {}
                for arm in group[0]['evaluation']:
                    good = [r['evaluation'][arm] for r in group if r['evaluation'][arm]['success']]
                    arms[arm] = {'count':len(group), 'success_count':len(good),
                        'failure_count':len(group)-len(good),
                        'joint_exceedance_count':sum(any(g['exceeds_tolerance'].values()) for g in good),
                        'mean_absolute_error':{t:float(np.mean([g['absolute_errors'][t] for g in good])) if good else None for t in protocol['targets']},
                        'maximum_absolute_error':{t:max((g['absolute_errors'][t] for g in good),default=None) for t in protocol['targets']},
                        'exceedance_count':{t:sum(g['exceeds_tolerance'][t] for g in good) for t in protocol['targets']},
                        'mean_response_rmse':float(np.mean([g['response_rmse'] for g in good])) if good and 'response_rmse' in good[0] else None,
                        'band_compatible_count':sum(g.get('fit_band_compatible',False) for g in good),
                        'mean_total_fit_seconds':float(np.mean([r['fit_seconds'].get(arm,0) for r in group]))}
                output[f'{family}|{window}|{scope}'] = arms
    return output


def run(phase):
    protocol = json.loads(SPEC.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / f'{phase}.json'
    if result_path.exists(): raise FileExistsError(result_path)
    parent_path = ROOT/'results'/protocol['parent_result']
    parent = json.loads(parent_path.read_text())
    classical = json.loads((ROOT/'results'/protocol['classical_result']).read_text())
    assert sha(parent_path.with_suffix('.npz')) == parent['witness_archive_sha256']
    assert classical['parent_result_sha256'] == sha(parent_path)
    cspec, original, heldout, pilot, schedules = inputs()
    old = {(r['case'],r['horizon']):r for r in parent['rows']}
    # Inspect schema without guessing the comparator's interval names.
    cr = {(r['case'],r['horizon']):r for r in classical['rows']}
    bindings = {str(p.relative_to(ROOT)):sha(p) for p in [SPEC,Path(__file__),parent_path,
                parent_path.with_suffix('.npz'),ROOT/'results'/protocol['classical_result']]}
    freeze = OUT/f'{phase}_protocol_receipt.json'
    if freeze.exists(): raise FileExistsError(freeze)
    freeze.write_text(json.dumps({'bindings':bindings,'phase':phase,'before_first_fit':True,
        'pid':os.getpid(),'numpy':np.__version__,'scipy':scipy.__version__,
        'scope':protocol['scope']},indent=2)+'\n')
    cases = [c for c in parent['cases'] if phase=='full' or c['seed'] in protocol['pilot_seed_ids_per_family']]
    records = []; started = time.perf_counter()
    with np.load(parent_path.with_suffix('.npz'), allow_pickle=False) as archive:
        for case in cases:
            cid = case['id']
            for hi, window in enumerate(protocol['windows']):
                pieces = pilot+schedules[str(window)]
                y = np.r_[archive[f'case{cid}_pilot'], archive[f'case{cid}_future_{hi}']]
                began = time.perf_counter(); models = fit_models(pieces,y,protocol)
                row = old[cid,window]
                classical_intervals = cr[cid,window]['methods']['classical_sign_convex']
                intervals = {t:classical_intervals[t]['interval'] for t in protocol['targets']}
                truth = parent['truth_targets'][case['truth']]
                evaluation = {name:evaluate(model,pieces,y,heldout,case['calibrated_bound'],truth,
                            intervals,protocol['absolute_point_tolerances']) for name,model in models.items()}
                for name, source in [('mid_symmetric',row['posterior']),('mid_classical',classical_intervals)]:
                    point = {t:float(np.mean(source[t]['interval'])) for t in protocol['targets']}
                    err = {t:abs(point[t]-truth[t]) for t in protocol['targets']}
                    evaluation[name] = {'success':True, 'predictions':point,'absolute_errors':err,
                         'exceeds_tolerance':{t:e>protocol['absolute_point_tolerances'][t]+2e-7 for t,e in err.items()}}
                fit_seconds = {name:fit.get('seconds',0)+sum(c['seconds'] for c in fit.get('selection',[])) for name,fit in models.items()}
                records.append({'case':cid,'family':case['truth'],'seed':case['seed'],'window':window,
                    'calibration_valid':case['calibration_valid'],'calibrated_bound':case['calibrated_bound'],
                    'models':models,'evaluation':evaluation,'fit_seconds':fit_seconds,
                    'seconds':time.perf_counter()-began})
            print(json.dumps({'phase':phase,'completed_cases':len(records)//2,'case':cid,
                    'seconds':round(time.perf_counter()-started,2)}),flush=True)
            # Checkpoint is a recoverable partial result, never a complete-study receipt.
            (OUT/f'{phase}_checkpoint.json').write_text(json.dumps({'complete':False,'records':records}))
    result = {'schema':'P4-direct-fitting-results-v1','phase':phase,'complete':True,
        'case_count':len(cases),'window_records':len(records),'new_random_trials':False,
        'new_fitting_executions':True,'bindings':bindings,'duration_seconds':time.perf_counter()-started,
        'summaries':summarize(records,protocol),'records':records,'limits':protocol['limits']}
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':True,'phase':phase,'cases':len(cases),'seconds':result['duration_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['pilot','full'],required=True)
    run(parser.parse_args().phase)
