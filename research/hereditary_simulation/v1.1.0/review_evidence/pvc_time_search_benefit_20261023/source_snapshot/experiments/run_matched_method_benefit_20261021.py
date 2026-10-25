"""Matched classical continuous positive-measure oracle; only scheduling differs.

This is our adaptation of Jay--Saccardi III.2--3, not an execution or a speed
claim about their software. The shared engine solves both moment LP endpoints,
adds the identical fast witness node and checks dual residuals on [0,infinity).
No finite rate/mass cap, weaker baseline, generating truth or cross-run cache.
"""
from pathlib import Path
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
import platform
import time
import numpy as np
import scipy
from scipy.optimize import linprog
from prediction_time_search import Information, strict_witness, bounded, decision, log_laplace
from prediction_time_dual import enclose_dual
from prediction_time_transport import transport_bracket, transferred_decision
from observation_weighted_prediction import integrate_truth

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT/'experiments/specs/matched_method_benefit_20261021.json'
OUT = ROOT/'results/matched_method_benefit_20261021/run_01'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(p, v):
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())+'\n', encoding='utf-8')


def workflow(info, waits, rho, spec, reuse, prefix, retain):
    """Deterministic endpoints then median, with anytime completed-check trace."""
    archive, logs, brackets, failures = {}, [], {}, []
    start = time.perf_counter()
    feasible = strict_witness(info)
    if feasible is None:
        raise RuntimeError('Strict feasibility unresolved; experiment cannot assert a result')
    feasibility_seconds = time.perf_counter()-start
    rows = {w: {'wait': w, 'state': 'unresolved', 'route': 'pending'} for w in waits}
    cache, evaluated, trace, cells = {}, set(), [], []
    endpoint_programs = solver_calls = 0
    for w in waits:
        if not bounded(info, info.beta, w):
            rows[w].update(state='unbounded', route='phase')
    def snapshot():
        trace.append({'endpoint_programs': endpoint_programs, 'solver_calls': solver_calls,
            'elapsed_seconds': time.perf_counter()-start,
            'states': {s: sum(r['state']==s for r in rows.values())
                       for s in ('unbounded', 'insufficient', 'sufficient', 'unresolved')}})
    def transport():
        if not reuse:
            return
        for left, right in zip(sorted(cache), sorted(cache)[1:]):
            pending = [w for w in waits if left < w < right and rows[w]['state']=='unresolved']
            if not pending:
                continue
            whole = transferred_decision(transport_bracket(cache[left], cache[right], left, right),
                spec['absolute'], spec['relative'], rho)
            if whole['state'] != 'unresolved':
                cells.append({'start':left,'end':right,'state':whole['state']})
            for w in pending:
                answer = whole if whole['state']!='unresolved' else transferred_decision(
                    transport_bracket(cache[left],cache[right],left,right,at=w),
                    spec['absolute'],spec['relative'],rho)
                if answer['state']!='unresolved':
                    rows[w].update(answer,route='time_cell' if whole['state']!='unresolved' else 'time_secant',
                                   source=[left,right])
    def evaluate(w):
        nonlocal endpoint_programs, solver_calls
        evaluated.add(w)
        for level in spec['levels']:
            label = f'{prefix}_w{w}_c{level}'
            try:
                bracket = enclose_dual(info,info.beta,w,level,label,archive,logs,feasible=feasible,
                    tail_exponent=spec['tail_exponent'],residual_tolerance=spec['residual_tolerance'],
                    max_certificate_cells=spec['max_certificate_cells'])
            except RuntimeError as error:
                endpoint_programs += getattr(error,'programs',0)
                solver_calls += getattr(error,'solver_calls',0)
                failures.append({'wait':w,'level':level,'message':str(error)})
                snapshot()
                break
            endpoint_programs += bracket['programs']
            solver_calls += bracket['solver_calls']
            cache[w] = bracket
            brackets[label] = bracket
            answer = decision(bracket,spec['absolute'],spec['relative'],rho)
            rows[w].update(answer,route='direct',level=level,label=label)
            transport()
            snapshot()
            if rows[w]['state']!='unresolved':
                break
    snapshot()
    active = [w for w in waits if rows[w]['route']=='pending']
    if active:
        evaluate(active[0])
        if active[-1]!=active[0]:
            evaluate(active[-1])
    while True:
        pending = [w for w in active if w not in evaluated and rows[w]['state']=='unresolved']
        if not pending:
            break
        evaluate(pending[len(pending)//2])
    total_seconds = time.perf_counter()-start
    passing = [w for w in waits if rows[w]['state']=='sufficient']
    result = {'rows':list(rows.values()), 'trace':trace, 'endpoint_programs':endpoint_programs,
        'solver_calls':solver_calls,'total_seconds':total_seconds,'feasibility_seconds':feasibility_seconds,
        'evaluated_times':sorted(evaluated),'whole_time_cells':cells,'failures':failures,
        'states':trace[-1]['states'], 'first_sufficient':min(passing) if passing else None}
    if retain:
        result.update(brackets=brackets,lp_logs=logs,strict_witness=feasible)
    return result, archive if retain else None


def main():
    if OUT.exists():
        raise FileExistsError('Frozen runs must not be overwritten')
    OUT.mkdir(parents=True)
    spec = json.loads(SPEC.read_text(encoding='utf-8'))
    parent = ROOT/spec['parent_result']
    truthfile = ROOT/spec['truth_spec']
    paths = [SPEC,Path(__file__),parent,truthfile] + [ROOT/'experiments'/f for f in (
        'prediction_time_search.py','prediction_time_transport.py','prediction_time_dual.py',
        'observation_weighted_prediction.py')]
    bindings = {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    put(OUT/'protocol_receipt.json',{'frozen_before_execution':True,'bindings':bindings,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'machine':{'system':platform.platform(),'processor':platform.processor(),'cpu_count':os.cpu_count()},
        'thread_environment':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},
        'sources':['https://arxiv.org/html/2606.19503v1#S3.SS2','https://arxiv.org/html/2606.19503v1#S3.SS3'],
        'interpretation':'Own matched adaptation, not authors original software. Classical continuous LP oracle is shared; only temporal scheduling changes.'})
    snap = OUT/'source_snapshot'
    for p in paths:
        if p==parent:
            continue
        target = snap/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    waits = json.loads(parent.read_text(encoding='utf-8'))['grids']['65']
    truths = json.loads(truthfile.read_text(encoding='utf-8'))['truths']
    rows=[]
    for repetition in range(spec['repetitions']):
        order = ('pointwise_classical','loading_guided_reuse') if repetition%2==0 else ('loading_guided_reuse','pointwise_classical')
        for truth in truths:
            centers = tuple(integrate_truth(truth,lambda x:float(np.exp(-(.1+d)*x+log_laplace(x,1.))))
                            for d in spec['observation_delays'])
            info=Information(1.,.1,tuple(spec['observation_delays']),centers,(.01,)*len(centers))
            for rho in spec['rhos']:
                for name in order:
                    # Identical untimed warm-up, no information or bracket cache is carried over.
                    linprog([1.],A_ub=[[-1.]],b_ub=[-1.],bounds=(0,None),method='highs-ds')
                    prefix=f'{truth["name"]}_r{rho}_{name}'
                    result,archive=workflow(info,waits,rho,spec,name=='loading_guided_reuse',prefix,repetition==0)
                    if archive is not None:
                        np.savez_compressed(OUT/f'{prefix}_witnesses.npz',**archive)
                    if repetition==0:
                        truthchecks=[]
                        for r in result['rows']:
                            if r['state']!='sufficient':continue
                            actual=integrate_truth(truth,lambda x:float(np.exp(-r['wait']*x+log_laplace(x,1.))))
                            for scale in (1.,1/(1+rho),1/(1-rho)):
                                value=actual*scale;z=r['joint_prediction']
                                truthchecks.append({'wait':r['wait'],'scale':scale,'value':value,
                                    'absolute_error':abs(z-value),'relative_error':abs(z-value)/value,
                                    'enclosed':r['outer'][0]<=value<=r['outer'][1]})
                        result['truth_checks']=truthchecks
                        put(OUT/f'{prefix}.json',result)
                    rows.append({'repetition':repetition,'scenario':truth['name'],'rho':rho,'algorithm':name,
                        **{k:result[k] for k in ('endpoint_programs','solver_calls','total_seconds','feasibility_seconds',
                                              'states','first_sufficient','trace','failures')},
                        'decisions':[r['state'] for r in result['rows']]})
                    put(OUT/'progress.json',{'completed_workflows':len(rows),'last':{k:rows[-1][k] for k in (
                        'repetition','scenario','rho','algorithm','endpoint_programs','total_seconds','states')}})
                    print(json.dumps(rows[-1]|{'trace':'omitted','decisions':'omitted'}),flush=True)
    assert all(sha(ROOT/k)==v for k,v in bindings.items())
    put(OUT/'result.json',{'schema':'P4-matched-method-benefit-v1','spec_sha256':sha(SPEC),
        'rows':rows,'waits':waits,'limitations':spec['limitations'],
        'protocol_receipt_sha256':sha(OUT/'protocol_receipt.json')})


if __name__=='__main__':
    main()
