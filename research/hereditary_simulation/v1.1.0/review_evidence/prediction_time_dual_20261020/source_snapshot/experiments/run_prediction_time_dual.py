"""Freeze and run a matched full-halfline dual / fast-witness ablation."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import scipy
from observation_weighted_prediction import integrate_truth
from prediction_time_search import Information, log_laplace, strict_witness, enclose, decision, bounded
from prediction_time_dual import enclose_dual

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT/'experiments/specs/prediction_time_dual_20261020.json'
OUT = ROOT/'results/prediction_time_dual_20261020/run_01'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    def native(x):
        return x.item() if isinstance(x, np.generic) else x.tolist()
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                               default=native)+'\n', encoding='utf-8')


def main():
    if OUT.exists():
        raise FileExistsError('Choose a fresh run; frozen results are never overwritten')
    OUT.mkdir(parents=True)
    spec = json.loads(SPEC.read_text(encoding='utf-8'))
    parent_path = ROOT/spec['parent_result']
    truth_path = SPEC.parent/spec['truth_spec']
    bindings = [SPEC, Path(__file__), ROOT/'experiments/prediction_time_dual.py',
        ROOT/'experiments/prediction_time_search.py', ROOT/'experiments/observation_weighted_prediction.py',
        ROOT/'tests/test_prediction_time_dual.py', truth_path, parent_path]
    put(OUT/'protocol_receipt.json', {'bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in bindings},
        'frozen_before_execution':True, 'scope':spec['scope'],
        'versions':{'numpy':np.__version__, 'scipy':scipy.__version__}})
    snapshot = OUT/'source_snapshot'
    for p in bindings:
        # The large parent result remains a hash-bound input, not a source copy.
        if p == parent_path:
            continue
        target = snapshot/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    truths = json.loads(truth_path.read_text(encoding='utf-8'))['truths']
    archive, logs, rows, bracket_records, failures, truthchecks = {}, [], [], [], [], []
    start = time.perf_counter()
    for truth in truths:
        for p in spec['reference_orders']:
            beta, dead = float(p+1), spec['reference_dead_time']
            delays = tuple(spec['observation_delays_from_anchor'])
            centers = tuple(integrate_truth(truth, lambda x:float(np.exp(-(dead+d)*x+log_laplace(x,beta)))) for d in delays)
            info = Information(beta,dead,delays,centers,(spec['observation_half_width'],)*len(delays))
            feasible = strict_witness(info)
            assert feasible is not None
            name = f"{truth['name']}_p{p}_same"
            cases = [('additional', round(dead+e,12), spec['additional_rhos']) for e in spec['additional_excess_delays']]
            if p == 0:
                cases += [('parent', w, spec['parent_rhos']) for w in parent['grids']['65']]
            cache = {}
            for cohort, wait, rhos in cases:
                if not bounded(info,beta,wait):
                    for rho in rhos:
                        rows.append({'cohort':cohort,'scenario':name,'beta':beta,'wait':wait,'rho':rho,
                            'algorithms':{a:{'state':'unbounded','lp_calls':0,'cell_variables':0,'residual_cells':0} for a in spec['algorithms']}})
                    continue
                actual = integrate_truth(truth, lambda x:float(np.exp(-wait*x+log_laplace(x,beta))))
                def get(level):
                    key = (wait,level)
                    if key not in cache:
                        label = f'{name}_w{wait}_c{level}'
                        try:
                            interval = enclose(info,beta,wait,level,label+'_uniform',archive,logs)
                            dual = enclose_dual(info,beta,wait,level,label+'_dual',archive,logs,
                                feasible=feasible,tail_exponent=spec['tail_exponent'],
                                residual_tolerance=spec['residual_tolerance'],max_certificate_cells=spec['max_certificate_cells'])
                            cache[key] = {'uniform_interval':interval,
                                'tail_completed_interval':{**interval,'inner':dual['inner'],'cells':dual['cells']},
                                'continuous_dual':dual}
                            bracket_records.append({'scenario':name,'beta':beta,'wait':wait,'level':level,
                                'label':label, 'centers':centers,'feasible':feasible,**cache[key]})
                        except RuntimeError as e:
                            failures.append({'scenario':name,'wait':wait,'level':level,'error':str(e)})
                            cache[key] = None
                    return cache[key]
                for rho in rhos:
                    algorithms = {}
                    for algorithm in spec['algorithms']:
                        lp_calls = variables = residual_cells = 0
                        answer = {'state':'unresolved'}
                        used = []
                        for level in spec['mesh_levels']:
                            values = get(level)
                            used.append(level)
                            if values is None:
                                # Preserve failure as unresolved; do not make a prediction.
                                break
                            bracket = values[algorithm]
                            count = 2 if algorithm == 'continuous_dual' else 4
                            lp_calls += count
                            variables += count*bracket['cells']
                            if algorithm == 'continuous_dual':
                                residual_cells += sum(c['cells'] for c in bracket['certificates'].values())
                            answer = decision(bracket,spec['absolute_error_tolerance'],spec['relative_error_tolerance'],rho)
                            if answer['state'] != 'unresolved':
                                break
                        algorithms[algorithm] = {**answer,'levels_used':used,'lp_calls':lp_calls,
                            'cell_variables':variables,'residual_cells':residual_cells}
                        if answer['state'] == 'sufficient':
                            for scale in (1.,1/(1+rho),1/(1-rho)):
                                value, prediction = actual*scale, answer['joint_prediction']
                                truthchecks.append({'cohort':cohort,'scenario':name,'wait':wait,'rho':rho,
                                    'algorithm':algorithm,'scale':scale,'value':value,'prediction':prediction,
                                    'absolute_error':abs(value-prediction),'relative_error':abs(value-prediction)/value,
                                    'enclosed':answer['outer'][0] <= value <= answer['outer'][1]})
                    old = None
                    if cohort == 'parent':
                        source = next(r for r in parent['rows'] if r['density']=='65' and r['scenario']==f"{truth['name']}_p0_g1" and r['rho']==rho)
                        old = next(r for r in source['algorithms']['adaptive_secant_transport']['rows'] if r['wait']==wait)['state']
                    rows.append({'cohort':cohort,'scenario':name,'beta':beta,'wait':wait,'rho':rho,
                                 'parent_state':old,'algorithms':algorithms})
            print(json.dumps({'scenario':name,'unique_LPs':len(logs),'failures':len(failures)}),flush=True)
    summary, opposites, lost, transitions = {}, [], [], []
    for cohort in ('parent','additional'):
        selected = [r for r in rows if r['cohort']==cohort]
        summary[cohort] = {}
        for algorithm in spec['algorithms']:
            values = [r['algorithms'][algorithm] for r in selected]
            summary[cohort][algorithm] = {'requests':len(values),
                'states':{s:sum(v['state']==s for v in values) for s in ('unbounded','insufficient','sufficient','unresolved')},
                **{k:sum(v[k] for v in values) for k in ('lp_calls','cell_variables','residual_cells')}}
        for row in selected:
            old = row['algorithms']['uniform_interval']['state']
            for algorithm in spec['algorithms'][1:]:
                new = row['algorithms'][algorithm]['state']
                item = {k:row[k] for k in ('cohort','scenario','wait','rho')}
                if old in ('sufficient','insufficient') and new in ('sufficient','insufficient') and old != new:
                    opposites.append({**item,'algorithm':algorithm,'old':old,'new':new})
                if old != 'unresolved' and new == 'unresolved':
                    lost.append({**item,'algorithm':algorithm,'old':old})
            if row.get('parent_state') == 'unresolved':
                transitions.append({**{k:row[k] for k in ('scenario','wait','rho')},
                    'new':row['algorithms']['continuous_dual']['state']})
    portfolio = dict(parent['summary']['65']['adaptive_secant_transport']['states'])
    for t in transitions:
        if t['new'] != 'unresolved':
            portfolio['unresolved'] -= 1
            portfolio[t['new']] += 1
    np.savez_compressed(OUT/'lp_witnesses.npz',**archive)
    result = {'schema':'P4-continuous-dual-result-v1','spec_sha256':sha(SPEC),
        'protocol_receipt_sha256':sha(OUT/'protocol_receipt.json'),'rows':rows,'brackets':bracket_records,
        'summary':summary,'opposite_decisive_labels':opposites,'lost_resolved':lost,
        'parent_unresolved_transitions':transitions,'parent_portfolio_after_unresolved_only_patch':portfolio,
        'positive_truth_checks':truthchecks,'numerical_failures':failures,'lp_logs':logs,
        'lp_witnesses_sha256':sha(OUT/'lp_witnesses.npz'),
        'elapsed_seconds_with_shared_cache':time.perf_counter()-start,'limitations':spec['limitations']}
    put(OUT/'result.json',result)
    print(json.dumps({'summary':summary,'portfolio':portfolio,'opposites':len(opposites),'lost':len(lost),
                      'transitions':{s:sum(t['new']==s for t in transitions) for s in ('sufficient','insufficient','unresolved')}}),flush=True)


if __name__ == '__main__':
    main()
