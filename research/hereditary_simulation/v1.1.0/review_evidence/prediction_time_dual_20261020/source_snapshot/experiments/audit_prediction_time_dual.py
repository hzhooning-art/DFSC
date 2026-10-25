"""Replay all LPs and independently recompute continuous bounds at 80 digits."""
from pathlib import Path
import hashlib
import json
import numpy as np
import mpmath as mp
from prediction_time_search import Information, representation, decision

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/prediction_time_dual_20261020/run_01'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    mp.mp.dps = 80
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    receipt = json.loads((OUT/'protocol_receipt.json').read_text(encoding='utf-8'))
    assert sha(OUT/'protocol_receipt.json') == result['protocol_receipt_sha256']
    assert all(sha(ROOT/p) == h for p,h in receipt['bindings'].items())
    assert sha(OUT/'lp_witnesses.npz') == result['lp_witnesses_sha256']
    spec = json.loads((ROOT/'experiments/specs/prediction_time_dual_20261020.json').read_text())
    arrays = np.load(OUT/'lp_witnesses.npz')
    m = lambda x: mp.mpf(float(x))  # Exact conversion of stored binary inputs.
    max_lp_residual, max_physical_violation = 0., mp.mpf(0)
    replayed = physical_checks = leaf_checks = tail_checks = 0
    certificate_gaps = []
    bracket_map = {}
    for item in result['brackets']:
        beta, wait, level, label = item['beta'],item['wait'],item['level'],item['label']
        info = Information(beta,.1,tuple(spec['observation_delays_from_anchor']),
                           tuple(item['centers']),(.01,)*5)
        rhs = np.r_[np.array(info.centers)+info.bands,-np.array(info.centers)+info.bands]
        rep = representation(info,beta,wait,level)
        for kind in ('uniform','dual'):
            sides = ('ol','oh','il','ih') if kind == 'uniform' else ('il','ih')
            for side in sides:
                stem = label+'_'+kind+'_'+side
                if kind == 'uniform':
                    a = rep['outer_a'] if side.startswith('o') else rep['inner_a']
                    c = rep['q_low'] if side=='ol' else -rep['q_high'] if side=='oh' else rep['inner_q']*(1 if side=='il' else -1)
                else:
                    nodes = arrays[stem+'_nodes']
                    moments = np.exp(-np.array(info.delays)[:,None]*nodes)
                    a = np.vstack((moments,-moments))
                    c = np.exp(-(wait-.1)*nodes)*(1 if side=='il' else -1)
                x,y,z = [arrays[stem+'_'+k] for k in ('x','y','z')]
                residual = max(0.,float(np.max(a@x-rhs)),float(-np.min(x)),float(np.max(np.abs(c-a.T@y-z))),
                    float(np.max(y)),float(-np.min(z)),float(abs(c@x-rhs@y)))
                assert residual <= spec['numerical_margin'], (stem,residual)
                max_lp_residual = max(max_lp_residual,residual)
                replayed += 1
                if kind == 'dual':
                    nodes, weights = arrays[stem+'_physical_nodes'],arrays[stem+'_physical_weights']
                    assert np.isfinite(nodes).all() and (nodes >= 0).all() and (weights > 0).all()
                    for d,center,band in zip(info.delays,info.centers,info.bands):
                        response = mp.fsum(m(w)*mp.exp(-m(d)*m(r)) for r,w in zip(nodes,weights))
                        violation = abs(response-m(center))-m(band)
                        assert violation <= mp.mpf('1e-25'), (stem,str(violation))
                        max_physical_violation = max(max_physical_violation,violation)
                    q = mp.fsum(m(w)*mp.exp(-(m(wait)-m(.1))*m(r)) for r,w in zip(nodes,weights))
                    reported = item['continuous_dual']['physical_witnesses'][side]['target']
                    assert abs(q-m(reported)) < mp.mpf('2e-12')
                    # Reconstruct the original physical mass in log form.
                    for rate,weight,stored in zip(nodes,weights,arrays[stem+'_log_physical_masses']):
                        r = m(rate)
                        phi = mp.gammainc(m(beta),0,r)/r**m(beta) if r else 1/m(beta)
                        logmass = mp.log(m(weight))+m(.1)*r-mp.log(phi)
                        assert abs(logmass-m(stored)) <= mp.mpf('2e-10')*(1+abs(logmass))
                    physical_checks += 1
        for endpoint, cert in item['continuous_dual']['certificates'].items():
            d, a = list(map(m,cert['rates'])), list(map(m,cert['coefficients']))
            def ev(x):
                return [mp.exp(-v*x) for v in d]
            def value(x):
                return mp.fsum(c*e for c,e in zip(a,ev(x)))
            upper_bound = -mp.inf
            last = mp.mpf(0)
            for l,r,stored in cert['leaves']:
                lo,hi = m(l),m(r)
                assert lo == last and hi > lo
                last = hi
                el,er = ev(lo),ev(hi)
                second_lower = mp.fsum(c*v*v*(eh if c>=0 else eh0) for c,v,eh,eh0 in zip(a,d,er,el))
                chord = max(value(lo),value(hi))+max(mp.mpf(0),-second_lower)*(hi-lo)**2/8
                natural = mp.fsum(c*(e0 if c>=0 else e1) for c,e0,e1 in zip(a,el,er))
                exact_bound = min(chord,natural)
                assert m(stored) >= exact_bound-mp.mpf('1e-25'), (label,endpoint,l,r)
                upper_bound = max(upper_bound,exact_bound)
                leaf_checks += 1
            assert last == m(cert['tail_start'])
            tail = mp.fsum(c for c,v in zip(a,d) if not v)+mp.fsum(c*mp.exp(-v*last) for c,v in zip(a,d) if v>0 and c>0)
            assert m(cert['tail_upper']) >= tail-mp.mpf('1e-25')
            upper_bound = max(upper_bound,tail)
            assert m(cert['upper']) >= upper_bound-mp.mpf('1e-25')
            # Check the saved residual coefficients actually derive from these duals.
            side = 'il' if endpoint=='lower' else 'ih'
            y = np.minimum(arrays[label+'_dual_'+side+'_y'],0.)
            c = y[:5]-y[5:] if endpoint=='lower' else -y[:5]+y[5:]
            expected_rates = np.r_[wait-.1,info.delays]
            expected_coeff = np.r_[-1.,c] if endpoint=='lower' else np.r_[1.,-c]
            for rate,coefficient in zip(cert['rates'],cert['coefficients']):
                assert abs(coefficient-expected_coeff[expected_rates==rate].sum()) < 1e-14
            certificate_gaps.append(cert['gap'])
            tail_checks += 1
        bracket_map[item['scenario'],wait,level] = item
    states = ('unbounded','insufficient','sufficient','unresolved')
    exact_costs = {}
    for cohort in result['summary']:
        exact_costs[cohort] = {a:0 for a in spec['algorithms']}
    for row in result['rows']:
        for algorithm, answer in row['algorithms'].items():
            if answer['state']=='unbounded':
                continue
            b = bracket_map[row['scenario'],row['wait'],answer['levels_used'][-1]][algorithm]
            actual_decision = decision(b,.025,.05,row['rho'])
            assert actual_decision['state'] == answer['state']
            if answer['state']=='sufficient':
                assert abs(actual_decision['joint_prediction']-answer['joint_prediction']) < 1e-12
            for level in answer['levels_used']:
                item = bracket_map[row['scenario'],row['wait'],level]
                variables = (4*item['uniform_interval']['cells'] if algorithm=='uniform_interval' else
                    2*item['uniform_interval']['cells']+2*item['continuous_dual']['cells'] if algorithm=='tail_completed_interval' else
                    2*item['continuous_dual']['cells'])
                exact_costs[row['cohort']][algorithm] += variables
    for check in result['positive_truth_checks']:
        assert check['enclosed'] and check['absolute_error'] <= .025 and check['relative_error'] <= .05
    assert not result['opposite_decisive_labels'] and not result['lost_resolved'] and not result['numerical_failures']
    parent = json.loads((ROOT/spec['parent_result']).read_text())
    opposite_parent, lost_parent = [], []
    patch_lp_calls = patch_variables = patch_residual_cells = 0
    for row in result['rows']:
        if row['cohort'] != 'parent' or row.get('parent_state') is None:
            continue
        old,new = row['parent_state'],row['algorithms']['continuous_dual']['state']
        if old in ('sufficient','insufficient') and new in ('sufficient','insufficient') and old != new:
            opposite_parent.append(row)
        if old != 'unresolved' and new == 'unresolved':
            lost_parent.append(row)
        if old == 'unresolved':
            v = row['algorithms']['continuous_dual']
            patch_lp_calls += v['lp_calls']
            patch_variables += v['cell_variables']
            patch_residual_cells += v['residual_cells']
    assert not opposite_parent and not lost_parent
    summary = {'all_checks_passed':True,'result_sha256':sha(OUT/'result.json'),
        'protocol_bindings_match':True,'lp_witnesses_sha256':sha(OUT/'lp_witnesses.npz'),
        'lp_replays':replayed,'max_lp_residual':max_lp_residual,
        'physical_witnesses_at_80_digits':physical_checks,'max_physical_band_violation_at_80_digits':str(max_physical_violation),
        'continuous_finite_cell_bounds_at_80_digits':leaf_checks,'infinite_tail_bounds_at_80_digits':tail_checks,
        'maximum_residual_certificate_gap':max(certificate_gaps),
        'positive_joint_prediction_checks':len(result['positive_truth_checks']),
        'exact_cell_variable_counts':exact_costs,
        'cost_accounting_note':'Runner tail-completed-interval cell_variables conservatively counts both outer LPs at the augmented inner node count; exact mixed sizes are given here.',
        'opposite_parent_labels':0,'lost_parent_resolved':0,
        'parent_patch_cost':{'additional_endpoint_LPs':patch_lp_calls,'additional_LP_variables':patch_variables,
            'additional_residual_cells':patch_residual_cells,
            'total_endpoint_LPs_with_prior_secant':parent['summary']['65']['adaptive_secant_transport']['lp_calls']+patch_lp_calls},
        'limits':['Same-developer high-precision replay, not independent external review',
            'Analytic interval remainder bounds checked at 80 digits; solver and production computations are floating point, not directed rounding']}
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (OUT/'source_snapshot/experiments/audit_prediction_time_dual.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
