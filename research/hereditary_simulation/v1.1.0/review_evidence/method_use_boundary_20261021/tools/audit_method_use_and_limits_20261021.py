"""Retrospective use/boundary audit; reads frozen evidence, runs no new LPs."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from prediction_time_search import Information
from prediction_time_polish import polish_unresolved


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    paths = {'parent':ROOT/'results/prediction_time_transport_20261020/run_02/result.json',
             'dual':ROOT/'results/prediction_time_dual_20261020/run_01/result.json',
             'dual_audit':ROOT/'results/prediction_time_dual_20261020/run_01/audit.json'}
    parent, dual = [json.loads(paths[k].read_text()) for k in ('parent','dual')]
    brackets = {(b['scenario'],b['wait'],b['level']):b['continuous_dual'] for b in dual['brackets']}
    rows, gains = [], []
    states = Counter()
    decisive_unchanged = unbounded_unchanged = 0
    for row in parent['rows']:
        if row['density']!='65':
            continue
        info = Information(row['beta'],.1,(0.,.5,1.,2.,4.),tuple(row['centers']),(.01,)*5)
        name = row['truth']+'_p0_same'
        def oracle(wait,level):
            return brackets[name,wait,level]
        old = row['algorithms']['adaptive_secant_transport']
        new = polish_unresolved(info,row['gamma'],old,.025,.05,row['rho'],[128,512,2048],oracle,row['strict_witness'])
        states.update(r['state'] for r in new['rows'])
        for a,b in zip(old['rows'],new['rows']):
            if a['state'] in ('sufficient','insufficient'):
                assert a==b
                decisive_unchanged += 1
            if a['state']=='unbounded':
                assert a==b
                unbounded_unchanged += 1
            if a['state']=='unresolved' and b['state']!='unresolved':
                gains.append({'scenario':row['scenario'],'rho':row['rho'],'wait':b['wait'],
                    'new_state':b['state'],'outer':b['outer'],'inner':b['inner'],
                    'absolute_radius_bracket':b['absolute_radius_bracket'],
                    'relative_radius_bracket':b['relative_radius_bracket'],
                    'joint_prediction':b['joint_prediction'],'dual_levels':b['dual_levels_used']})
        rows.append({'scenario':row['scenario'],'rho':row['rho'],
            'old_first_sufficient':old['first_sufficient_candidate'],'new_first_sufficient':new['first_sufficient_candidate'],
            'old_earliest_list_proved':old['earliest_candidate_proved'],'new_earliest_list_proved':new['earliest_candidate_proved'],
            'extension':new['continuous_dual_extension']})
        if row['scenario']=='two_rate_p0_g1' and row['rho']==0:
            example = {str(t):next(r for r in new['rows'] if r['wait']==t) for t in (.1,.101,.11567690878,.13)}
    first_changes = [r for r in rows if r['old_first_sufficient']!=r['new_first_sufficient']]
    proof_changes = [r for r in rows if r['old_earliest_list_proved']!=r['new_earliest_list_proved']]
    positive = [g for g in gains if g['new_state']=='sufficient']
    groups = Counter((g['scenario'],g['rho']) for g in positive)
    assert len(gains)==76 and len(positive)==7 and len(groups)==1
    assert not first_changes and not proof_changes
    assert example['0.1']['state']=='sufficient' and example['0.101']['state']=='insufficient'
    assert example['0.11567690878']['state']=='sufficient'
    assert dict(states)==dual['parent_portfolio_after_unresolved_only_patch']
    result = {'schema':'P4-method-use-boundary-audit-v1','date':'2026-10-21',
        'analysis_type':'retrospective aggregation of existing frozen records; no new optimization or observations',
        'bindings':{k:{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for k,p in paths.items()},
        'analysis_source_sha256':sha(Path(__file__)),
        'workflow_source_sha256':sha(ROOT/'experiments/prediction_time_polish.py'),
        'workflow_cases':len(rows),'requests':sum(states.values()),'portfolio_states':dict(states),
        'unchanged_prior_decisive_rows':decisive_unchanged,'unchanged_prior_unbounded_rows':unbounded_unchanged,
        'new_decisions':dict(Counter(g['new_state'] for g in gains)),
        'new_positive_groups':[{'scenario':k[0],'rho':k[1],'count':v} for k,v in groups.items()],
        'earliest_list_time_changes':first_changes,'earliest_list_proof_changes':proof_changes,
        'example':example,'cases':rows,'gains':gains,
        'supported_use':'Target-specific admissibility and decision-map completion with reusable endpoint computation',
        'unsupported_use':['Earlier first feasible candidate from the dual extension in this benchmark',
            'Monotone threshold or permanent sufficiency at all later target times',
            'Generic novelty of positivity-based target bounds or residual correction',
            'Improved point prediction accuracy, measured runtime, physical-material validation',
            'Applicability of uncapped fast-atom obstructions under supplied common physical caps'],
        'all_checks_passed':True}
    output = ROOT/'results/p4_method_use_boundary_audit_20261021.json'
    if output.exists():
        raise FileExistsError('Use a new audit revision; do not overwrite historical evidence')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('workflow_cases','requests','portfolio_states','new_decisions',
        'new_positive_groups','earliest_list_time_changes','earliest_list_proof_changes','all_checks_passed')},indent=2))


if __name__=='__main__':
    main()
