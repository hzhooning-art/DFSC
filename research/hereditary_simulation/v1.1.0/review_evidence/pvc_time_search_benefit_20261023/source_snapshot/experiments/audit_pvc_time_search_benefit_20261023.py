"""Audit frozen PVC task accounting, physical spectra, full-rate residuals and reuse."""
from pathlib import Path
import hashlib,json,re,statistics
import numpy as np
import mpmath as mp
from prediction_time_search import decision
from prediction_time_transport import transport_bracket,transferred_decision

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/pvc_time_search_benefit_20261023'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    mp.mp.dps=80;m=lambda x:mp.mpf(float(x))
    spec=json.loads((ROOT/'experiments/specs/pvc_time_search_benefit_20261023.json').read_text())
    parent=json.loads((ROOT/spec['parent_spec']).read_text());raw=json.loads((ROOT/spec['parent_data']).read_text())
    result=json.loads((OUT/'result.json').read_text());receipt=json.loads((OUT/'protocol_receipt.json').read_text())
    cases=json.loads((OUT/'cases.json').read_text());assert len(cases)==30
    assert sha(OUT/'protocol_receipt.json')==result['protocol_sha256']
    assert sha(ROOT/'experiments/specs/pvc_time_search_benefit_20261023.json')==result['spec_sha256']
    assert all(sha(ROOT/rel)==digest for rel,digest in receipt['bindings'].items())
    # Reconstruct every case from the previously source-audited measured values.
    for case in cases:
        tr=next(x for x in raw['traces'] if x['temperature_K']==case['temperature_K'])
        t=np.array([x['time_seconds'] for x in tr['records']]);y=np.array([x['modulus_MPa'] for x in tr['records']])
        span=t[-1]-t[0]
        obs=[int(np.argmin(abs(t-(t[0]+case['window_fraction']*f*span)))) for f in parent['observation_within_window_fractions']]
        indices=np.flatnonzero(t>max(t[obs]))
        assert np.array_equal(t[indices],case['actual_seconds'])
        assert np.array_equal((t[indices]-t[0])/span,case['waits'])
        assert np.array_equal((t[obs]-t[0])/span,case['delays'])
        assert np.array_equal(y[obs]/y[0],case['centers'])
        assert np.array_equal(y[indices]/y[0],case['measured_targets_scaled'])
        assert case['window_fraction'] in parent['observation_window_fractions']
        assert case['band_fraction'] in parent['assumed_relative_to_first_measured_modulus_bands']
    counts={'lp':0,'physical':0,'finite_cells':0,'tails':0,'transport_decisions':0,'positive_raw_target_checks':0}
    maxres=0.;file_hashes={};loaded={};positives=[];all_coverage=[]
    for case in cases:
        centers=np.array(case['centers']);delays=np.array(case['delays']);band=case['band_fraction']
        rhs=np.r_[centers+band,-centers+band]
        for name in ('pointwise_classical','loading_guided_reuse'):
            path=OUT/(case['id']+'_'+name+'.json');arraypath=path.with_suffix('.npz')
            item=json.loads(path.read_text());arrays=np.load(arraypath);loaded[(case['id'],name)]=item
            file_hashes[path.name]=sha(path);file_hashes[arraypath.name]=sha(arraypath)
            assert len(item['rows'])==len(case['waits']) and not item['failures']
            latest={}
            for label,b in item['brackets'].items():
                parsed=re.search(r'_w([^_]+)_c([0-9]+)$',label);assert parsed
                wait=float(parsed[1]);latest[wait]=b
                assert wait in case['waits'] and int(parsed[2]) in spec['levels']
                for side in ('il','ih'):
                    stem=label+'_'+side
                    nodes,x,dual,z=[arrays[stem+'_'+k] for k in ('nodes','x','y','z')]
                    E=np.exp(-delays[:,None]*nodes);A=np.vstack((E,-E))
                    c=np.exp(-wait*nodes)*(1 if side=='il' else -1)
                    residual=max(0.,float(np.max(A@x-rhs)),float(-np.min(x)),float(np.max(abs(c-A.T@dual-z))),
                        float(np.max(dual)),float(-np.min(z)),float(abs(c@x-rhs@dual)))
                    assert residual<=2e-7,(label,residual)
                    maxres=max(maxres,residual);counts['lp']+=1
                    pn,pw=arrays[stem+'_physical_nodes'],arrays[stem+'_physical_weights']
                    assert (pn>=0).all() and (pw>0).all() and np.isfinite(pn).all() and np.isfinite(pw).all()
                    for d,center in zip(delays,centers):
                        val=mp.fsum(m(w)*mp.exp(-m(d)*m(rate)) for rate,w in zip(pn,pw))
                        assert abs(val-m(center))<=m(band)+mp.mpf('1e-25')
                    q=mp.fsum(m(w)*mp.exp(-m(wait)*m(rate)) for rate,w in zip(pn,pw))
                    assert abs(q-m(b['physical_witnesses'][side]['target']))<mp.mpf('2e-12')
                    counts['physical']+=1
                for endpoint,cert in b['certificates'].items():
                    rates,coefs=list(map(m,cert['rates'])),list(map(m,cert['coefficients']))
                    ev=lambda x:[mp.exp(-d*x) for d in rates]
                    value=lambda x:mp.fsum(c*e for c,e in zip(coefs,ev(x)))
                    previous=mp.mpf(0)
                    for left,right,stored in cert['leaves']:
                        l,r=m(left),m(right);assert l==previous and r>l;previous=r
                        el,er=ev(l),ev(r)
                        second=mp.fsum(c*d*d*(e1 if c>=0 else e0) for c,d,e0,e1 in zip(coefs,rates,el,er))
                        bound=min(max(value(l),value(r))+max(mp.mpf(0),-second)*(r-l)**2/8,
                            mp.fsum(c*(e0 if c>=0 else e1) for c,e0,e1 in zip(coefs,el,er)))
                        assert m(stored)>=bound-mp.mpf('1e-25') and cert['upper']>=stored
                        counts['finite_cells']+=1
                    assert previous==m(cert['tail_start'])
                    tail=mp.fsum(c for c,d in zip(coefs,rates) if not d)+mp.fsum(c*mp.exp(-d*previous) for c,d in zip(coefs,rates) if d>0 and c>0)
                    assert m(cert['tail_upper'])>=tail-mp.mpf('1e-25') and cert['upper']>=cert['tail_upper']
                    side='il' if endpoint=='lower' else 'ih';dual=np.minimum(arrays[label+'_'+side+'_y'],0.)
                    coefficients=dual[:5]-dual[5:] if side=='il' else -dual[:5]+dual[5:]
                    expected_rates=np.r_[wait,delays]
                    expected_coefs=np.r_[-1.,coefficients] if side=='il' else np.r_[1.,-coefficients]
                    for d,c in zip(cert['rates'],cert['coefficients']):
                        assert abs(c-expected_coefs[expected_rates==d].sum())<1e-14
                    counts['tails']+=1
                lo=np.minimum(arrays[label+'_il_y'],0.);hi=np.minimum(arrays[label+'_ih_y'],0.);mass=centers[0]+band
                expected=[max(0.,float(rhs@lo)-mass*max(0.,b['certificates']['lower']['upper'])-2e-7),
                    float(-rhs@hi)+mass*max(0.,b['certificates']['upper']['upper'])+2e-7]
                assert np.max(abs(np.array(expected)-b['outer']))<1e-13
                assert b['outer'][0]<=b['inner'][0]<=b['inner'][1]<=b['outer'][1]
            assert len(item['lp_logs'])==item['endpoint_programs']==2*len(item['brackets'])
            for i,row in enumerate(item['rows']):
                assert row['wait']==case['waits'][i]
                if row['route']=='direct':
                    b=item['brackets'][row['label']];answer=decision(b,.05,.10,0.)
                else:
                    assert row['route'] in ('time_cell','time_secant') and name=='loading_guided_reuse'
                    left,right=row['source'];assert left<row['wait']<right
                    b=transport_bracket(latest[left],latest[right],left,right,at=row['wait'] if row['route']=='time_secant' else None)
                    answer=transferred_decision(b,.05,.10,0.);counts['transport_decisions']+=1
                assert answer['state']==row['state']
                actual=case['measured_targets_scaled'][i]
                all_coverage.append({'case':case['id'],'algorithm':name,'seconds':case['actual_seconds'][i],
                    'in_outer':bool(b['outer'][0]<=actual<=b['outer'][1])})
                if row['state']=='sufficient':
                    q=row['joint_prediction'];ae=abs(q-actual);relative_error=ae/abs(actual)
                    # Validate the original stored recovery estimate, not a newer transferred one.
                    assert ae<=.05 and relative_error<=.10
                    positives.append({'case':case['id'],'algorithm':name,'seconds':case['actual_seconds'][i],
                        'absolute_error_scaled':ae,'relative_error':relative_error,'joint_tolerance_met':True})
                    counts['positive_raw_target_checks']+=1
            state_counts={s:sum(r['state']==s for r in item['rows']) for s in ('unbounded','sufficient','insufficient','unresolved')}
            assert state_counts==item['states']
            first=min((r['wait'] for r in item['rows'] if r['state']=='sufficient'),default=None)
            assert first==item['first_sufficient']
            assert all(a['endpoint_programs']<=b['endpoint_programs'] for a,b in zip(item['trace'],item['trace'][1:]))
            arrays.close()
    positive_key=lambda r:(r['case'],r['algorithm'],r['seconds'])
    assert sorted(positives,key=positive_key)==sorted(result['accepted_against_raw_targets'],key=positive_key)
    assert len(all_coverage)==2*result['correlated_requests']==630 and counts['lp']==884
    assert counts['physical']==counts['tails']==counts['lp']
    label_differences=[]
    for case in cases:
        a,b=[loaded[(case['id'],name)] for name in ('pointwise_classical','loading_guided_reuse')]
        for x,y in zip(a['rows'],b['rows']):
            if x['state']!=y['state']:label_differences.append({'case':case['id'],'wait':x['wait'],'states':[x['state'],y['state']]})
        assert a['first_sufficient']==b['first_sufficient']
    assert label_differences==result['all_label_differences']==[]
    recomputed_budgets=[]
    for budget in spec['endpoint_program_budgets']:
        row={'endpoint_program_budget_per_case':budget}
        for name in ('pointwise_classical','loading_guided_reuse'):
            snaps=[next(s for s in reversed(loaded[(c['id'],name)]['trace']) if s['endpoint_programs']<=budget) for c in cases]
            row[name]={s:sum(x['states'][s] for x in snaps) for s in ('sufficient','insufficient','unresolved')}
        recomputed_budgets.append(row)
    assert recomputed_budgets==result['budget_results']
    aggregates=[];paired=[]
    for rep in range(3):
        totals={name:{'total_seconds':0.,'endpoint_programs':0} for name in ('pointwise_classical','loading_guided_reuse')}
        for case in cases:
            a,b=[next(x for x in result['timing_rows'] if x['case']==case['id'] and x['algorithm']==name and x['repetition']==rep) for name in totals]
            for name,row in zip(totals,(a,b)):
                assert row['states']==loaded[(case['id'],name)]['states'] and not row['failures']
                assert row['endpoint_programs']==loaded[(case['id'],name)]['endpoint_programs']
                totals[name]['total_seconds']+=row['total_seconds'];totals[name]['endpoint_programs']+=row['endpoint_programs']
            paired.append({'case':case['id'],'repetition':rep,'pointwise_seconds':a['total_seconds'],
                'reuse_seconds':b['total_seconds'],'speed_ratio':a['total_seconds']/b['total_seconds']})
        aggregates.append({'repetition':rep,**totals,'speed_ratio':totals['pointwise_classical']['total_seconds']/totals['loading_guided_reuse']['total_seconds']})
    assert aggregates==result['aggregate'] and len(result['timing_rows'])==180
    reduction=1-aggregates[0]['loading_guided_reuse']['endpoint_programs']/aggregates[0]['pointwise_classical']['endpoint_programs']
    summary={'all_checks_passed':True,'result_sha256':sha(OUT/'result.json'),'cases_sha256':sha(OUT/'cases.json'),
        'protocol_sha256':sha(OUT/'protocol_receipt.json'),'retained_files_sha256':file_hashes,
        'counts':counts,'maximum_LP_residual':maxres,'cases':len(cases),'correlated_requests':315,
        'physical_curves':5,'distinct_observation_windows':10,'conditional_band_settings':3,
        'final_states':{name:loaded[(cases[0]['id'],name)]['states'] for name in []},
        'aggregate_times':aggregates,'median_speed_ratio':statistics.median(x['speed_ratio'] for x in aggregates),
        'endpoint_program_reduction_fraction':reduction,
        'case_repeat_runtime_losses':[x for x in paired if x['reuse_seconds']>x['pointwise_seconds']],
        'paired_timings':paired,'budget_results':recomputed_budgets,'final_label_differences':label_differences,
        'first_sufficient_changes':0,'positive_decisions_per_algorithm':len(positives)//2,
        'raw_target_joint_exceedances':0,'raw_target_outer_misses':[x for x in all_coverage if not x['in_outer']],
        'calibrated_material_validation':False,'cross_loading_boundary_validated':False,
        'waiting_time_recovery_exponent_validated_on_material':False,
        'cautions':['Assumed bands and conditional hold-state class; not calibrated material guarantee',
            'Same real curves previously inspected; exploratory reuse study, not independent validation',
            'Identical classical oracle; time reuse can also be implemented by prior frameworks',
            'Same-process machine-specific timings; no universal speedup or better point prediction',
            '80-digit replay is not external audit or directed-rounding proof']}
    summary['final_states']={name:{s:sum(loaded[(c['id'],name)]['states'][s] for c in cases) for s in ('sufficient','insufficient','unresolved')} for name in ('pointwise_classical','loading_guided_reuse')}
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('retained_files_sha256','paired_timings','budget_results')},indent=2),flush=True)
if __name__=='__main__':main()
