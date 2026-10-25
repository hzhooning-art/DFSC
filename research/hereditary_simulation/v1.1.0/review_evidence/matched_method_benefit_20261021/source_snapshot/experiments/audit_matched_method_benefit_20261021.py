"""Replay retained witnesses, continuum cells and paired benefit accounting."""
from pathlib import Path
import hashlib,json,statistics,re
import numpy as np
import mpmath as mp
from prediction_time_search import Information, decision
from prediction_time_transport import transport_bracket, transferred_decision
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/matched_method_benefit_20261021/run_01'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    mp.mp.dps=80
    m=lambda x:mp.mpf(float(x))
    result=json.loads((OUT/'result.json').read_text())
    receipt=json.loads((OUT/'protocol_receipt.json').read_text())
    assert sha(OUT/'protocol_receipt.json')==result['protocol_receipt_sha256']
    assert all(sha(ROOT/k)==v for k,v in receipt['bindings'].items())
    spec=json.loads((ROOT/'experiments/specs/matched_method_benefit_20261021.json').read_text())
    counts=dict(lp=0,physical=0,finite_cells=0,tails=0,positive_truth=0,transport=0)
    maxres=0.;files={};details=[]
    for path in sorted(OUT.glob('*_r*_*.json')):
        item=json.loads(path.read_text());arrayspath=path.with_name(path.stem+'_witnesses.npz')
        arrays=np.load(arrayspath);files[path.name]=sha(path);files[arrayspath.name]=sha(arrayspath)
        centers=item['strict_witness']['observed_responses']
        # Original centers are regenerated from immutable truth specification,
        # not approximated by the feasible witness's responses.
        from observation_weighted_prediction import integrate_truth
        from prediction_time_search import log_laplace
        truthspec=json.loads((ROOT/spec['truth_spec']).read_text())
        parsed=re.fullmatch(r'(.+)_r([0-9.]+)_(pointwise_classical|loading_guided_reuse)',path.stem)
        assert parsed,path.stem
        name=parsed[1]
        truth=next(t for t in truthspec['truths'] if t['name']==name)
        centers=tuple(integrate_truth(truth,lambda x:float(np.exp(-(.1+d)*x+log_laplace(x,1.)))) for d in spec['observation_delays'])
        info=Information(1.,.1,tuple(spec['observation_delays']),centers,(.01,)*5)
        rhs=np.r_[np.array(centers)+.01,-np.array(centers)+.01]
        rho=float(parsed[2])
        for label,b in item['brackets'].items():
            wait=float(label.split('_w')[1].split('_c')[0])
            for side in ('il','ih'):
                stem=label+'_'+side
                nodes,x,y,z=[arrays[stem+'_'+k] for k in ('nodes','x','y','z')]
                E=np.exp(-np.array(info.delays)[:,None]*nodes);a=np.vstack((E,-E))
                c=np.exp(-(wait-.1)*nodes)*(1 if side=='il' else -1)
                residual=max(0.,float(np.max(a@x-rhs)),float(-np.min(x)),float(np.max(np.abs(c-a.T@y-z))),float(np.max(y)),float(-np.min(z)),float(abs(c@x-rhs@y)))
                assert residual<=2e-7,(label,residual)
                maxres=max(maxres,residual);counts['lp']+=1
                pn,pw=arrays[stem+'_physical_nodes'],arrays[stem+'_physical_weights']
                assert np.isfinite(pn).all() and (pn>=0).all() and (pw>0).all()
                for d,center in zip(info.delays,centers):
                    val=mp.fsum(m(w)*mp.exp(-m(d)*m(r)) for r,w in zip(pn,pw))
                    assert abs(val-m(center))<=m(.01)+mp.mpf('1e-25')
                counts['physical']+=1
                target=mp.fsum(m(w)*mp.exp(-(m(wait)-m(.1))*m(r)) for r,w in zip(pn,pw))
                assert abs(target-m(b['physical_witnesses'][side]['target']))<mp.mpf('2e-12')
            for endpoint,cert in b['certificates'].items():
                rates,coefs=list(map(m,cert['rates'])),list(map(m,cert['coefficients']))
                ev=lambda x:[mp.exp(-v*x) for v in rates]
                value=lambda x:mp.fsum(c*e for c,e in zip(coefs,ev(x)))
                previous=mp.mpf(0)
                for left,right,stored in cert['leaves']:
                    l,r=m(left),m(right);assert l==previous and r>l;previous=r
                    el,er=ev(l),ev(r)
                    second=mp.fsum(c*d*d*(e1 if c>=0 else e0) for c,d,e0,e1 in zip(coefs,rates,el,er))
                    bound=min(max(value(l),value(r))+max(mp.mpf(0),-second)*(r-l)**2/8,
                              mp.fsum(c*(e0 if c>=0 else e1) for c,e0,e1 in zip(coefs,el,er)))
                    assert m(stored)>=bound-mp.mpf('1e-25')
                    assert m(cert['upper'])>=m(stored)-mp.mpf('1e-25')
                    counts['finite_cells']+=1
                assert previous==m(cert['tail_start'])
                tail=mp.fsum(c for c,d in zip(coefs,rates) if not d)+mp.fsum(c*mp.exp(-d*previous) for c,d in zip(coefs,rates) if d>0 and c>0)
                assert m(cert['tail_upper'])>=tail-mp.mpf('1e-25')
                assert cert['upper']>=cert['tail_upper']
                side='il' if endpoint=='lower' else 'ih'
                y=np.minimum(arrays[label+'_'+side+'_y'],0.)
                c=y[:5]-y[5:] if endpoint=='lower' else -y[:5]+y[5:]
                expected_rates=np.r_[wait-.1,info.delays]
                expected_coefs=np.r_[-1.,c] if endpoint=='lower' else np.r_[1.,-c]
                for d,c in zip(cert['rates'],cert['coefficients']):
                    assert abs(c-expected_coefs[expected_rates==d].sum())<1e-14
                counts['tails']+=1
            lo_y=np.minimum(arrays[label+'_il_y'],0.);hi_y=np.minimum(arrays[label+'_ih_y'],0.)
            B=centers[0]+.01
            expected=[max(0.,float(rhs@lo_y)-B*max(0,b['certificates']['lower']['upper'])-2e-7),
                      float(-rhs@hi_y)+B*max(0,b['certificates']['upper']['upper'])+2e-7]
            assert np.max(np.abs(np.array(expected)-b['outer']))<1e-13
        latest={}
        for label,b in item['brackets'].items():
            wait=float(label.split('_w')[1].split('_c')[0]);latest[wait]=b
        for row in item['rows']:
            if row['route']=='phase':continue
            if row['route']=='direct':
                d=decision(item['brackets'][row['label']],spec['absolute'],spec['relative'],rho)
            else:
                left,right=row['source']
                b=transport_bracket(latest[left],latest[right],left,right,at=row['wait'] if row['route']=='time_secant' else None)
                d=transferred_decision(b,spec['absolute'],spec['relative'],rho)
                counts['transport']+=1
            assert d['state']==row['state']
        for check in item['truth_checks']:
            assert check['enclosed'] and check['absolute_error']<=spec['absolute'] and check['relative_error']<=spec['relative']
            counts['positive_truth']+=1
        assert not item['failures']
    paired=[];budgets=[];wallbudgets=[]
    for repetition in range(spec['repetitions']):
        for scene in ('two_rate','wide_rate','uniform','logwide'):
            for rho in spec['rhos']:
                a,b=[next(r for r in result['rows'] if r['repetition']==repetition and r['scenario']==scene and r['rho']==rho and r['algorithm']==name) for name in ('pointwise_classical','loading_guided_reuse')]
                assert a['decisions']==b['decisions'] and a['first_sufficient']==b['first_sufficient']
                paired.append({'repetition':repetition,'scenario':scene,'rho':rho,
                    'pointwise_seconds':a['total_seconds'],'reuse_seconds':b['total_seconds'],
                    'pointwise_programs':a['endpoint_programs'],'reuse_programs':b['endpoint_programs'],
                    'speedup':a['total_seconds']/b['total_seconds']})
                for key,values,destination in (('endpoint_programs',spec['endpoint_program_budgets'],budgets),('elapsed_seconds',spec['wall_second_budgets'],wallbudgets)):
                    for budget in values:
                        snapshots=[next(x for x in reversed(r['trace']) if x[key]<=budget) for r in (a,b)]
                        destination.append({'repetition':repetition,'scenario':scene,'rho':rho,'budget':budget,
                            **{name:nextsnap['states'] for name,nextsnap in zip(('pointwise','reuse'),snapshots)}})
    totals=[]
    for rep in range(spec['repetitions']):
        selected=[p for p in paired if p['repetition']==rep]
        a=sum(p['pointwise_seconds'] for p in selected);b=sum(p['reuse_seconds'] for p in selected)
        totals.append({'repetition':rep,'pointwise_seconds':a,'reuse_seconds':b,'speedup':a/b})
    summary={'all_checks_passed':True,'result_sha256':sha(OUT/'result.json'),'protocol_sha256':sha(OUT/'protocol_receipt.json'),
        'retained_files_sha256':files,'counts':counts,'maximum_LP_residual':maxres,
        'paired':paired,'aggregate_times':totals,'aggregate_median_speedup':statistics.median(t['speedup'] for t in totals),
        'endpoint_budgets':budgets,'wall_budgets':wallbudgets,
        'opposite_or_lost_final_labels':0,'earliest_sufficient_changes':0,
        'cautions':['Own classical-framework adaptation with shared oracle; not external author software comparison',
            'Four reused synthetic observation sets, not 584 independent experiments',
            'Some tasks are slower; no universal speedup or improved physical point prediction',
            'Same-developer 80-digit replay, not external review or directed rounding']}
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('paired','endpoint_budgets','wall_budgets','retained_files_sha256')},indent=2),flush=True)
if __name__=='__main__':main()
