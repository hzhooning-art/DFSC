"""Audit extraction, no-leakage time split, all LP witnesses and failed hold-outs."""
from pathlib import Path
import json,hashlib
import numpy as np
import mpmath as mp
from prediction_time_search import decision
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/orbital_hold_pilot_20261021'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    mp.mp.dps=80;m=lambda x:mp.mpf(float(x))
    spec=json.loads((ROOT/'experiments/specs/orbital_hold_pilot_20261021.json').read_text())
    result=json.loads((OUT/'result.json').read_text());receipt=json.loads((OUT/'protocol_receipt.json').read_text())
    data=json.loads((OUT/'extracted.json').read_text());arrays=np.load(OUT/'lp_witnesses.npz')
    assert all(sha(ROOT/k)==v for k,v in receipt['bindings'].items())
    assert sha(OUT/'lp_witnesses.npz')==result['lp_witnesses_sha256']
    assert len(data['traces'])==12 and len(result['results'])==108 and not result['guarantee_validated']
    lps=cells=tails=physical=0;maxres=0.;violations=[]
    for row in result['results']:
        trace=next(t for t in data['traces'] if t['sheet']==row['sheet'])
        records=sorted(trace['raw_records'],key=lambda r:r['time_seconds']);uniq={}
        for rec in records:uniq.setdefault(rec['time_seconds'],rec)
        records=list(uniq.values())
        ts=np.array([r['time_seconds'] for r in records]);ss=np.array([r['stress_Pa'] for r in records])
        pick=lambda t:int(np.argmin(abs(ts-t)))
        obs=[pick(x) for x in spec['observation_seconds']];anchor=pick(spec['anchor_seconds'])
        target=pick(row['nominal_target_seconds'])
        assert ts[target]>max(ts[obs])
        assert np.max(abs(ss[obs]/ss[anchor]-row['centers']))<1e-14
        assert abs(ss[target]/ss[anchor]-row['actual_normalized_stress'])<1e-14
        if row['bracket'] is None:assert row['decision']['state']=='unresolved';continue
        b=row['bracket'];rhs=np.r_[np.array(row['centers'])+row['band'],-np.array(row['centers'])+row['band']]
        for side,record in b['physical_witnesses'].items():
            stem=record['label'];nodes,x,y,z=[arrays[stem+'_'+k] for k in ('nodes','x','y','z')]
            E=np.exp(-np.array(row['delays'])[:,None]*nodes);a=np.vstack((E,-E))
            c=np.exp(-row['wait']*nodes)*(1 if side=='il' else -1)
            residual=max(0.,float(np.max(a@x-rhs)),float(-np.min(x)),float(np.max(np.abs(c-a.T@y-z))),float(np.max(y)),float(-np.min(z)),float(abs(c@x-rhs@y)))
            assert residual<=2e-7;maxres=max(maxres,residual);lps+=1
            pn,pw=arrays[stem+'_physical_nodes'],arrays[stem+'_physical_weights']
            for d,center in zip(row['delays'],row['centers']):
                response=mp.fsum(m(w)*mp.exp(-m(d)*m(r)) for r,w in zip(pn,pw))
                assert abs(response-m(center))<=m(row['band'])+mp.mpf('1e-25')
            physical+=1
        for name,cert in b['certificates'].items():
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
                assert m(stored)>=bound-mp.mpf('1e-25') and cert['upper']>=stored;cells+=1
            assert previous==m(cert['tail_start'])
            tail=mp.fsum(c for c,d in zip(coefs,rates) if not d)+mp.fsum(c*mp.exp(-d*previous) for c,d in zip(coefs,rates) if d>0 and c>0)
            assert m(cert['tail_upper'])>=tail-mp.mpf('1e-25') and cert['upper']>=cert['tail_upper'];tails+=1
        assert decision(b,spec['absolute_tolerance'],spec['relative_tolerance'],0.)['state']==row['decision']['state']
        if row['decision']['state']=='sufficient':
            z=row['decision']['joint_prediction'];actual=row['actual_normalized_stress']
            if abs(z-actual)>.05 or abs(z-actual)/abs(actual)>.1:
                violations.append({'sheet':row['sheet'],'band':row['band'],'actual_target_seconds':row['actual_target_seconds'],
                    'measured_normalized_response':actual,'computed_outer':b['outer'],'joint_estimate':z,
                    'absolute_error':abs(z-actual),'relative_error':abs(z-actual)/abs(actual)})
    assert lps==12 and len(violations)==1
    audit={'all_numerical_checks_passed':True,'physical_validation_passed':False,'result_sha256':sha(OUT/'result.json'),
        'spec_sha256':result['spec_sha256'],'lp_replays':lps,'physical_witnesses_at_80_digits':physical,
        'finite_cells_at_80_digits':cells,'infinite_tails_at_80_digits':tails,'max_lp_residual':maxres,
        'strict_feasibility_failures_are_unresolved_not_infeasibility_certificates':True,
        'accepted_but_measured_error_exceeds_tolerance':violations,
        'decision':'Do not promote to calibrated real-material validation or expand this source after its failed prespecified pilot.',
        'limits':spec['qualification_gaps']}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
