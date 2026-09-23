"""High precision confirmation of the crossover; not a global certificate."""
from pathlib import Path
import hashlib,json
import numpy as np
import mpmath as mp
from uniform_crossover import fit,coefficients


def precise(lam,p,delta,r):
    mp.mp.dps=60
    lam,p,delta,r=map(lambda v:mp.mpf(str(v)),(lam,p,delta,r))
    ts=[mp.mpf(8)*i/23 for i in range(24)]
    center=lambda xs:[v-sum(xs)/len(xs) for v in xs]
    e=[mp.exp(-lam*t) for t in ts]
    g=[v*mp.expm1(-delta*t) for v,t in zip(e,ts)]
    xs=[center([v+p*w for v,w in zip(e,g)]),center([mp.sqrt(r)*v for v in g])]
    def evaluate(k):
        ek=[mp.exp(-k*t) for t in ts]
        q=center(ek);ep=center([-t*v for t,v in zip(ts,ek)])
        norm=sum(v*v for v in q);cross=sum(v*w for v,w in zip(q,ep))
        d2=mp.mpf(0);grad=mp.mpf(0)
        for x in xs:
            dot=sum(v*w for v,w in zip(x,q));dote=sum(v*w for v,w in zip(x,ep))
            d2+=sum((v-dot*w/norm)**2 for v,w in zip(x,q))
            grad+=-2*dot*dote/norm+2*dot**2*cross/norm**2
        return grad,d2
    k=mp.findroot(lambda k:evaluate(k)[0],(lam+p*delta-delta/10,lam+p*delta+delta/10),tol=mp.mpf('1e-50'))
    return float(k),float(evaluate(k)[1])


def audit():
    here=Path(__file__).resolve().parent
    data=json.loads((here/'uniform_crossover_results.json').read_text())
    for name,digest in data['hashes'].items():
        assert hashlib.sha256((here/name).read_bytes()).hexdigest()==digest
    for row in data['rows']:
        assert row['roots']==1
        v,cw=coefficients(np.linspace(0,8,24),row['lam'],row['p'])
        leading=row['r']*v*row['delta']**2+cw*row['delta']**4
        assert abs(leading/row['leading']-1)<1e-12
        assert abs(abs(row['distance_squared']/leading-1)-row['relative_error'])<1e-12
    selected=[row for row in data['rows'] if row['lam'] in (.3,.6) and row['p'] in (.2,.8)
              and row['delta']==.0002 and row['path'].startswith('balance_')]
    comparisons=[]
    for row in selected:
        rate,d2=precise(row['lam'],row['p'],row['delta'],row['r'])
        error=abs(d2/row['distance_squared']-1)
        assert error<1e-5 and abs(rate-row['rate'])<1e-10
        comparisons.append(dict(lam=row['lam'],p=row['p'],path=row['path'],relative_error=error))
    # Extra signed-gap diagnostic, outside the frozen positive-gap grid.
    negative=[]
    for r in (0.,1e-8,.01):
        rate,d2=precise(.45,.5,-.0002,r)
        k,ordinary,_=fit(np.linspace(0,8,24),.45,.5,-.0002,r)
        assert abs(d2/ordinary-1)<1e-5 and abs(k-rate)<1e-10
        v,cw=coefficients(np.linspace(0,8,24),.45,.5)
        error=abs(d2/(r*v*.0002**2+cw*.0002**4)-1)
        assert error<.05
        negative.append(dict(r=r,leading_relative_error=error))
    boundaries={r['name']:r for r in data['boundaries']}
    assert boundaries['three_times']['curvature_coefficient']<1e-25
    assert boundaries['three_times']['distance_squared']<1e-25
    assert boundaries['absent_mode']['distance_squared']<1e-25
    boundary=boundaries['upper_rate_boundary']
    assert boundary['rate']==.8
    ratio=boundary['distance_squared']/boundary['interior_prediction']
    assert ratio>1e6
    # Correct one-sided boundary leading term: p² V delta².
    boundary_first_ratio=boundary['distance_squared']/(.4**2*boundary['V']*.001**2)
    assert abs(boundary_first_ratio-1)<.01
    return dict(grid_checks=len(data['rows']),high_precision_digits=60,comparisons=comparisons,
                max_high_precision_relative_error=max(r['relative_error'] for r in comparisons),
                negative_gap_checks=negative,boundary_wrong_prediction_factor=ratio,
                boundary_first_order_squared_ratio=boundary_first_ratio,
                status='local_high_precision_and_global_grid_screen_not_interval_certification')


if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
