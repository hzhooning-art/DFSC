from pathlib import Path
import json,hashlib
import numpy as np
from scipy.optimize import brentq


def coefficients(t,lam,p):
    e=np.exp(-lam*t);ep=-t*e;epp=t*t*e
    q=np.linalg.qr(np.column_stack([np.ones(len(t)),e]))[0]
    q3=np.linalg.qr(np.column_stack([np.ones(len(t)),e,ep]))[0]
    return float(np.sum((ep-q@(q.T@ep))**2)),float((p*(1-p)/2)**2*np.sum((epp-q3@(q3.T@epp))**2))


def fit(t,lam,p,delta,r):
    e=np.exp(-lam*t);g=e*np.expm1(-delta*t)
    x=np.array([e+p*g,np.sqrt(r)*g]);x-=x.mean(1,keepdims=True)
    def at(k):
        ek=np.exp(-k*t);ec=ek-ek.mean();q=ec/np.linalg.norm(ec)
        ep=-t*ek;ep-=ep.mean();qp=(ep-q*(q@ep))/np.linalg.norm(ec)
        dots=x@q;res=x-dots[:,None]*q
        return float(-2*np.sum(res*dots[:,None]*qp)),float(np.sum(res**2))
    grid=np.linspace(.2,.8,401)
    ee=np.exp(-grid[:,None]*t);ee-=ee.mean(1,keepdims=True)
    norms=np.linalg.norm(ee,axis=1);qs=ee/norms[:,None]
    ep=-t*np.exp(-grid[:,None]*t);ep-=ep.mean(1,keepdims=True)
    qps=(ep-qs*np.sum(qs*ep,axis=1)[:,None])/norms[:,None]
    grads=-2*np.sum((x@qs.T)*(x@qps.T),axis=0)
    roots=[brentq(lambda k:at(k)[0],a,b,xtol=5e-16) for a,b,ga,gb in zip(grid[:-1],grid[1:],grads[:-1],grads[1:]) if ga*gb<0]
    candidates=[(.2,at(.2)[1]),(.8,at(.8)[1])]+[(k,at(k)[1]) for k in roots]
    k,d2=min(candidates,key=lambda v:v[1])
    return k,d2,len(roots)


def run():
    t=np.linspace(0,8,24);rows=[]
    for lam in (.3,.45,.6):
        for p in (.2,.5,.8):
            v,cw=coefficients(t,lam,p)
            for delta in (.0002,.0005,.001,.002,.005,.01,.02):
                paths={'zero':0.,'fixed':.01,'linear_r':delta,'cubic_r':delta**3}
                paths.update({f'balance_{z}':z*cw/v*delta**2 for z in (.1,1.,10.)})
                for name,r in paths.items():
                    k,d2,nroots=fit(t,lam,p,delta,r);lead=r*v*delta**2+cw*delta**4
                    rows.append(dict(lam=lam,p=p,delta=delta,path=name,r=r,rate=k,distance_squared=d2,
                        leading=lead,relative_error=abs(d2/lead-1),scaled_remainder=abs(d2-lead)/(r*delta**3+delta**5),roots=nroots))
    small=[r for r in rows if r['delta']<=.002]
    b=[]
    for name,times,lam,p,d,r in [('three_times',np.array([0.,4.,8.]),.4,.4,.001,0.),
            ('absent_mode',t,.4,0.,.001,0.),('upper_rate_boundary',t,.8,.4,.001,0.)]:
        v,cw=coefficients(times,lam,p);k,d2,n=fit(times,lam,p,d,r)
        b.append(dict(name=name,rate=k,distance_squared=d2,V=v,curvature_coefficient=cw,
                      interior_prediction=r*v*d*d+cw*d**4))
    here=Path(__file__).resolve().parent
    return dict(rows=rows,boundaries=b,gate=all(r['relative_error']<=.05 for r in small),
                max_small_gap_relative_error=max(r['relative_error'] for r in small),
                max_scaled_remainder=max(r['scaled_remainder'] for r in rows),
                hashes={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('UNIFORM_CROSSOVER_THEORY.md','uniform_crossover.py')})


if __name__=='__main__':
    out=run();Path(__file__).with_name('uniform_crossover_results.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('rows','hashes')},indent=2))
