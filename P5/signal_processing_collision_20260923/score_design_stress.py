import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import f, ncf


HERE = Path(__file__).resolve().parent
M = 16384
ALPHA = .05
KLOW, KHIGH = .2, .8
P = .5
R = .04
TARGET = 8.0
ALT_TRIALS = 512
NULL_TRIALS = 1024


def time_grid(n, irregular):
    times = np.linspace(0.0, 8.0, n)
    if irregular:
        step = 8.0 / (n - 1)
        times[1:-1] += .18 * step * np.sin(np.arange(1, n - 1) * 1.7)
        times.sort()
    return times


DESIGNS = (
    ("baseline", 6, 24, .4, 10000., False, None),
    ("channels_2", 2, 24, .4, 10000., False, None),
    ("channels_12", 12, 24, .4, 10000., False, None),
    ("times_12", 6, 12, .4, 10000., False, None),
    ("times_48", 6, 48, .4, 10000., False, None),
    ("rate_025", 6, 24, .25, 10000., False, None),
    ("rate_065", 6, 24, .65, 10000., False, None),
    ("irregular_24", 6, 24, .4, 10000., True, None),
    ("energy_100", 6, 24, .4, 100., False, None),
    ("scale_nu_144", 6, 24, .4, 10000., False, 144),
    ("combined_small", 4, 16, .3, 1000., True, 144),
    ("combined_large", 8, 36, .6, 1000., True, None),
)


def projections(times, rate):
    n = len(times)
    e = np.exp(-rate * times)
    ep = -times * e
    epp = times * times * e
    x2 = np.column_stack((np.ones(n), e))
    q2, _ = np.linalg.qr(x2)
    v1 = ep - q2 @ (q2.T @ ep)
    V = float(v1 @ v1)
    g1 = v1 / np.sqrt(V)
    x3 = np.column_stack((np.ones(n), e, ep))
    q3, _ = np.linalg.qr(x3)
    v2 = epp - q3 @ (q3.T @ epp)
    W = float(v2 @ v2)
    g2 = v2 / np.sqrt(W)
    return V, W, g1, g2


def profiler(times):
    grid = np.linspace(KLOW, KHIGH, 401)
    vectors = np.stack([np.exp(-rate * times) for rate in grid])
    vectors -= vectors.mean(axis=1, keepdims=True)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    def profile(y):
        yc = y - y.mean(axis=1, keepdims=True)
        total = float(np.sum(yc * yc))
        scores = yc @ vectors.T
        rss = total - np.sum(scores * scores, axis=0)
        index = int(np.argmin(rss))

        def objective(rate):
            v = np.exp(-rate * times)
            v -= v.mean(); v /= np.linalg.norm(v)
            z = yc @ v
            return max(0.0, total - float(z @ z))

        lo, hi = grid[max(0, index - 1)], grid[min(len(grid) - 1, index + 1)]
        candidates = [(float(rss[index]), float(grid[index])),
                      (objective(KLOW), KLOW), (objective(KHIGH), KHIGH)]
        if hi > lo:
            fit = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-12})
            candidates.append((float(fit.fun), float(fit.x)))
        return min(candidates)
    return profile


def cone_critical(c, nu):
    return float(brentq(
        lambda q: .5*f.cdf(q/(c-1), c-1, nu)+.5*f.cdf(q/c, c, nu)-(1-ALPHA),
        0.0, 100.0,
    ))


def make_condition(c, times, rate, S, path):
    V, W, _, _ = projections(times, rate)
    c_mix = (P * (1-P) / 2)**2 * W
    if path == "diversity":
        r = R; delta = np.sqrt(TARGET/(S*R*V))*M**-.5
    elif path == "crossover":
        r = R*M**-.5; aa=S*R*V; bb=S*c_mix
        x=(-aa+np.sqrt(aa*aa+4*bb*TARGET))/(2*bb); delta=np.sqrt(x)*M**-.25
    elif path == "proportional":
        r=0.; delta=(TARGET/(S*c_mix))**.25*M**-.25
    elif path == "null":
        r=0.; delta=0.
    else:
        raise ValueError(path)
    s=np.ones(c)*np.sqrt(S/c)
    u=np.zeros(c);u[0],u[1]=1/np.sqrt(2),-1/np.sqrt(2)
    b=P*s+np.sqrt(S*r)*u;a=s-b
    mean=a[:,None]*np.exp(-rate*times)[None,:]
    mean+=b[:,None]*np.exp(-(rate+delta)*times)[None,:]
    return mean,float(delta),float(min(a.min(),b.min()))


def score_energy(y, times, fitted_rate):
    n=len(times); design=np.column_stack((np.ones(n),np.exp(-fitted_rate*times)))
    amplitude=np.linalg.lstsq(design,y.T,rcond=None)[0][1]
    _,_,g1,g2=projections(times,fitted_rate)
    sunit=amplitude/np.linalg.norm(amplitude)
    z1=y@g1; z1-=sunit*float(sunit@z1)
    z2=float(sunit@(y@g2))
    return float(z1@z1),z2


def evaluate_design(spec,index):
    name,c,n,rate,S,irregular,fixed_nu=spec
    times=time_grid(n,irregular);profile=profiler(times)
    nu=int(fixed_nu or c*n*(M-1));d=c*(n-2);r0=d-1
    crit_union=f.ppf(1-ALPHA,d,nu);crit_phase=f.ppf(1-ALPHA,c,nu)
    crit_cone=cone_critical(c,nu)
    output={"name":name,"C":c,"n":n,"lambda":rate,"S":S,"irregular":irregular,"nu":nu,"paths":{}}
    for pindex,path in enumerate(("null","diversity","crossover","proportional")):
        mean,delta,minamp=make_condition(c,times,rate,S,path)
        pop_rss,pop_rate=profile(mean);pop_div,pop_curv=score_energy(mean,times,pop_rate)
        score_nc=M*(pop_div+pop_curv**2);distance_nc=M*pop_rss
        pred=float(1-ncf.cdf(crit_phase,c,nu,score_nc))
        trials=NULL_TRIALS if path=="null" else ALT_TRIALS
        base=6091100000000+1000000*index+100000*pindex
        decisions=[]
        for j in range(trials):
            rng=np.random.default_rng(base+j)
            y=mean+rng.standard_normal(mean.shape)/np.sqrt(M)
            s2=rng.chisquare(nu)/nu
            rss,khat=profile(y);div,curv=score_energy(y,times,khat)
            decisions.append((M*rss/(d*s2)>crit_union,
                              M*(div+curv**2)/(c*s2)>crit_phase,
                              M*(div+max(curv,0.)**2)/s2>crit_cone))
        arr=np.asarray(decisions,float)
        output["paths"][path]={
            "delta":delta,"minimum_amplitude":minamp,"trials":trials,
            "distance_nc":distance_nc,"score_nc":score_nc,
            "retention":score_nc/distance_nc if distance_nc>0 else 1.,
            "prediction":pred,"union":float(arr[:,0].mean()),
            "phase":float(arr[:,1].mean()),"cone":float(arr[:,2].mean()),
            "prediction_error":abs(float(arr[:,1].mean())-pred),
        }
    return output


def run():
    designs=[evaluate_design(spec,i) for i,spec in enumerate(DESIGNS)]
    alts=[row for design in designs for name,row in design["paths"].items() if name!="null"]
    nulls=[design["paths"]["null"] for design in designs]
    gates={
        "null_phase_and_cone_between_002_008":all(.02<=row[k]<=.08 for row in nulls for k in ("phase","cone")),
        "phase_prediction_error_at_most_007":all(row["prediction_error"]<=.07 for row in alts),
        "retention_at_least_085":all(row["retention"]>=.85 for row in alts),
        "phase_gain_over_union_at_least_008":all(row["phase"]-row["union"]>=.08 for row in alts),
        "cone_loss_at_most_005":all(row["cone"]-row["phase"]>=-.05 for row in alts),
        "finite_and_positive":all(row["minimum_amplitude"]>0 and np.isfinite(list(row.values())).all() for row in alts),
    }
    return {"settings":{"m":M,"target_nc":TARGET,"alt_trials":ALT_TRIALS,"null_trials":NULL_TRIALS},
            "designs":designs,"gates":gates,
            "hashes":{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                      for name in ("SCORE_DESIGN_STRESS_PROTOCOL.md","score_design_stress.py")}}


if __name__=="__main__":
    result=run();(HERE/"score_design_stress_results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result["gates"],indent=2))
