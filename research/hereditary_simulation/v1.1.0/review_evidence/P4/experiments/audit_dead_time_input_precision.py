"""Alternative time integrals, physical input/spectrum pairs, and LP replay."""
import json
import numpy as np
from scipy.integrate import quad

from dead_time_input_precision import ROOT, SPEC, OUT, ARCHIVE, representation, decisions
from observation_weighted_prediction import digest, integrate_truth

AUDIT=ROOT/"results/p4_dead_time_input_precision_audit_20261012.json"


def time_integral(rate, span, loading_rate, rho=0., shape=0):
    if rate == 0:
        return loading_rate*quad(lambda r:1+shape*rho*np.sin(6*np.pi*r/span),0,span,epsabs=1e-13)[0]
    return loading_rate/rate*quad(lambda z:np.exp(-z)*(1+shape*rho*np.sin(6*np.pi*z/(rate*span))),
                                  0,min(rate*span,80.),epsabs=1e-13,epsrel=1e-13)[0]


def base(rate, design):
    return time_integral(rate,1.,1.) if design=="constant" else time_integral(rate,1/3,3.)


def target(rate,wait):
    return np.exp(-wait*rate)*time_integral(rate,.5,2.)


def main():
    if AUDIT.exists():
        raise FileExistsError("Frozen audit exists")
    result=json.loads(OUT.read_text(encoding="utf-8"))
    spec=json.loads(SPEC.read_text(encoding="utf-8"))
    truthpath=SPEC.parent/spec["parent_truth_spec"]
    truths={t["name"]:t for t in json.loads(truthpath.read_text())["truths"]}
    assert result["spec_sha256"]==digest(SPEC)
    assert result["source_sha256"]==digest(ROOT/"experiments/dead_time_input_precision.py")
    assert result["archive_sha256"]==digest(ARCHIVE)
    assert result["truth_spec_sha256"]==digest(truthpath)
    assert result["parent_result_sha256"]==spec["parent_result_sha256"]==digest(ROOT/"results"/spec["parent_result"])
    for name,value in result["dependencies_sha256"].items():
        assert digest(ROOT/"experiments"/name)==value
    data={(r["truth"],r["design"],r["dead_time"]):r for r in result["data"]}
    mx={k:0. for k in ("primal","stationarity","gap","sign","physical_excess","target_error","input_pair_observation_error")}
    source_checks=cell_checks=lp_checks=physical=envelope_checks=unbounded=0
    for datum in data.values():
        for d,val in zip(spec["observation_delays_from_first"],datum["observations"]):
            actual=integrate_truth(truths[datum["truth"]],lambda rate:base(rate,datum["design"])*np.exp(-(datum["dead_time"]+d)*rate))
            assert abs(actual-val)<2e-11
            source_checks+=1
    reps={}
    with np.load(ARCHIVE,allow_pickle=False) as saved:
        for row in result["rows"]:
            key=(row["design"],row["dead_time"],row["wait"],row["cells"])
            if key not in reps:
                reps[key]=representation(spec,*key)
                rep=reps[key]
                candidates=[]
                for i in np.linspace(0,row["cells"]-2,40,dtype=int):
                    a,b=rep["edges"][i:i+2]
                    candidates += [(i,.25*a+.75*b),(i,.75*a+.25*b)]
                if rep["critical_rate"] is not None:
                    rate=rep["critical_rate"]
                    i=np.searchsorted(rep["edges"],rate,side="right")-1
                    candidates.append((i,rate))
                candidates += [(row["cells"]-1,r) for r in (2048.,10000.,1e7)]
                for i,rate in candidates:
                    # Cancel the shared reference dead time before evaluating;
                    # avoids a 0/0 underflow at extremely fast tail rates.
                    ratio=np.exp(-(row["wait"]-row["dead_time"])*rate)*time_integral(rate,.5,2.)/base(rate,row["design"])
                    assert rep["q_low"][i]-2e-10<=ratio<=rep["q_high"][i]+2e-10
                    cell_checks+=1
            rep=reps[key]
            datum=data[row["truth"],row["design"],row["dead_time"]]
            obs=np.array(datum["observations"])
            rhs=np.r_[obs+spec["absolute_response_band"],-obs+spec["absolute_response_band"]]
            true=integrate_truth(truths[row["truth"]],lambda rate:target(rate,row["wait"]))
            assert abs(true-row["truth_target"])<2e-11
            for method,a,cs in (("outer",rep["outer_a"],(rep["q_low"],-rep["q_high"])),
                                ("inner",rep["inner_a"],(rep["inner_q"],-rep["inner_q"]))):
                for side,(suffix,c) in enumerate(zip(("low","high"),cs)):
                    label=row["label"]+"_"+method+"_"+suffix
                    x,y,z=[saved[label+"_"+k] for k in ("x","y","z")]
                    mx["primal"]=max(mx["primal"],float(max(0.,np.max(a@x-rhs),-np.min(x))))
                    mx["stationarity"]=max(mx["stationarity"],float(np.max(np.abs(c-a.T@y-z))))
                    mx["gap"]=max(mx["gap"],float(abs(c@x-rhs@y)))
                    mx["sign"]=max(mx["sign"],float(max(0.,np.max(y),-np.min(z))))
                    assert abs((1 if side==0 else -1)*float(c@x)-row[method+"_interval"][side])<2e-10
                    lp_checks+=1
                    if method!="inner":
                        continue
                    active=np.flatnonzero(x>0)
                    rates=rep["nodes"][active]
                    kernels=np.array([base(float(r),row["design"])*np.exp(-row["dead_time"]*r) for r in rates])
                    mass=x[active]/kernels
                    assert np.all(np.isfinite(mass)) and np.all(mass>=0)
                    resp=np.array([sum(m*k*np.exp(-d*r) for m,k,r in zip(mass,kernels,rates)) for d in spec["observation_delays_from_first"]])
                    tq=np.array([target(float(r),row["wait"]) for r in rates])
                    nominal=float(mass@tq)
                    mx["physical_excess"]=max(mx["physical_excess"],float(np.max(abs(resp-obs))-spec["absolute_response_band"]))
                    mx["target_error"]=max(mx["target_error"],abs(nominal-row["inner_interval"][side]))
                    physical+=1
                    for rho in spec["input_relative_envelopes"]:
                        if rho==0:
                            continue
                        gain=1+rho if side==0 else 1-rho
                        actual_mass=mass/gain
                        actual_resp=np.array([sum(m*gain*k*np.exp(-d*r) for m,k,r in zip(actual_mass,kernels,rates)) for d in spec["observation_delays_from_first"]])
                        actual_target=float(actual_mass@tq)
                        mx["input_pair_observation_error"]=max(mx["input_pair_observation_error"],float(np.max(abs(actual_resp-resp))))
                        assert abs(actual_target-nominal/gain)<2e-11
                        envelope_checks+=1
    for row in result["unbounded_witnesses"]:
        r=row["rate"]
        g=base(r,row["design"])*np.exp(-row["dead_time"]*r)
        changes=np.array([row["physical_added_mass"]*g*np.exp(-d*r) for d in spec["observation_delays_from_first"]])
        val=row["physical_added_mass"]*target(r,row["wait"])
        assert np.max(abs(changes-np.array(row["observation_changes"])))<2e-12
        assert abs(val-row["target_change"])/max(1.,abs(val))<2e-12
        assert max(changes)<spec["absolute_response_band"]
        unbounded+=1
    # Genuine time-varying inputs inside the envelope, not just endpoint gains.
    shapes=0
    rates=np.array([0.,.25,2.,20.]); nominal_mass=np.array([.2,.3,.3,.2])
    for design in spec["designs"]:
        span,load=(1.,1.) if design=="constant" else (1/3,3.)
        for rho in (.01,.03,.05):
            for sign in (-1,1):
                gn=np.array([base(float(r),design) for r in rates])
                gf=np.array([time_integral(float(r),span,load,rho,sign) for r in rates])
                assert np.all((1-rho)*gn-1e-13<=gf) and np.all(gf<=(1+rho)*gn+1e-13)
                actual_mass=nominal_mass*gn/gf
                for dead in spec["reference_dead_times"]:
                    for d in spec["observation_delays_from_first"]:
                        original=np.sum(nominal_mass*gn*np.exp(-(dead+d)*rates))
                        changed=np.sum(actual_mass*gf*np.exp(-(dead+d)*rates))
                        assert abs(original-changed)<2e-12
                    for wait in (.0,.25,1.):
                        q=np.array([target(float(r),wait) for r in rates])
                        before=float(nominal_mass@q); after=float(actual_mass@q)
                        assert before/(1+rho)-2e-12<=after<=before/(1-rho)+2e-12
                        shapes+=1
    selected,summaries,gate=decisions(result["rows"],spec)
    assert selected==result["decisions"] and summaries==result["summaries"] and gate==result["continuation_gate_pass"]
    for r in selected:
        assert r["relative_radius_bracket"][0]>=r["rho"]-2e-12
    assert max(mx.values())<=spec["lp_replay_tolerance"]
    audit={"schema":"P4-dead-time-input-precision-audit-v1","source_sha256":digest(__file__),"result_sha256":digest(OUT),
           "LPs_replayed":lp_checks,"nominal_physical_endpoints":physical,"envelope_physical_endpoints":envelope_checks,
           "independent_observation_integrals":source_checks,"independent_cell_ratio_checks":cell_checks,
           "unbounded_pairs":unbounded,"nonconstant_input_shape_checks":shapes,"max_residuals":mx,
           "max_finest_endpoint_bracket":result["max_finest_endpoint_bracket"],"continuation_gate_pass":gate,
           "all_checks_pass":True,"limits":"Same developer alternative time integrals, physical envelope pairs and saved-LP replay; not external review or directed rounding."}
    AUDIT.write_text(json.dumps(audit,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    main()
