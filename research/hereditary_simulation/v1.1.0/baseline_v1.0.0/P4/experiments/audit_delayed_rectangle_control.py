"""Replay new rectangle LPs and physical endpoints with time-domain quadrature."""
import json
import numpy as np
from scipy.integrate import quad
from delayed_rectangle_control import ROOT, SPEC, OUT, ARCHIVE, representation, comparisons
from observation_weighted_prediction import digest, integrate_truth

AUDIT=ROOT/"results/p4_delayed_rectangle_control_audit_20261011.json"

def direct_kernel(rate):
    if rate == 0:
        return quad(lambda t:3.,2/3,1,epsabs=1e-13)[0]
    return 3*quad(lambda z:np.exp(-z),0,min(rate/3,80.),epsabs=1e-13,epsrel=1e-13)[0]/rate

def direct_target(rate,delay):
    if rate == 0:
        return 1.
    return 2*np.exp(-delay*rate)*quad(lambda z:np.exp(-z),0,min(rate/2,80.),epsabs=1e-13)[0]/rate

def main():
    if AUDIT.exists():
        raise FileExistsError("Frozen audit exists")
    spec=json.loads(SPEC.read_text(encoding="utf-8"))
    result=json.loads(OUT.read_text(encoding="utf-8"))
    parent_spec_path=SPEC.parent/spec["parent_spec"]
    settings=json.loads(parent_spec_path.read_text(encoding="utf-8"))
    parent_path=ROOT/"results/p4_matched_endpoint_design_20261011.json"
    truth_path=SPEC.parent/settings["parent_truth_spec"]
    truths={t["name"]:t for t in json.loads(truth_path.read_text(encoding="utf-8"))["truths"]}
    assert digest(SPEC)==result["spec_sha256"]
    assert digest(ROOT/"experiments/delayed_rectangle_control.py")==result["source_sha256"]
    assert digest(ARCHIVE)==result["archive_sha256"]
    assert digest(parent_path)==result["parent_result_sha256"]==spec["parent_result_sha256"]
    assert digest(parent_spec_path)==result["parent_spec_sha256"]
    assert digest(truth_path)==result["truth_spec_sha256"]
    for name,value in result["dependencies_sha256"].items():
        assert digest(ROOT/"experiments"/name)==value
    parent=json.loads(parent_path.read_text(encoding="utf-8"))
    inherited=[dict(r,provenance="frozen_parent") for r in parent["rows"] if r["design"] in spec["retained_designs"]]
    assert inherited==[r for r in result["rows"] if r["provenance"]=="frozen_parent"]
    data={(r["truth"],r["band_mode"]):r for r in result["data"]}
    maximum={"primal":0.,"stationarity":0.,"gap":0.,"sign":0.,"physical_excess":0.,"physical_target_error":0.}
    source_checks=0
    for datum in result["data"]:
        for delay,value in zip(settings["observation_delays"],datum["observations"]):
            direct=integrate_truth(truths[datum["truth"]],lambda rate:direct_kernel(rate)*np.exp(-delay*rate))
            assert abs(direct-value)<2e-11
            source_checks+=1
        band=settings["absolute_response_band"] if datum["band_mode"]=="absolute" else settings["relative_anchor_band"]*datum["observations"][0]
        assert abs(band-datum["band"])<1e-14
    reps={}
    checked=physical=samples=0
    with np.load(ARCHIVE,allow_pickle=False) as saved:
        for row in result["rows"]:
            if row["provenance"]!="new_rectangle_control":
                continue
            key=(row["target"],row["cells"])
            if key not in reps:
                reps[key]=representation(settings,*key)
                rep=reps[key]
                for i in np.linspace(0,row["cells"]-2,80,dtype=int):
                    a,b=rep["edges"][i:i+2]
                    for rate in (.75*a+.25*b,.25*a+.75*b):
                        ratio=direct_target(rate,settings["targets"][row["target"]])/direct_kernel(rate)
                        assert rep["q_low"][i]-2e-10<=ratio<=rep["q_high"][i]+2e-10
                        samples+=1
                for rate in (2048.,10000.,1e7):
                    ratio=direct_target(rate,settings["targets"][row["target"]])/direct_kernel(rate)
                    assert rep["q_low"][-1]-2e-10<=ratio<=rep["q_high"][-1]+2e-10
                    samples+=1
            rep=reps[key]
            datum=data[row["truth"],row["band_mode"]]
            obs=np.array(datum["observations"])
            rhs=np.r_[obs+datum["band"],-obs+datum["band"]]
            stem=f"{row['truth']}_delayed_rectangle_{row['band_mode']}_{row['target']}_{row['cells']}"
            expected=integrate_truth(truths[row["truth"]],lambda rate:direct_target(rate,settings["targets"][row["target"]]))
            assert abs(expected-row["truth_target"])<2e-11
            for method,a,cs in (("outer",rep["outer_a"],(rep["q_low"],-rep["q_high"])),
                                ("inner",rep["inner_a"],(rep["inner_q"],-rep["inner_q"]))):
                for si,(side,c) in enumerate(zip(("low","high"),cs)):
                    label=stem+"_"+method+"_"+side
                    x,y,z=[saved[label+"_"+k] for k in ("x","y","z")]
                    maximum["primal"]=max(maximum["primal"],float(max(0.,np.max(a@x-rhs),-np.min(x))))
                    maximum["stationarity"]=max(maximum["stationarity"],float(np.max(np.abs(c-a.T@y-z))))
                    maximum["gap"]=max(maximum["gap"],float(abs(c@x-rhs@y)))
                    maximum["sign"]=max(maximum["sign"],float(max(0.,np.max(y),-np.min(z))))
                    assert abs(float(c@x)*(1 if si==0 else -1)-row[method+"_interval"][si])<2e-10
                    checked+=1
                    if method=="inner":
                        ix=np.flatnonzero(x>0)
                        rates=rep["nodes"][ix]
                        mass=np.array([x[i]/direct_kernel(float(rep["nodes"][i])) for i in ix])
                        response=np.array([sum(m*direct_kernel(float(r))*np.exp(-d*r) for m,r in zip(mass,rates)) for d in settings["observation_delays"]])
                        val=float(sum(m*direct_target(float(r),settings["targets"][row["target"]]) for m,r in zip(mass,rates)))
                        maximum["physical_excess"]=max(maximum["physical_excess"],float(np.max(abs(response-obs))-datum["band"]))
                        maximum["physical_target_error"]=max(maximum["physical_target_error"],abs(val-row["inner_interval"][si]))
                        assert np.all(np.isfinite(mass)) and np.all(mass>=0)
                        physical+=1
    assert max(maximum.values())<=settings["lp_replay_tolerance"]
    summaries,decisions=comparisons(result["rows"],settings,spec)
    assert summaries==result["summaries"] and decisions==result["decisions"]
    finest=[r for r in result["rows"] if r["provenance"]=="new_rectangle_control" and r["cells"]==max(settings["cell_counts"])]
    assert max(r["endpoint_bracket"] for r in finest)<=settings["endpoint_bracket_tolerance"]
    audit={"schema":"P4-delayed-rectangle-control-audit-v1","result_sha256":digest(OUT),
           "source_sha256":digest(__file__),"LPs_replayed":checked,"physical_inner_endpoints":physical,
           "independent_observation_checks":source_checks,"independent_ratio_checks":samples,
           "inherited_rows_checked":len(inherited),"max_residuals":maximum,
           "max_finest_endpoint_bracket":max(r["endpoint_bracket"] for r in finest),
           "all_checks_pass":True,"limits":"Same developer alternative time integral; frozen parent rows hash/equality checked, not rerun. No directed rounding or external proof review."}
    AUDIT.write_text(json.dumps(audit,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(audit),flush=True)

if __name__=="__main__":
    main()

