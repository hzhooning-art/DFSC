"""New synthetic spectra, fresh noise and separately calibrated error bounds."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.special import exp1
from scipy.stats import beta

from target_loading_headroom_gate import TARGETS, solve, stress_truth_integral, modulus_for
from target_support_sensitivity import (
    inputs as support_inputs, matrices, digest, pair_with_distinct_noise,
    posterior_with_distinct_noise,
)

HERE=Path(__file__).resolve().parent
SPEC=HERE/"specs"/"ultraslow_calibrated_validation_20261011.json"
OUTPUT=HERE.parent/"results"/"p4_ultraslow_calibrated_validation_20261011.json"
ARCHIVE=OUTPUT.with_suffix(".npz")


def calibrated_bound(blank, confidence):
    blank=np.asarray(blank)
    if blank.size==0 or not np.isfinite(blank).all() or not 0<confidence<1:
        raise ValueError("Finite independent blank samples and confidence in (0,1) required")
    return float(np.max(np.abs(blank))/(1-confidence)**(1/blank.size))


def new_modulus(info):
    if info["kind"] != "uniform_log_rate":
        return modulus_for(info)
    a,b=info["rate_low"],info["rate_high"]
    def evaluate(t):
        x=np.asarray(t,dtype=float)
        safe=np.where(x==0,1.,x)
        transient=np.where(x==0,1.,(exp1(a*safe)-exp1(b*safe))/np.log(b/a))
        value=info["equilibrium"]+info["transient_mass"]*transient
        return float(value) if x.ndim==0 else value
    return evaluate


def binomial_interval(successes, count, confidence=.95):
    if count==0:
        return None
    alpha=1-confidence
    return [0. if successes==0 else float(beta.ppf(alpha/2,successes,count-successes+1)),
            1. if successes==count else float(beta.ppf(1-alpha/2,successes+1,count-successes))]


def inputs():
    spec=json.loads(SPEC.read_text(encoding="utf-8"))
    prior_path=OUTPUT.parent/spec["parent_result"]
    prior=json.loads(prior_path.read_text(encoding="utf-8"))
    for name,expected in prior["source_hashes"].items():
        assert digest(HERE.parent/name)==expected,name
    assert digest(prior_path.with_suffix(".npz"))==prior["witness_archive_sha256"]
    _,_,original,_,_,_,heldout,pilot,schedules=support_inputs()
    assert spec["rate_support"]==[0.,original["rate_support"][1]]
    assert spec["known_initial_modulus"]==1.
    return spec,original,heldout,pilot,schedules


def infer_case(yp,yf,bhat,data_reps,prefix,archive,logs,infeasible):
    diameters={rep:{} for rep in data_reps}
    for rep,data in data_reps.items():
        ap,ep=data["pilot"];af,ef=data["future"]
        for target in TARGETS:
            problem=pair_with_distinct_noise(ap,ep,yp,af,ef,bhat,bhat,data["q"][target],data["f"][target])
            label=f"pair_{prefix}_{rep}_{target}"
            solved=solve(problem,label,archive,logs)
            if solved is None:
                infeasible.append(label)
                diameters[rep][target]=None
            else:
                primal,dual=solved
                diameters[rep][target]=max(0.,-primal-1e-8 if rep=="inner" else -dual+1e-8)
    intervals={}
    data=data_reps["outer"]
    ap,ep=data["pilot"];af,ef=data["future"]
    for target in TARGETS:
        values=[]
        for lower in (True,False):
            label=f"post_{prefix}_{target}_{int(lower)}"
            problem=posterior_with_distinct_noise(ap,ep,yp,af,ef,yf,bhat,bhat,data["q"][target],data["f"][target],lower)
            solved=solve(problem,label,archive,logs)
            if solved is None:
                infeasible.append(label)
                values.append(None)
            else:
                _,dual=solved
                values.append(dual-1e-8 if lower else -dual+1e-8)
        if None in values:
            intervals[target]=None
        else:
            if target=="G20":values=[max(0.,values[0]),min(1.,values[1])]
            intervals[target]={"interval":values,"width":values[1]-values[0]}
    scores={rep:None if any(v[t] is None for t in TARGETS) else
            max(v[t]/limits[t] for t in TARGETS) for rep,v in diameters.items()}
    return diameters,scores,intervals


# Frozen scientific utility scales, verified against the protocol in main.
limits={"G20":.2,"heldout_sigma20":.1}


def summarize(spec,rows,cases):
    summaries={}
    for truth in spec["truths"]:
        trial=[c for c in cases if c["truth"]==truth]
        valid=[c for c in trial if c["calibration_valid"]]
        short=[r for r in rows if r["truth"]==truth and r["horizon"]==8. and r["calibration_valid"]]
        long=[r for r in rows if r["truth"]==truth and r["horizon"]==16.]
        long_valid=[r for r in long if r["calibration_valid"]]
        full_brackets=len(valid)>0 and all(r["score"][rep] is not None for r in short+long_valid for rep in ("inner","outer"))
        ratio=(float(np.mean([r["score"]["outer"] for r in long_valid]) /
                     np.mean([r["score"]["inner"] for r in short])) if full_brackets else None)
        joint=sum(r["joint_covered"] for r in long)
        covered_valid=sum(r["joint_covered"] for r in long_valid)
        useful=sum(r["score"]["outer"] is not None and r["score"]["outer"]<=1 for r in long_valid)
        summaries[truth]={"trial_count":len(trial),"valid_calibration_count":len(valid),
            "calibration_valid_fraction":len(valid)/len(trial),
            "calibration_valid_binomial_95_interval":binomial_interval(len(valid),len(trial)),
            "long_unconditional_joint_covered_count":joint,"long_unconditional_joint_coverage":joint/len(long),
            "long_joint_coverage_binomial_95_interval":binomial_interval(joint,len(long)),
            "long_valid_joint_covered_count":covered_valid,
            "long_valid_joint_coverage":covered_valid/len(valid) if valid else None,
            "long_valid_utility_count":useful,"long_valid_utility_fraction":useful/len(valid) if valid else None,
            "all_valid_pair_brackets_available":full_brackets,
            "long_outer_to_short_inner_mean_ratio_valid_calibration":ratio,
            "valid_mean_worst_diameters":{str(h):{target:float(np.mean([r["diameter"]["outer"][target]
                for r in rows if r["truth"]==truth and r["horizon"]==h and r["calibration_valid"]]))
                for target in TARGETS} for h in (8.,16.)},
            "valid_long_mean_realized_widths":{target:float(np.mean([r["posterior"][target]["width"] for r in long_valid
                if r["posterior"][target] is not None])) for target in TARGETS}}
    candidates=[t for t,v in summaries.items() if v["long_outer_to_short_inner_mean_ratio_valid_calibration"] is not None and
                v["long_outer_to_short_inner_mean_ratio_valid_calibration"]<=spec["gate"]["headroom_ratio_max"]]
    continuous={t for t,v in spec["truths"].items() if v["kind"]!="discrete"}
    passed=all(v["long_valid_joint_coverage"]==1. and
        v["long_unconditional_joint_coverage"]>=spec["gate"]["minimum_observed_joint_coverage_per_truth"] and
        v["long_valid_utility_fraction"]>=spec["gate"]["minimum_valid_calibration_long_utility_fraction_per_truth"] and
        v["all_valid_pair_brackets_available"] for v in summaries.values()) and len(candidates)>=2 and bool(continuous.intersection(candidates))
    return summaries,passed


def main():
    if OUTPUT.exists() or ARCHIVE.exists():raise FileExistsError("Frozen validation output exists")
    started=time.perf_counter()
    spec,original,heldout,pilot,schedules=inputs()
    assert spec["utility_limits"]==limits
    reps={h:matrices(original,0.,pilot,schedules[str(h)],heldout) for h in spec["confirmation_horizons"]}
    archive,logs,rows,cases,infeasible={ },[],[],[],[]
    truth_targets={}
    true_a=spec["calibration"]["amplitude_truth_for_generation_and_evaluation_only"]
    for ti,(truth,details) in enumerate(spec["truths"].items()):
        modulus=new_modulus(details)
        assert abs(modulus(0.)-1)<1e-12
        clean_pilot=stress_truth_integral(pilot[0][0],pilot[0][1],modulus)
        clean_future={h:np.concatenate([stress_truth_integral(t,s,modulus) for t,s in schedules[str(h)]]) for h in reps}
        archive[f"truth{ti}_clean_pilot"]=clean_pilot
        for hi,h in enumerate(reps):archive[f"truth{ti}_clean_future_{hi}"]=clean_future[h]
        truth_targets[truth]={"G20":modulus(20.),"heldout_sigma20":float(stress_truth_integral(np.array([20.]),heldout,modulus)[0])}
        for seed in range(spec["replicates_per_truth"]):
            cid=len(cases)
            def rng(stream):return np.random.default_rng(np.random.SeedSequence([spec["seed_base"],ti,seed,stream]))
            blank=rng(0).uniform(-true_a,true_a,spec["calibration"]["blank_count"])
            bhat=calibrated_bound(blank,spec["calibration"]["upper_confidence"])
            valid=bhat>=true_a
            yp=clean_pilot+rng(1).uniform(-true_a,true_a,len(clean_pilot))
            noise=rng(2).uniform(-true_a,true_a,len(clean_future[8.]))
            archive[f"case{cid}_blank"]=blank
            archive[f"case{cid}_pilot"]=yp
            archive[f"case{cid}_future_noise"]=noise
            for hi,h in enumerate(reps):
                yf=clean_future[h]+noise
                archive[f"case{cid}_future_{hi}"]=yf
                # Inference sees observations, calibrated noise and fixed model only.
                diameters,scores,intervals=infer_case(yp,yf,bhat,reps[h],f"{cid}_{hi}",archive,logs,infeasible)
                for target in TARGETS:
                    if intervals[target] is not None:
                        lo,upper=intervals[target]["interval"]
                        real=truth_targets[truth][target]
                        intervals[target]["contains_truth"]=lo-2e-7<=real<=upper+2e-7
                        if diameters["outer"][target] is not None:
                            assert upper-lo<=diameters["outer"][target]+2e-7
                    if diameters["inner"][target] is not None and diameters["outer"][target] is not None:
                        assert diameters["inner"][target]<=diameters["outer"][target]+2e-7
                joint=all(intervals[t] is not None and intervals[t]["contains_truth"] for t in TARGETS)
                if valid and not joint:
                    raise AssertionError(f"Valid-calibration supported target excluded: {cid}, {h}")
                rows.append({"case":cid,"truth":truth,"truth_index":ti,"seed":seed,"horizon":h,"horizon_index":hi,
                    "calibrated_bound":bhat,"calibration_valid":bool(valid),"diameter":diameters,
                    "score":scores,"posterior":intervals,"joint_covered":bool(joint)})
            cases.append({"id":cid,"truth":truth,"truth_index":ti,"seed":seed,
                          "calibrated_bound":bhat,"calibration_valid":bool(valid)})
            if (seed+1)%16==0:print(json.dumps({"truth":truth,"completed_replicates":seed+1}),flush=True)
    summaries,passed=summarize(spec,rows,cases)
    np.savez_compressed(ARCHIVE,**archive)
    sources=[SPEC,Path(__file__),HERE/"target_support_sensitivity.py",HERE/"target_resource_frontier.py",
        HERE/"target_loading_headroom_gate.py",HERE/"constitutive_response_gate.py",HERE/"calibrated_multiload_gate.py",
        HERE/"offgrid_curvature_envelope_gate.py",SPEC.parent/spec["parent_spec"]]
    joint_all=sum(all(r["joint_covered"] for r in rows if r["case"]==c["id"]) for c in cases)
    valid_count=sum(c["calibration_valid"] for c in cases)
    result={"scope":spec["scope"],"spec_sha256":digest(SPEC),"parent_result_sha256":digest(OUTPUT.parent/spec["parent_result"]),
        "source_hashes":{str(p.relative_to(HERE.parent)):digest(p) for p in sources},"witness_archive_sha256":digest(ARCHIVE),
        "case_count":len(cases),"row_count":len(rows),"solver_count":len(logs),"infeasible_lp_labels":infeasible,
        "duration_seconds":time.perf_counter()-started,"truth_targets":truth_targets,"summaries":summaries,
        "valid_calibration_count":valid_count,"calibration_valid_binomial_95_interval":binomial_interval(valid_count,len(cases)),
        "both_horizons_both_targets_joint_covered_count":joint_all,
        "both_horizons_joint_coverage_binomial_95_interval":binomial_interval(joint_all,len(cases)),
        "validation_gate_pass":passed,"cases":cases,"rows":rows,"lp_replay_logs":logs,"limitations":spec["limitations"]}
    OUTPUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"summaries":summaries,"validation_gate_pass":passed,"valid_calibration_count":valid_count,
        "both_horizons_joint_covered_count":joint_all,"solver_count":len(logs),"infeasible_count":len(infeasible),
        "duration_seconds":result["duration_seconds"]}),flush=True)


if __name__=="__main__":main()
