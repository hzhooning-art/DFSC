"""Replay new-seed calibration, reference responses and saved LP witnesses."""
import json

import numpy as np
from scipy.integrate import quad

from constitutive_response_gate import strain, ode_maxwell_reference
from ultraslow_calibrated_validation import (
    SPEC,OUTPUT,ARCHIVE,inputs,digest,calibrated_bound,new_modulus,
    summarize,matrices,pair_with_distinct_noise,posterior_with_distinct_noise,TARGETS,
)
from target_support_sensitivity import zero_safe_basis


def main():
    destination=OUTPUT.with_name("p4_ultraslow_calibrated_validation_audit_20261011.json")
    if destination.exists():raise FileExistsError("Frozen audit exists")
    spec,original,heldout,pilot,schedules=inputs()
    result=json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert digest(SPEC)==result["spec_sha256"]
    assert digest(ARCHIVE)==result["witness_archive_sha256"]
    assert digest(OUTPUT.parent/spec["parent_result"])==result["parent_result_sha256"]
    for name,expected in result["source_hashes"].items():assert digest(SPEC.parents[2]/name)==expected,name
    reps={h:matrices(original,0.,pilot,schedules[str(h)],heldout) for h in spec["confirmation_horizons"]}
    maxima={k:0. for k in ("inequality","equality","negative_mass","stationarity","duality_gap","dual_sign")}
    counts={"pair_inner":0,"pair_outer":0,"posterior":0}
    logs={log["label"]:log for log in result["lp_replay_logs"]}
    assert len(logs)==result["solver_count"]
    assert len(logs)+len(result["infeasible_lp_labels"])==len(result["cases"])*16
    assert not set(logs).intersection(result["infeasible_lp_labels"])
    assert len(result["cases"])==4*spec["replicates_per_truth"]
    valid_coverage_checks=0
    quadrature_errors=[]
    ode_errors=[]
    with np.load(ARCHIVE,allow_pickle=False) as saved:
        for case in result["cases"]:
            cid,ti,seed=case["id"],case["truth_index"],case["seed"]
            a=spec["calibration"]["amplitude_truth_for_generation_and_evaluation_only"]
            def rng(stream):return np.random.default_rng(np.random.SeedSequence([spec["seed_base"],ti,seed,stream]))
            blank=rng(0).uniform(-a,a,spec["calibration"]["blank_count"])
            np.testing.assert_array_equal(saved[f"case{cid}_blank"],blank)
            bhat=calibrated_bound(blank,spec["calibration"]["upper_confidence"])
            assert abs(bhat-case["calibrated_bound"])<1e-15
            assert case["calibration_valid"]==(bhat>=a)
            np.testing.assert_array_equal(saved[f"case{cid}_pilot"],
                saved[f"truth{ti}_clean_pilot"]+rng(1).uniform(-a,a,spec["pilot_count"]))
            noise=rng(2).uniform(-a,a,2*spec["confirmation_count_each"])
            np.testing.assert_array_equal(saved[f"case{cid}_future_noise"],noise)
            for hi,h in enumerate(spec["confirmation_horizons"]):
                np.testing.assert_array_equal(saved[f"case{cid}_future_{hi}"],saved[f"truth{ti}_clean_future_{hi}"]+noise)
        for label,log in logs.items():
            parts=label.split("_")
            cid,hi=map(int,parts[1:3]);h=spec["confirmation_horizons"][hi]
            yp=saved[f"case{cid}_pilot"];case=result["cases"][cid];b=case["calibrated_bound"]
            if parts[0]=="pair":
                rep,target=parts[3],"_".join(parts[4:]);data=reps[h][rep]
                ap,ep=data["pilot"];af,ef=data["future"]
                problem=pair_with_distinct_noise(ap,ep,yp,af,ef,b,b,data["q"][target],data["f"][target])
                counts[f"pair_{rep}"]+=1
            else:
                target,lower="_".join(parts[3:-1]),bool(int(parts[-1]));data=reps[h]["outer"]
                ap,ep=data["pilot"];af,ef=data["future"]
                problem=posterior_with_distinct_noise(ap,ep,yp,af,ef,saved[f"case{cid}_future_{hi}"],b,b,data["q"][target],data["f"][target],lower)
                counts["posterior"]+=1
            c,mat,d,eq,rhs=problem
            x,yu,ye,zl=[saved[f"{label}_{name}"] for name in ("x","yu","ye","zl")]
            residuals={"inequality":max(0.,float(np.max(mat@x-d))),"equality":float(np.max(np.abs(eq@x-rhs))),
                "negative_mass":max(0.,float(-np.min(x))),"stationarity":float(np.max(np.abs(c-mat.T@yu-eq.T@ye-zl))),
                "duality_gap":float(abs(c@x-d@yu-rhs@ye)),"dual_sign":max(0.,float(np.max(yu)),float(-np.min(zl)))}
            assert max(residuals.values())<=2e-7,(label,residuals)
            assert abs(c@x-log["primal"])<=2e-7
            assert abs(d@yu+rhs@ye-log["dual"])<=2e-7
            for k,v in residuals.items():maxima[k]=max(maxima[k],v)
        for row in result["rows"]:
            cid,hi=row["case"],row["horizon_index"]
            for target in TARGETS:
                for rep in ("inner","outer"):
                    label=f"pair_{cid}_{hi}_{rep}_{target}"
                    if label in logs:
                        log=logs[label];expected=max(0.,-log["primal"]-1e-8 if rep=="inner" else -log["dual"]+1e-8)
                        assert abs(expected-row["diameter"][rep][target])<1e-12
                    else:
                        assert label in result["infeasible_lp_labels"] and row["diameter"][rep][target] is None
                post_labels=[f"post_{cid}_{hi}_{target}_{lower}" for lower in (1,0)]
                if all(label in logs for label in post_labels):
                    lo,upper=logs[post_labels[0]]["dual"]-1e-8,-logs[post_labels[1]]["dual"]+1e-8
                    if target=="G20":lo,upper=max(0.,lo),min(1.,upper)
                    np.testing.assert_allclose(row["posterior"][target]["interval"],[lo,upper],atol=1e-12,rtol=0)
                    real=result["truth_targets"][row["truth"]][target]
                    covered=lo-2e-7<=real<=upper+2e-7
                    assert covered==row["posterior"][target]["contains_truth"]
                    if row["calibration_valid"]:
                        assert covered
                        valid_coverage_checks+=1
                    if row["diameter"]["outer"][target] is not None:
                        assert upper-lo<=row["diameter"]["outer"][target]+2e-7
                else:
                    assert row["posterior"][target] is None
            assert row["joint_covered"]==all(row["posterior"][t] is not None and row["posterior"][t]["contains_truth"] for t in TARGETS)
        # Reference responses in another integration order. Discrete spectra
        # additionally use split-history adaptive ODE integration.
        for ti,(truth,details) in enumerate(spec["truths"].items()):
            modulus=new_modulus(details)
            for h in spec["confirmation_horizons"]:
                hi=spec["confirmation_horizons"].index(h)
                offset=0
                for times,segments in schedules[str(h)]:
                    selected=np.array([times[0],times[7],times[19],times[-1]])
                    indices=[0,7,19,len(times)-1]
                    clean=saved[f"truth{ti}_clean_future_{hi}"][offset+np.array(indices)]
                    if details["kind"]=="discrete":
                        alternate=ode_maxwell_reference(selected,segments,details["equilibrium"],np.array(details["weights"]),np.array(details["rates"]))
                        ode_errors.extend(np.abs(alternate-clean).tolist())
                    else:
                        low,high=details["rate_low"],details["rate_high"]
                        def atom_response(t,rate):return zero_safe_basis(np.array([t]),segments,np.array([rate]))[0,1]
                        if details["kind"]=="uniform_rate":
                            alternate=np.array([details["equilibrium"]*strain(np.array([t]),segments)[0]+details["transient_mass"]*
                                quad(lambda rate:atom_response(t,rate),low,high,epsabs=1e-12)[0]/(high-low) for t in selected])
                        else:
                            alternate=np.array([details["equilibrium"]*strain(np.array([t]),segments)[0]+details["transient_mass"]*
                                quad(lambda z:atom_response(t,np.exp(z)),np.log(low),np.log(high),epsabs=1e-12)[0]/np.log(high/low) for t in selected])
                        quadrature_errors.extend(np.abs(alternate-clean).tolist())
                    offset+=len(times)
            assert abs(modulus(20.)-result["truth_targets"][truth]["G20"])<1e-12
    assert max(ode_errors)<1e-9 and max(quadrature_errors)<1e-10
    expected_summaries,expected_pass=summarize(spec,result["rows"],result["cases"])
    assert expected_summaries==result["summaries"] and expected_pass==result["validation_gate_pass"]
    audit={"status":"PASS","result_sha256":digest(OUTPUT),"spec_sha256":digest(SPEC),
        "replayed_successful_lp_count":sum(counts.values()),"counts":counts,"maximum_residuals":maxima,
        "infeasible_lp_report_count":len(result["infeasible_lp_labels"]),
        "calibration_and_noise_seed_replays":len(result["cases"]),"valid_calibration_target_coverage_checks":valid_coverage_checks,
        "independent_ode_response_checks":len(ode_errors),"maximum_ode_error":max(ode_errors),
        "independent_rate_quadrature_response_checks":len(quadrature_errors),"maximum_rate_quadrature_error":max(quadrature_errors),
        "limitations":"Uses shared LP matrix and summary builders; infeasible solver reports are not independently certified; no external review or interval arithmetic"}
    destination.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit),flush=True)


if __name__=="__main__":main()
