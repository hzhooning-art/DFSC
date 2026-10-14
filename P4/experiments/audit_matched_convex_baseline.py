"""Read-only replay of matched baseline observations and every saved LP witness."""
import json
from pathlib import Path

import numpy as np

from matched_convex_baseline import (
    HERE, SPEC, OUTPUT, ARCHIVE, METHODS, TARGETS, inputs, representations,
    problem, summarize, digest,
)


def main():
    destination = OUTPUT.with_name("p4_matched_convex_baseline_audit_20261011.json")
    if destination.exists(): raise FileExistsError("Frozen audit exists")
    spec,parent,pspec,original,heldout,pilot,schedules = inputs()
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert result["spec_sha256"] == digest(SPEC)
    assert result["witness_archive_sha256"] == digest(ARCHIVE)
    assert result["parent_result_sha256"] == digest(OUTPUT.parent/spec["parent_result"])
    for name,expected in result["source_hashes"].items(): assert digest(HERE.parent/name) == expected,name
    reps = {h:representations(original,heldout,pilot,schedules[str(h)]) for h in pspec["confirmation_horizons"]}
    maxima = {k:0. for k in ("inequality","equality","negative_mass","stationarity","duality_gap","dual_sign")}
    logs = {v["label"]:v for v in result["lp_replay_logs"]}
    assert len(logs) == result["solver_count"]
    assert len(logs)+len(result["infeasible_lp_labels"]) == 512*8+16*4
    assert len(set(result["infeasible_lp_labels"])) == len(result["infeasible_lp_labels"])
    assert not set(logs).intersection(result["infeasible_lp_labels"])
    labels = set(logs)|set(result["infeasible_lp_labels"])
    expected_labels = set()
    old_rows = {(r["case"],r["horizon_index"]):r for r in parent["rows"]}
    counts = {method:0 for method in ("current_symmetric",)+METHODS}
    replays = {(r["case"],r["horizon_index"]):r for r in result["current_replays"]}
    assert len(replays)==16
    parent_archive = (OUTPUT.parent/spec["parent_result"]).with_suffix(".npz")
    nested = 0; valid_classic = 0; valid_finite = 0; finite_missing = 0
    with np.load(ARCHIVE,allow_pickle=False) as saved, np.load(parent_archive,allow_pickle=False) as observations:
        assert len(saved.files)==4*len(logs)
        for label,log in logs.items():
            cid,hi,method,target,lower = label.split("__")
            cid,hi,lower = int(cid),int(hi),bool(int(lower))
            old = old_rows[cid,hi]
            y = np.r_[observations[f"case{cid}_pilot"],observations[f"case{cid}_future_{hi}"]]
            c,a,d,eq,rhs = problem(reps[old["horizon"]][method],y,old["calibrated_bound"],target,lower)
            x,yu,ye,zl = [saved[f"{label}_{name}"] for name in ("x","yu","ye","zl")]
            residual = {"inequality":max(0.,float(np.max(a@x-d))),
                "equality":float(np.max(np.abs(eq@x-rhs))),"negative_mass":max(0.,float(-x.min())),
                "stationarity":float(np.max(np.abs(c-a.T@yu-eq.T@ye-zl))),
                "duality_gap":abs(float(c@x-d@yu-rhs@ye)),
                "dual_sign":max(0.,float(yu.max()),float(-zl.min()))}
            assert max(residual.values())<=2e-7,(label,residual)
            assert abs(float(c@x)-log["primal"])<=2e-7
            assert abs(float(d@yu+rhs@ye)-log["dual"])<=2e-7
            for k,v in residual.items(): maxima[k]=max(maxima[k],v)
            counts[method]+=1
        assert len(result["rows"])==len(old_rows)==512
        assert len({(r["case"],r["horizon_index"]) for r in result["rows"]})==512
        for row in result["rows"]:
            cid,hi = row["case"],row["horizon_index"]
            old = old_rows[cid,hi]
            for k in ("truth","truth_index","seed","horizon","horizon_index","calibrated_bound","calibration_valid"):
                assert row[k]==old[k]
            assert row["methods"]["current_symmetric"]==old["posterior"]
            for method in METHODS+("current_symmetric",) if (cid,hi) in replays else METHODS:
                values = replays[cid,hi]["posterior"] if method=="current_symmetric" else row["methods"][method]
                for target in TARGETS:
                    pair = [f"{cid}__{hi}__{method}__{target}__{int(lower)}" for lower in (True,False)]
                    expected_labels.update(pair)
                    if not all(label in logs for label in pair):
                        assert any(label in result["infeasible_lp_labels"] for label in pair)
                        assert values[target] is None
                        continue
                    bounds = [logs[pair[0]]["dual"]-1e-8,-logs[pair[1]]["dual"]+1e-8]
                    if target=="G20": bounds=[max(0.,bounds[0]),min(1.,bounds[1])]
                    np.testing.assert_allclose(bounds,values[target]["interval"],atol=1e-12,rtol=0)
                    assert abs(bounds[1]-bounds[0]-values[target]["width"])<1e-12
                    if method=="current_symmetric":
                        np.testing.assert_allclose(bounds,old["posterior"][target]["interval"],atol=2e-7,rtol=0)
                        continue
                    real = parent["truth_targets"][row["truth"]][target]
                    covered = bounds[0]-2e-7<=real<=bounds[1]+2e-7
                    assert covered==values[target]["contains_truth"]
                    if method=="classical_sign_convex":
                        current=old["posterior"][target]["interval"]
                        assert bounds[0]>=current[0]-2e-7 and bounds[1]<=current[1]+2e-7
                        nested+=2
                        if row["calibration_valid"]:
                            assert covered; valid_classic+=1
                    elif row["calibration_valid"]:
                        valid_finite+=1; finite_missing+=int(not covered)
        assert expected_labels==labels
    assert nested==result["nested_endpoint_checks"]
    summaries,passed = summarize(spec,parent,result["rows"])
    assert summaries==result["summaries"] and passed==result["technical_increment_gate_pass"]
    audit={"status":"PASS","result_sha256":digest(OUTPUT),"archive_sha256":digest(ARCHIVE),
        "audit_source_sha256":digest(Path(__file__)),"replayed_successful_lp_count":len(logs),
        "counts":counts,"maximum_residuals":maxima,"nested_endpoint_checks":nested,
        "valid_classical_target_checks":valid_classic,"valid_finite_target_checks":valid_finite,
        "valid_finite_target_exclusions":finite_missing,"current_replay_case_count":len(replays),
        "technical_increment_gate_pass":passed,
        "scope":"Local saved-witness replay, not external blind replication or outward-rounded arithmetic"}
    destination.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit),flush=True)


if __name__=="__main__":main()
